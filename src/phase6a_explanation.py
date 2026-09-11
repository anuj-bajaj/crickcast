"""
Phase 6a — live commentary layer

Turns one ball's before/after match state into a short, natural-language
line of cricket commentary that also explains why the model's win
probability just moved. Talks to Groq's chat completions API, with an
OpenRouter fallback on a separate free-tier quota.

Design
------
Free-text generation from an LLM has one structural problem: nothing
stops it from being *wrong* in a way that only shows up once a human
reads it (an invented number, the wrong team credited for a wicket, a
sentence that never finishes). The fix isn't to keep adding instructions
and hoping the model follows all of them at once — every extra rule is
itself a new way for a single call to go wrong. Instead this file keeps
the model's job as small and unambiguous as possible, and treats its
output as untrusted until checked:

  1. build_facts()      — pure Python, no LLM involved. Computes every
                           fact about this ball (score, probability
                           swing, chase context, ...) deterministically,
                           already correctly worded, from the raw match
                           state. The model never computes a number or
                           decides who did what — it only receives
                           facts that are already right.
  2. build_user_prompt() — hands those facts to the model and asks it
                           to weave a handful of them into ONE sentence.
                           No randomized "vary your structure this
                           time" directives and no coin-flip over which
                           name to use — those add variety at the cost
                           of consistency, which is exactly backwards
                           for a project that has to be trustworthy
                           before it's stylish.
  3. validate_output()   — checks the model's text, in code, against
                           the exact facts it was given: a stray
                           ALL-CAPS token, a number that doesn't trace
                           back to a fact, or an unfinished sentence
                           gets flagged. A flagged response gets one
                           corrective regeneration; if that also fails,
                           the caller gets a plain sentence assembled
                           directly from the facts instead of the
                           model's prose, so a bad LLM output can never
                           reach the screen.

Set GROQ_API_KEY as an environment variable before running (and
OPENROUTER_API_KEY, optional, for the fallback provider):
  Windows (PowerShell):  $env:GROQ_API_KEY="your_key_here"
"""

import logging
import os
import random
import re

import requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1 — facts
# ---------------------------------------------------------------------------
#
# Anything that has exactly one correct answer (which team gets credited,
# how many balls are left in the over, whether a rate rose or fell) is
# resolved here in code, never left for the model to work out. Word choice
# within a fact (which of a few equally-valid verbs to use) is the only
# thing randomized, via random.choice() over a small curated pool.

# A single, consistent naming rule: use the real team name whenever it's
# known, otherwise a plain generic. Used for TEAM-LEVEL facts only (Chase
# state, Risk framing, Streak, Momentum) — facts about the whole side's
# situation across many balls, where a team name genuinely fits.
def _batting_side(batting_team):
    return batting_team or "the batting side"


def _bowling_side(bowling_team):
    return bowling_team or "the bowling side"


# A single ball's action is one player's doing, not the team's — "India
# finds the gap for four" reads like the team collectively did it, which
# is exactly the kind of subject a viewer has to stop and mentally
# resolve. A single-ball action always gets a person-level generic
# instead, regardless of whether a team name is known.
def _batting_actor():
    return "the batsman"


def _bowling_actor():
    return "the bowler"


# Real commentator phrasing for each delivery type, split by which side
# actually performs the action so the correct subject is baked directly
# into the phrase — a bare fragment ("finds the gap for four") forces
# whoever restates it to guess an actor, and a guess is exactly where a
# wrong-side attribution comes from. Every phrase here keeps the named
# actor as the one actually doing the thing described — no phrase where
# the literal subject should be the ball or the shot rather than the
# player (e.g. "races away to the fence" describes the BALL racing, not
# the batsman, so it's deliberately not in this pool).
_BATTING_EVENT_PHRASES = {
    "single": [
        "{actor} works one away", "{actor} rotates strike for a single",
        "{actor} tucks it away for a single",
    ],
    # "a couple" rather than a bare "two" — a bare number reads too
    # easily as a wicket count once it's echoed back as part of a
    # previous-ball reference on a later call.
    "two": [
        "{actor} runs a quick two", "{actor} finds the gap for a couple",
    ],
    "three": [
        "{actor} hustles through for three", "{actor} finds the gap for three",
    ],
    "four": [
        "{actor} finds the gap for four", "{actor} times it to perfection for four",
        "{actor} drives it away for four", "{actor} cracks one through the line for four",
    ],
    "six": [
        "{actor} launches it into the stands", "{actor} clears the ropes with ease",
        "{actor} goes big for a maximum",
    ],
}
_BOWLING_EVENT_PHRASES = {
    "dot_ball": [
        "{actor} keeps it tight, dot ball", "{actor} dries up the scoring",
        "{actor} holds firm, no run",
    ],
    "wicket": [
        "{actor} strikes", "{actor} gets the breakthrough", "{actor} takes a big wicket",
    ],
    "wide": [
        "{actor} strays down leg, wide called", "{actor} loses the line, wide conceded",
    ],
    "noball": [
        "{actor} oversteps, no-ball called", "{actor} strays over the line, no-ball",
    ],
}
# Short noun form of each event, used only to build the "after the
# {noun} on the previous ball, ..." lead-in below — deliberately
# separate from the full sentences above, which don't fit into that
# template ("after the strays down leg on the previous ball" isn't a
# phrase; "after the wide on the previous ball" is).
_EVENT_NOUN = {
    "dot_ball": "dot ball", "single": "single", "two": "couple of runs",
    "three": "three runs", "four": "boundary", "six": "six",
    "wicket": "wicket", "wide": "wide", "noball": "no-ball",
}


def _delivery_phrase(raw_event):
    """One curated phrase for a delivery type, with the correct
    person-level actor already named as its subject."""
    if raw_event in _BATTING_EVENT_PHRASES:
        return random.choice(_BATTING_EVENT_PHRASES[raw_event]).format(actor=_batting_actor())
    if raw_event in _BOWLING_EVENT_PHRASES:
        return random.choice(_BOWLING_EVENT_PHRASES[raw_event]).format(actor=_bowling_actor())
    return None


def _is_previous_ball_worth_mentioning(previous_event, raw_event):
    """A real commentator only calls back to the previous ball when
    there's an actual thread to follow — most consecutive balls (a dot
    then a single, a single then a dot) have no such thread, and
    mentioning "the previous ball" on every single call is exactly what
    made it feel like noise. This fires only for the handful of cases
    where the callback is genuinely earned:
      - the previous ball was an extra (wide/no-ball) — it didn't count
        toward the over and, for a no-ball, sets up a free hit, so it's
        rarely irrelevant context.
      - recovering immediately after a wicket with a scoring shot.
      - back-to-back boundaries (building momentum) or back-to-back
        wickets (a bowling spell).
    """
    if previous_event is None:
        return False
    if previous_event in ("wide", "noball"):
        return True
    if previous_event == "wicket" and raw_event in ("single", "two", "three", "four", "six"):
        return True
    if previous_event in ("four", "six") and raw_event in ("four", "six"):
        return True
    if previous_event == "wicket" and raw_event == "wicket":
        return True
    return False


_PROBA_WORDS = {
    "tier1": {"UP": ["ticks up", "edges higher", "inches up"],
              "DOWN": ["ticks down", "edges lower", "dips a touch"]},
    "tier2": {"UP": ["climbs a little", "eases higher", "creeps up"],
              "DOWN": ["eases back", "slips", "dips back"]},
    "tier3": {"UP": ["swings up", "jumps", "climbs sharply"],
              "DOWN": ["swings down", "slides", "drops"]},
    "tier4": {"UP": ["surges", "rockets up", "soars"],
              "DOWN": ["crashes", "plummets", "collapses"]},
}
_RATE_WORDS = {"UP": ["climbs", "rises", "edges higher"], "DOWN": ["eases", "drops", "dips"]}
_SCORE_VERBS = [
    "lifts the total to", "moves the score to", "takes the total to", "brings the total to",
]
# One grammatical shape per tier (a full clause, always endable with a
# period) — no separate "fragment vs. clause" bookkeeping needed, unlike
# a pool that mixes both forms.
#
# Five tiers instead of three, each with several variants: when the
# probability sits in a broad band (e.g. 65-100%) for a long stretch of
# play, a 3-tier/2-variant pool cycles through the same couple of lines
# over and over and starts to feel copy-pasted. More, narrower tiers
# means the wording actually shifts as the game genuinely tightens or
# opens up, not just at the three original cutoffs — and each tier
# having more variants means even a long spell inside one tier doesn't
# repeat the same line every couple of balls. Ordered highest threshold
# first; the first matching threshold wins.
_CHASE_STATE_TIERS = [
    (85, [
        "{team} are cruising towards this target",
        "this is turning into a comfortable stroll for {team}",
        "{team} look to have this all but wrapped up",
    ]),
    (65, [
        "{team} are firmly on top in this chase",
        "it's a strong position for {team}",
        "{team} hold a clear upper hand here",
    ]),
    (45, [
        "this chase remains a genuine contest",
        "it's finely balanced right now",
        "neither side has a clear edge at this point",
    ]),
    (25, [
        "{team} have it all still to do from here",
        "{team} face an uphill task now",
        "the pressure is mounting on {team}",
    ]),
    (0, [
        "{team} are up against it in a big way",
        "this is fast slipping away from {team}",
        "{team} would need something special from here",
    ]),
]

# A finished chase isn't a "how is this going" framing anymore — it's a
# result, and it gets a plain, confirmative statement instead of a
# probability-band description. Two separate pools rather than one,
# since the two outcomes need different subjects: the batting side is
# the one who wins, but a loss is really the bowling/defending side
# completing a successful defense, not the batting side "losing" as an
# action of their own.
_CHASE_STATE_WON_PHRASES = [
    "{batting} have won this match",
    "{batting} get over the line to seal the chase",
    "{batting} complete the chase in style",
    "that's the winning moment for {batting}",
    "{batting} close out the win",
]
_CHASE_STATE_LOST_PHRASES = [
    "{bowling} defend their total and complete the win",
    "it's all over — {bowling} have won this contest",
    "{batting} fall short as {bowling} take the win",
    "{bowling} close out the match, {batting} left short",
    "the chase ends there for {batting}, {bowling} celebrate the win",
]


def build_facts(ctx: dict) -> dict:
    """Deterministically derives every fact about this ball from the raw
    match state. The only randomness is which pre-approved word/phrase
    variant is picked — never what a fact actually says. Returns a dict
    with "lines" (the fact lines to hand the model, in prompt order) plus
    a handful of raw values validate_output() and the deterministic
    fallback both need directly. Pure function — no I/O, fully
    unit-testable on its own.
    """
    proba_before = ctx["proba_before"]
    proba_after = ctx["proba_after"]
    swing = ctx["swing"]
    before_pct = proba_before * 100
    after_pct = proba_after * 100
    swing_points = abs(swing) * 100
    direction = "UP" if swing > 1e-9 else "DOWN" if swing < -1e-9 else "UNCHANGED"

    balls_remaining = ctx.get("balls_remaining", 0)
    cum_wickets = ctx["cum_wickets"]
    batting_team = ctx.get("batting_team")
    bowling_team = ctx.get("bowling_team")
    target = ctx["cum_runs"] + ctx.get("runs_required", 0)

    # --- Is the match actually decided already? Beyond /predict's own
    # short-circuit cases (target reached; bowled out or overs run out
    # while still short), a chase can become mathematically impossible
    # BEFORE either of those literally happens — needing 14 off the
    # last 2 balls, say, can't be done even by hitting a six off every
    # remaining ball (6 x 2 = 12 max), regardless of what percentage
    # the trained model's own probability estimate still shows (that's
    # a statistical estimate over historical matches, not a hard rule —
    # it can and does still return a small nonzero number here). Chase
    # state should say so the moment it becomes true, not wait for the
    # innings to literally end to admit it. Computed once, up front,
    # since it changes how BOTH the Chase equation fact and Chase state
    # fact should read below — "needs 5 more off 0 balls" and
    # "comfortable stroll" are both the wrong kind of statement once
    # this is no longer an ongoing forecast but an actual (or already
    # certain) result.
    runs_required_val = ctx.get("runs_required")
    required_run_rate_val = ctx.get("required_run_rate")
    match_won = runs_required_val is not None and runs_required_val <= 0
    mathematically_impossible = (
        not match_won and (
            (runs_required_val is not None and runs_required_val > 6 * balls_remaining)
            or (required_run_rate_val is not None and required_run_rate_val > 36)
        )
    )
    match_lost = not match_won and (cum_wickets >= 10 or balls_remaining <= 0 or mathematically_impossible)
    match_decided = match_won or match_lost

    # --- Probability: always included, verb picked here.
    proba_steady = swing_points < 0.05
    if proba_steady:
        proba_fact = f"Probability: holds steady around {after_pct:.0f}% (barely moved — don't present this as a from/to change)"
        proba_before_fmt = proba_after_fmt = f"{after_pct:.0f}"
        proba_verb = "holds steady around"
    else:
        if round(before_pct, 1) == round(after_pct, 1):
            # Rounding collision at 1 decimal (e.g. 33.71% and 33.72%
            # both read as "33.7%") — escalate to 2 decimals so the two
            # numbers are visibly different.
            proba_before_fmt, proba_after_fmt = f"{before_pct:.2f}", f"{after_pct:.2f}"
        else:
            proba_before_fmt, proba_after_fmt = f"{before_pct:.1f}", f"{after_pct:.1f}"
        tier = "tier1" if swing_points < 3 else "tier2" if swing_points < 8 else "tier3" if swing_points < 15 else "tier4"
        proba_verb = random.choice(_PROBA_WORDS[tier][direction if direction != "UNCHANGED" else "UP"])
        proba_fact = f"Probability: the win probability {proba_verb} from {proba_before_fmt}% to {proba_after_fmt}%"

    # --- Chase equation / required run rate. In the death overs, real
    # commentators state what's needed off what's left rather than an
    # abstract rate — and only the batting side ever "needs" runs, so
    # naming it explicitly removes any guesswork about which side this
    # describes. Suppressed entirely once the match is already decided
    # ("needs 5 more off 0 balls" is nonsensical once there are no
    # balls left to bowl it off) — Chase state's confirmative result
    # line covers that instead.
    in_death_overs = balls_remaining <= 60
    runs_required = ctx.get("runs_required", "unknown")
    after_rate = ctx.get("required_run_rate")
    before_rate = ctx.get("required_run_rate_before")
    if match_decided:
        rate_fact = None
    elif in_death_overs:
        rate_fact = f"Chase equation: {_batting_side(batting_team)} needs {runs_required} more off {balls_remaining} balls (never state a required-run-rate number this response)"
    elif before_rate is None or before_rate == after_rate:
        rate_fact = None
    else:
        rate_dir = "UP" if after_rate > before_rate else "DOWN"
        verb = random.choice(_RATE_WORDS[rate_dir])
        rate_fact = f"Required run rate: {verb} to {after_rate:.1f} ({'harder' if rate_dir == 'UP' else 'easier'} chase now)"

    # --- Score / wickets: only present when they actually changed this
    # ball (a dot ball, wide, or no-ball with no wicket changes neither).
    cum_runs = ctx["cum_runs"]
    cum_runs_before = ctx.get("cum_runs_before")
    if cum_runs_before is not None and cum_runs == cum_runs_before:
        score_fact = None
    else:
        score_fact = f"Score: {random.choice(_SCORE_VERBS)} {cum_runs}"

    cum_wickets_before = ctx.get("cum_wickets_before")
    if cum_wickets_before is not None and cum_wickets != cum_wickets_before:
        wickets_fact = (
            f"Wickets: a wicket falls, {cum_wickets} down now "
            f"(wickets down means lost — never phrase this as \"down to {cum_wickets} wickets\", that would mean {cum_wickets} remain)"
        )
    else:
        wickets_fact = None

    # --- Milestone: team total crossing 50/100/150/200 on THIS ball.
    milestone_fact = None
    if cum_runs_before is not None:
        for m in (200, 150, 100, 50):
            if cum_runs_before < m <= cum_runs:
                milestone_fact = f"Milestone: {_batting_side(batting_team)}'s total just crossed {m} this ball — worth a mention if it fits naturally"
                break

    # --- Risk framing: wickets in hand vs. overs left, only surfaced
    # once it's genuinely tight (4 or fewer in hand), past the Powerplay,
    # and while there's still enough of the innings left for it to
    # actually be a constraint — with only a ball or two of the innings
    # remaining, "wickets in hand" stops being a real limit on how much
    # risk a side can take (they're swinging at everything regardless),
    # so this stops firing inside the last 2 overs.
    overs_bowled = (120 - balls_remaining) / 6
    wickets_in_hand = 10 - cum_wickets
    # Real cricket-over notation: an over has 6 balls, so the fraction
    # after the point only ever runs 0-5, never 6-9 — floor-division and
    # modulo give that; plain float division (balls_remaining / 6) does
    # not and can silently produce an impossible ".7" or ".8".
    overs_remaining_str = f"{balls_remaining // 6}.{balls_remaining % 6}"
    wicket_word = "wicket" if wickets_in_hand == 1 else "wickets"
    if not match_decided and wickets_in_hand <= 4 and overs_bowled >= 6 and balls_remaining > 12:
        if in_death_overs:
            # The Chase equation fact already states "off {balls_remaining}
            # balls" in the death overs — restating the exact same span
            # again here as "{overs}.{balls} overs left" is the same
            # number said twice in two different units within one
            # sentence ("needs 31 more off 15 balls ... with 2.3 overs
            # left"), which is where the redundancy actually is.
            risk_fact = f"Risk framing: only {wickets_in_hand} {wicket_word} in hand — a real constraint on how many risks {_batting_side(batting_team)} can take"
        else:
            risk_fact = f"Risk framing: only {wickets_in_hand} {wicket_word} in hand with {overs_remaining_str} overs left — a real constraint on how many risks {_batting_side(batting_team)} can take"
    else:
        risk_fact = None

    # --- Streak / momentum: genuine multi-ball trends a single
    # before/after snapshot can't capture on its own, supplied by the
    # frontend's own ball-by-ball history (see computeStreakFact in
    # App.jsx).
    #
    # A wicket/wide/no-ball streak is NOT suppressed once the match is
    # decided — unlike the others, it isn't a forward-looking "pressure
    # is building" trend, it's a plain description of what just
    # happened, and for a wicket streak specifically it's very often
    # THE reason the match just ended (a cluster of wickets that
    # finished the innings). Suppressing it here was a real, observed
    # bug: the exact wicket that ends a match is exactly the one ball
    # where "2 wickets in 2 balls" is the most useful, correct framing,
    # and losing it meant the previous-ball callback below (which is
    # only suppressed when a matching streak fires) fired instead,
    # producing a sentence that described the SAME wicket twice ("after
    # the wicket on the previous ball, the bowler takes a big wicket").
    # Dot/boundary/four/six streaks keep the original reasoning: their
    # wording ("pressure building", "under real pressure") implies an
    # ongoing trend that still matters for what happens next, which
    # stops being true the moment there's no "next" left to play.
    streak_type = ctx.get("streak_type")
    if match_decided and streak_type in ("dot", "boundary", "four", "six"):
        streak_type = None
    streak_count = ctx.get("streak_count")
    if streak_type == "dot" and streak_count:
        streak_fact = f"Streak: {streak_count} dot balls in a row — pressure building on {_batting_side(batting_team)}"
    elif streak_type == "boundary" and streak_count:
        streak_fact = f"Streak: {streak_count} boundaries in a row — {_bowling_side(bowling_team)} under real pressure"
    elif streak_type == "four" and streak_count:
        streak_fact = f"Streak: {streak_count} fours on the trot — {_bowling_side(bowling_team)} under real pressure"
    elif streak_type == "six" and streak_count:
        streak_fact = f"Streak: {streak_count} consecutive sixes — {_bowling_side(bowling_team)} being taken apart"
    elif streak_type == "wicket" and streak_count:
        streak_fact = f"Streak: {streak_count} wickets in {streak_count} balls — {_batting_side(batting_team)} in serious trouble"
    elif streak_type == "wide" and streak_count:
        streak_fact = f"Streak: {streak_count} wides in a row — {_bowling_side(bowling_team)} completely off target"
    elif streak_type == "noball" and streak_count:
        streak_fact = f"Streak: {streak_count} no-balls in a row — {_bowling_side(bowling_team)} struggling for control"
    else:
        streak_fact = None

    rate_momentum = ctx.get("rate_momentum") if not in_death_overs and not match_decided else None
    if rate_momentum == "rising":
        momentum_fact = f"Momentum: the required rate has been creeping up over the last few overs — pressure building on {_batting_side(batting_team)}"
    elif rate_momentum == "falling":
        momentum_fact = f"Momentum: the required rate has been easing over the last few overs — {_batting_side(batting_team)} clawing it back"
    else:
        momentum_fact = None

    # --- Delivery, with an optional previous-ball lead-in folded
    # directly in when (and only when) there's a genuine narrative
    # thread — see _is_previous_ball_worth_mentioning. Building the
    # combined phrase here, in code, guarantees the ordering the moment
    # it's read by a person following along: "after {event} on the
    # previous ball, {this ball's action}" — never the reverse, and
    # never left for the model to decide where it goes.
    #
    # A no-ball gets its own framing rather than the generic template:
    # its very next delivery is automatically a free hit, and saying so
    # here IS the free-hit fact — there used to be a separate "Free
    # hit: ..." line stating the same thing a second time, which was
    # pure redundancy once the lead-in already exists.
    #
    # A single "after the wicket on the previous ball" callback also
    # stops making sense once it's really the THIRD wicket in a row
    # (say) rather than a one-off — at that point the Streak fact above
    # already IS the correct way to frame it ("3 wickets in 3 balls"),
    # and pairing both would say the same thing twice in two different
    # shapes. Whenever a same-type streak just fired, the lead-in is
    # skipped in favor of it — same reasoning for back-to-back wides
    # and no-balls.
    raw_event = ctx.get("raw_event")
    previous_event = ctx.get("previous_event")
    current_phrase = _delivery_phrase(raw_event)
    streak_covers_this_callback = streak_type in ("wicket", "wide", "noball", "four", "six")
    if current_phrase and not streak_covers_this_callback and _is_previous_ball_worth_mentioning(previous_event, raw_event):
        if previous_event == "noball":
            delivery_phrase = f"on the free hit that followed the no-ball, {current_phrase}"
        else:
            prev_noun = _EVENT_NOUN.get(previous_event)
            delivery_phrase = f"after the {prev_noun} on the previous ball, {current_phrase}"
    else:
        delivery_phrase = current_phrase
    delivery_fact = f"Delivery: {delivery_phrase}" if delivery_phrase else None

    # --- Span: how many legal deliveries this update actually covers —
    # without this, a multi-ball jump (Auto-predict off, or a big Alter
    # State edit) would otherwise read as a single event. A wide/no-ball
    # (balls_elapsed == 0) deliberately gets NO span fact here — "0 legal
    # deliveries" is an internal bookkeeping detail, not something a real
    # commentator ever says out loud, and the Delivery fact's own
    # wording ("wide called" / "no-ball called") already makes clear no
    # legal delivery was bowled without needing a second, more literal
    # fact to say the same thing.
    balls_elapsed = ctx.get("balls_elapsed")
    if balls_elapsed is not None and balls_elapsed > 1:
        span_fact = f"Span: {balls_elapsed} deliveries happened since the last update, not one — frame this as a stretch of play, not a single ball"
    else:
        span_fact = None

    # --- Chase state: which "how is this going" framing is earned,
    # based on the CURRENT probability (a small dip at 82% is still
    # comfortable, not "slipping away") — UNLESS the match is actually
    # decided already (match_won/match_lost, computed up front), in
    # which case a probability-band description ("comfortable stroll")
    # is the wrong kind of statement entirely; this is now a result,
    # not an ongoing forecast, and gets a plain confirmative line
    # instead.
    if match_won:
        chase_state_phrase = random.choice(_CHASE_STATE_WON_PHRASES).format(
            batting=_batting_side(batting_team)
        )
    elif match_lost:
        chase_state_phrase = random.choice(_CHASE_STATE_LOST_PHRASES).format(
            batting=_batting_side(batting_team), bowling=_bowling_side(bowling_team)
        )
    else:
        # Checked highest threshold first so the first tier the
        # percentage clears is the one used.
        for threshold, phrases in _CHASE_STATE_TIERS:
            if after_pct >= threshold:
                chase_state_phrase = random.choice(phrases)
                break
        if "{team}" in chase_state_phrase:
            chase_state_phrase = chase_state_phrase.format(team=_batting_side(batting_team))

    # --- Assemble. Probability, Chase state, and Target are always
    # present. Score and the rate/chase-equation fact are both routinely
    # true on the very same ball — Delivery already says what happened,
    # so stacking both supporting stats on top of it every single time
    # produces the same rigid shape over and over ("does X, taking the
    # total to Y, needing Z off W"). When both are available, keep only
    # one at random, so the commentary leans on a different supporting
    # detail from one ball to the next rather than reciting the full set
    # every time.
    supporting_facts = [f for f in (score_fact, rate_fact) if f]
    if len(supporting_facts) == 2:
        supporting_facts = [random.choice(supporting_facts)]

    core_facts = [delivery_fact, span_fact, wickets_fact] + supporting_facts
    MAX_FLAVOR_FACTS = 2
    flavor_by_priority = [milestone_fact, streak_fact, momentum_fact, risk_fact]
    flavor_facts = [f for f in flavor_by_priority if f][:MAX_FLAVOR_FACTS]

    # Shuffling the order these are handed over in (rather than always
    # the same fixed sequence) nudges the model away from settling into
    # one habitual sentence template, on top of the fact-count reduction
    # above.
    middle = [f for f in core_facts + flavor_facts if f]
    random.shuffle(middle)

    lines = [proba_fact] + middle
    lines.append(f"Chase state: {chase_state_phrase}")
    lines.append(f"Target: {target} (fixed all innings — mention at most once, only if truly needed)")

    return {
        "lines": lines,
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "delivery_phrase": delivery_phrase,
        "chase_state_phrase": chase_state_phrase,
        "proba_verb": proba_verb,
        "proba_before_fmt": proba_before_fmt,
        "proba_after_fmt": proba_after_fmt,
        "proba_steady": proba_steady,
        "score_fact": score_fact,
        "wickets_fact": wickets_fact,
        "target": target,
    }


def build_user_prompt(ctx: dict) -> str:
    """Public wrapper kept separate from build_facts so callers (and
    tests) that only need the prompt text don't have to know about the
    richer facts dict generate_explanation() uses internally."""
    facts = build_facts(ctx)
    return "\n".join(f"- {line}" for line in facts["lines"]) + "\n"


EXPLANATION_SYSTEM_PROMPT = """You are a knowledgeable T20 cricket commentator writing one short line of live commentary for a fan following a win-probability tracker.

You're given a list of facts about this ball, already correct and worded the way a commentator would say them. Weave 2-4 of them into ONE flowing sentence. You don't decide what happened, choose vocabulary for numbers, or pick which side did what — that's all decided for you. Your only job is fluency: connect the given facts naturally, and imply what they mean rather than defining them ("keeps it tight, dot ball, pressure building" — not "a dot ball, meaning no runs were scored").

A routine ball gets one clean sentence. A genuinely big moment (a wicket, a six, a huge swing) can open with a short, punchy reaction first:
  "Massive strike! That six sends the total to 155 as the win probability leaps from 81% to 90%."

Rules:
- Each fact is written as "Label: content" — the label before the colon is only there to help you tell facts apart. Never output the label word itself.
- For "Probability", "Delivery", "Score", "Chase equation"/"Required run rate", and "Chase state": use the exact wording given, word for word. These are pre-chosen commentator phrasing, not something to paraphrase.
- The "Probability" fact always keeps both numbers together ("from X% to Y%") — never trim it to just the destination number.
- Only present a fact as the CAUSE of the swing if it's the event of this ball or is explicitly marked as changed.
- The "Delivery" fact occasionally already opens with a lead-in like "after the wicket on the previous ball, ..." or "on the free hit that followed the no-ball, ..." when there's a genuine connection to this ball — that lead-in is pre-decided and part of the exact wording, not something to add, remove, or rephrase yourself.
- Mention the target at most once, and only if it helps explain the stakes.
- Never invent a name, number, or detail that isn't in the facts below.
- Join facts with real connecting words ("and", "as", "while", "with") — never a bare comma standing in for a missing verb.
- Only use a contrastive connector ("but", "however", "yet") when the event and the probability move genuinely pull in opposite directions (e.g. a wicket falls yet the probability barely moves). An extra conceded, a boundary hit, or a wicket that moves the probability the way you'd expect it to is NOT a contradiction — connect those with "and"/"as"/"while", never "but".
- Never write a word in ALL CAPS, and always finish the sentence.
- Vary how you connect and order facts from one response to the next — don't settle into reciting the same sentence template every ball.

"Delivery" already correctly names its subject as "the batsman" or "the bowler" — never swap in a team name or a different actor. Team names only belong in facts that are already written with one (Chase equation, Chase state, Risk framing, Streak, Momentum).
"""


# ---------------------------------------------------------------------------
# Stage 3 — validation
# ---------------------------------------------------------------------------
#
# Checks the model's own output, in code, against the facts it was handed.
# Three checks, each targeting a real failure mode free-text generation has
# no structural defense against otherwise: an ALL-CAPS token leaking
# through, a number that doesn't trace back to any given fact, and an
# unfinished sentence. False positives here are cheap (one corrective
# regeneration, or a plain but always-correct fallback) — false negatives
# are the real risk, so these are tuned to over-flag rather than under-flag.

_ALL_CAPS_RE = re.compile(r"\b[A-Z]{4,}\b")
_NUMBER_RE = re.compile(r"\d+\.?\d*")
_TITLE_CASE_RE = re.compile(r"\b[A-Z][a-z]+\b")

# Capitalized vocabulary the commentary legitimately uses that isn't a
# team name — extend this rather than loosen the check itself.
_ALLOWED_CAPITALIZED_WORDS = {
    "Massive", "Big",
    "One", "Two", "Three", "Four", "Five", "Six",
}


def _strip_sentence_leads(text: str) -> str:
    """Drops the first word of every sentence before the proper-noun scan
    below — ordinary sentence-initial capitalization (including a team
    name that legitimately opens the sentence) is the single biggest
    source of false positives, and dropping it costs nothing: a
    hallucinated name showing up ONLY as a sentence's first word and
    nowhere else is not a realistic failure mode."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    remainders = []
    for sentence in sentences:
        parts = sentence.split(" ", 1)
        if len(parts) > 1:
            remainders.append(parts[1])
    return " ".join(remainders)


def validate_output(text: str, facts: dict) -> list:
    """Returns a list of issue codes (empty means valid). Never raises —
    a validator that can crash the pipeline it's protecting would defeat
    the point."""
    issues = []
    if not text or not text.strip():
        return ["empty_output"]

    if _ALL_CAPS_RE.search(text):
        issues.append("all_caps_leak")

    # A truncated response (a reasoning model's hidden thinking step can
    # eat into its answer budget) is just as unusable as a wrong one.
    # Cheap, reliable signal: real commentary always ends on terminal
    # punctuation.
    stripped = text.strip()
    if stripped[-1] not in ".!?\"'\u201d\u2019)":
        issues.append("incomplete_output")

    # Label leak: each fact line is "Label: content" — the label is
    # only for the model's own categorization and must never appear in
    # its answer (a real, observed bug: "Delivery: after the wide on
    # the previous ball, ..." was echoed back label and all, reading as
    # if the model pasted its own prompt into the reply). Checked
    # dynamically against whatever labels actually appear in THIS
    # call's facts, rather than a hardcoded list, so it can't drift out
    # of sync with build_facts as facts are added or renamed later.
    fact_labels = {line.split(":", 1)[0] for line in facts["lines"] if ":" in line}
    for label in fact_labels:
        if re.search(rf"\b{re.escape(label)}\s*:", text):
            issues.append(f"label_leak:{label}")

    # Number hallucination: every number in the output must trace back
    # to a number that appeared somewhere in the facts. Small numbers
    # (<=6) are exempt — extremely common as connective phrasing (over
    # counts, "two balls left") and essentially never a hallucinated
    # statistic.
    facts_text = " ".join(facts["lines"])
    allowed_numbers = set(_NUMBER_RE.findall(facts_text))

    def _number_is_allowed(num_str):
        if num_str in allowed_numbers:
            return True
        try:
            value = float(num_str)
        except ValueError:
            return True
        if value <= 6:
            return True
        for allowed in allowed_numbers:
            try:
                if abs(float(allowed) - value) < 0.15:
                    return True
            except ValueError:
                continue
        return False

    bad_numbers = [n for n in _NUMBER_RE.findall(text) if not _number_is_allowed(n)]
    if bad_numbers:
        issues.append("unrecognized_number:" + ",".join(sorted(set(bad_numbers))))

    # Names: any capitalized word must match a token from the known
    # batting/bowling team names, or the small curated vocabulary list.
    known_name_tokens = set()
    for team in (facts.get("batting_team"), facts.get("bowling_team")):
        if team:
            known_name_tokens.update(team.split())

    scan_text = _strip_sentence_leads(text)
    for word in _TITLE_CASE_RE.findall(scan_text):
        if word in _ALLOWED_CAPITALIZED_WORDS or word in known_name_tokens:
            continue
        issues.append(f"unrecognized_name:{word}")

    return issues


def _deterministic_fallback_sentence(facts: dict) -> str:
    """A plain, always-correct sentence built directly from the facts,
    with no LLM involved — used only when a provider's output fails
    validation twice in a row. Deliberately unexciting: correctness
    beats style the moment style can't be trusted."""
    parts = []
    if facts.get("delivery_phrase"):
        parts.append(facts["delivery_phrase"][0].upper() + facts["delivery_phrase"][1:] + ".")
    elif facts.get("wickets_fact"):
        parts.append("A wicket falls.")
    if facts["proba_steady"]:
        proba_clause = f"Win probability holds steady around {facts['proba_after_fmt']}%"
    else:
        proba_clause = f"Win probability {facts['proba_verb']} from {facts['proba_before_fmt']}% to {facts['proba_after_fmt']}%"
    parts.append(proba_clause + ".")
    chase_phrase = facts["chase_state_phrase"]
    parts.append(chase_phrase[0].upper() + chase_phrase[1:] + ".")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

# Built lazily (and re-checked on every call) rather than at import time, so
# a missing/invalid GROQ_API_KEY doesn't take the whole module — and by
# extension the FastAPI app that imports it — down before it even starts.
_client = None


def _get_client():
    global _client
    if _client is None:
        # .strip() defensively — a .env file saved on Windows can leave a
        # trailing \r on the value, which silently corrupts the
        # Authorization header.
        api_key = (os.environ.get("GROQ_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        # max_retries=0: with an OpenRouter fallback available, retrying
        # the SAME rate-limited provider before giving up is pure wasted
        # time — falling over to OpenRouter's independent quota
        # immediately is the real "retry".
        _client = Groq(api_key=api_key, timeout=8.0, max_retries=0)
    return _client


def _call_groq(user_prompt: str) -> str:
    """Raises on ANY failure — missing/invalid key, network error, rate
    limit, or empty content — rather than degrading to None itself.
    generate_explanation() decides what to do about a failure; this
    function's only job is "did Groq actually produce usable text"."""
    client = _get_client()
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        # gpt-oss-20b is a reasoning model and spends part of its
        # completion-token budget on a hidden chain-of-thought before
        # writing the actual answer — reasoning_effort="low" keeps that
        # step short, and max_tokens is set well above what the visible
        # answer alone needs so a heavier-than-usual reasoning pass
        # doesn't silently eat the whole budget and return empty content.
        reasoning_effort="low",
        max_tokens=1200,
        temperature=0.6,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError(
            f"Groq returned empty content (finish_reason="
            f"{getattr(response.choices[0], 'finish_reason', 'unknown')})"
        )
    return content.strip()


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# openrouter/free is OpenRouter's own router over whatever free models are
# currently live, chosen specifically so this fallback path doesn't go
# stale every few months the way a hand-pinned free-tier slug does (free
# model catalogs churn on a timeline outside this project's control).
OPENROUTER_MODEL = "openrouter/free"


def _call_openrouter(user_prompt: str) -> str:
    """Fallback provider, tried only when Groq fails — a separate account
    with its own independent free-tier quota, so it still has headroom at
    exactly the moment Groq's is exhausted. Optional: if
    OPENROUTER_API_KEY isn't set, this raises immediately and
    generate_explanation degrades the same as if this function didn't
    exist."""
    api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 1200,
            "temperature": 0.6,
        },
        timeout=8,
    )
    if not response.ok:
        raise RuntimeError(f"OpenRouter returned {response.status_code}: {response.text[:500]}")
    data = response.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    if not content or not content.strip():
        raise RuntimeError("OpenRouter returned empty content")
    return content.strip()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _capitalize_first(text: str) -> str:
    """Guarantees the returned commentary starts with a capital letter,
    no matter what produced it. Several fact strings are deliberately
    lowercase ("the batsman...", "the bowler...") since they're normally
    used mid-sentence — but when a provider opens ITS sentence with one
    of them verbatim (following the "use this fact's exact wording"
    rule a little too literally), the result starts with a lowercase
    letter. A one-line fix in code removes this failure mode entirely
    instead of hoping every provider call gets it right on its own."""
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1:]
    return text


def generate_explanation(ball_context: dict) -> str | None:
    """
    ball_context expected keys: event_type, proba_before, proba_after, swing,
    cum_runs, cum_wickets, balls_remaining, required_run_rate. Optional keys
    (all improve commentary quality but degrade gracefully if omitted):
    cum_runs_before, cum_wickets_before, balls_remaining_before,
    required_run_rate_before, balls_elapsed, raw_event, previous_event
    (folded into the Delivery fact as a lead-in when there's a genuine
    narrative connection — a no-ball's lead-in also covers the free-hit
    context on its own, so is_free_hit is accepted but no longer read),
    batting_team, bowling_team, streak_type, streak_count, rate_momentum.

    For each provider, in order (Groq, then OpenRouter): call it once; if
    its output passes validate_output(), return it immediately. If it
    fails validation, ask the SAME provider for one corrective rewrite; if
    that also fails validation, return a deterministic fallback sentence
    built straight from the facts (never the model's unvalidated prose).
    Only if a provider's call raises outright (network error, missing
    key, rate limit) does this move on to the next provider.

    Returns None only if every provider's call itself failed outright —
    a down/rate-limited explanation service should degrade the
    commentary, never take out a /predict response that already has a
    perfectly good win_probability.
    """
    facts = build_facts(ball_context)
    user_prompt = "\n".join(f"- {line}" for line in facts["lines"])

    for provider in (_call_groq, _call_openrouter):
        try:
            text = provider(user_prompt)
        except Exception:
            logger.warning(
                "%s commentary call failed outright — trying the next option.",
                provider.__name__, exc_info=True,
            )
            continue

        issues = validate_output(text, facts)
        if not issues:
            return _capitalize_first(text)

        logger.warning(
            "%s output failed validation (%s) — requesting one corrective rewrite.",
            provider.__name__, issues,
        )
        corrective_prompt = (
            user_prompt
            + f"\n\nCORRECTION NEEDED: your previous answer had a problem ({', '.join(issues)}). "
            "Rewrite using ONLY the facts above, respecting the exact-wording rule, "
            "with no invented numbers or names, no ALL-CAPS words, no fact labels "
            "(the word before a colon) repeated in your answer, and make sure your "
            "sentence actually finishes."
        )
        try:
            text2 = provider(corrective_prompt)
        except Exception:
            logger.warning(
                "%s corrective rewrite call failed outright — falling back to a deterministic sentence.",
                provider.__name__, exc_info=True,
            )
            return _capitalize_first(_deterministic_fallback_sentence(facts))

        issues2 = validate_output(text2, facts)
        if not issues2:
            return _capitalize_first(text2)

        logger.warning(
            "%s corrective rewrite still failed validation (%s) — falling back to a deterministic sentence.",
            provider.__name__, issues2,
        )
        return _capitalize_first(_deterministic_fallback_sentence(facts))

    # Every provider's call raised outright — nothing usable to validate.
    return None


if __name__ == "__main__":
    # Manual smoke-test scenarios covering the main branches: a wicket, a
    # quiet dot ball, a wide vs. a no-ball, and continuity via
    # previous_event.
    scenarios = {
        "wicket (single ball)": {
            "event_type": "wicket", "raw_event": "wicket",
            "balls_elapsed": 1,
            "proba_before": 0.62, "proba_after": 0.47, "swing": -0.15,
            "cum_runs": 88, "cum_runs_before": 88,
            "cum_wickets": 5, "cum_wickets_before": 4,
            "balls_remaining": 24, "balls_remaining_before": 25,
            "required_run_rate": 11.25, "required_run_rate_before": 10.8,
            "runs_required": 40, "batting_team": "India", "bowling_team": "Australia",
        },
        "quiet dot ball (single ball)": {
            "event_type": "dot_ball", "raw_event": "dot_ball",
            "balls_elapsed": 1,
            "proba_before": 0.49, "proba_after": 0.436, "swing": -0.054,
            "cum_runs": 11, "cum_runs_before": 11,
            "cum_wickets": 0, "cum_wickets_before": 0,
            "balls_remaining": 115, "balls_remaining_before": 116,
            "required_run_rate": 8.03, "required_run_rate_before": 7.94,
            "runs_required": 154,
        },
        "wide (must NOT be read as a no-ball)": {
            "event_type": "other_runs", "raw_event": "wide",
            "balls_elapsed": 0,
            "proba_before": 0.460, "proba_after": 0.464, "swing": 0.004,
            "cum_runs": 13, "cum_runs_before": 12,
            "cum_wickets": 1, "cum_wickets_before": 1,
            "balls_remaining": 115, "balls_remaining_before": 115,
            "required_run_rate": 7.9, "required_run_rate_before": 8.0,
            "runs_required": 152,
        },
        "recovery after a wicket (previous_event demonstrates continuity)": {
            "event_type": "four", "raw_event": "four", "previous_event": "wicket",
            "balls_elapsed": 1,
            "proba_before": 0.55, "proba_after": 0.62, "swing": 0.07,
            "cum_runs": 95, "cum_runs_before": 91,
            "cum_wickets": 3, "cum_wickets_before": 3,
            "balls_remaining": 55, "balls_remaining_before": 56,
            "required_run_rate": 8.5, "required_run_rate_before": 8.7,
            "runs_required": 60, "batting_team": "Pakistan", "bowling_team": "England",
        },
    }
    for name, example in scenarios.items():
        print(f"--- {name} ---")
        print(build_user_prompt(example))
        result = generate_explanation(example)
        print(result if result is not None else "(explanation unavailable)")
        print()