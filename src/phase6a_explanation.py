"""
Phase 6a — LLM explanation layer

Given a ball's before/after match state and the model's probability swing,
generates a short natural-language explanation of why the win probability
moved. Talks to Groq's chat completions API, falling back to OpenRouter
(a separate account, a separate free-tier quota) if Groq fails — see
generate_explanation() below for why a single-provider design turned out
not to be enough.

This file has gone through several rounds of real-bug fixes, all driven by
actually reading generated commentary against live screenshots rather than
guessing what an LLM prompt "should" say. Rewritten from scratch here to
consolidate all of them into one coherent design instead of a stack of
patches. What's fixed, in the order it was found:

  1. A missing/invalid GROQ_API_KEY must not crash the app at import time,
     and a down/rate-limited Groq call must not take out a /predict
     response that already has a perfectly good win_probability.
  2. Citing a match-state number that didn't actually change as if it
     caused the swing (e.g. "pressure mounts... with 0 wickets down" when
     wickets were 0 both before and after — a number that never moved
     can't explain anything).
  3. Saying "after that over" when the ball was actually mid-over.
  4. A one-shot outage: when several balls get compressed into a single
     prediction call (Auto-predict toggled off, or a big Alter State edit),
     the single most recent event was being blamed for the entire swing,
     even when it was actually five or six deliveries' combined effect.
  5. Quoting an abstract required run rate deep in the death overs, where
     real commentators say "need 9 off the last 6" instead.
  6. Converging on one sentence template ("[Team]'s win probability slides
     from X% to Y%...") almost every single call.
  7. Using a decrease-word ("slides") for a swing that was actually an
     INCREASE, and vice versa.
  8. Citing a vague "from a low point" instead of the actual number.
  9. A rounding collision: e.g. 33.71% and 33.72% both display as "33.7%",
     producing a sentence like "slides from 33.7% to 33.7%" — a visible
     self-contradiction to the reader.
 10. A real formatting bug: the raw 0-1 probability float was passed to
     the model as `.2f` (e.g. 0.547 -> "0.55"), which is only whole-percent
     precision — the model then correctly read that as "55%" instead of
     the true 54.7%. Not a hallucination; the prompt was handing it a
     truncated number.
 11. "Flurry of runs" used to describe a SINGLE six — plural-implying
     language for a one-ball event.
 12. A no-ball narrated as a wide. Both collapse to the same "other_runs"
     category for the model's categorical feature (the model only knows 5
     event categories), so the explanation layer had no way to tell them
     apart once collapsed — it was guessing.
 13. Groq retired the model this originally called (llama-3.1-8b-instant)
     with no warning, breaking every call outright — see _call_groq below.
 14. Groq's free tier caps out at 30 requests/minute; clicking through
     balls quickly outruns that, and every call over the limit degraded
     to no commentary even though the win probability itself was fine.
     A single-provider design has no recourse when THAT provider is the
     one being throttled — see generate_explanation's OpenRouter fallback.

Set your GROQ_API_KEY as an environment variable before running (and
OPENROUTER_API_KEY, optional, for the fallback):
  Windows (PowerShell):  $env:GROQ_API_KEY="your_key_here"
"""

import logging
import os
import random
import requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)

# Each /predict call generates commentary independently, with no memory of
# what shape the previous ball's sentence took — so "vary your sentence
# construction" as a standing instruction has nothing to push against
# call-to-call, and in practice the model still converges on one "safe"
# template almost every time regardless of how strongly that's discouraged
# in the prompt. Rather than rely on the model to self-vary, force it: pick
# one of these concrete directives at random per call and append it to the
# user prompt, so consecutive balls in the same match are structurally
# different by construction, not by hoping for LLM creativity.
STRUCTURE_DIRECTIVES = [
    "For THIS response: open with the ball/event itself. Do not mention any percentage until the middle or end of the sentence.",
    "For THIS response: open with what this means for the chase (comfortable / tightening / slipping away) before any numbers appear.",
    "For THIS response: state the win probability change as a short, blunt clause using 'becomes' or 'now, up/down from' — do not use the word 'slides' or 'climbs' at all.",
    "For THIS response: lead with the most important non-probability number (balls left, runs needed, wickets, required rate) before circling back to the percentage.",
    "For THIS response: put the percentage change in the middle of the sentence, not the start or the very end.",
]

# Readable labels for the literal delivery clicked. See "raw_event" below —
# this is what fixes a no-ball being read as a wide: both collapse to the
# same "other_runs" model category, but here they get distinct, unambiguous
# descriptions the model can't confuse.
RAW_EVENT_LABELS = {
    "dot_ball": "Dot ball (0 runs, legal delivery)",
    "single": "Single (1 run, legal delivery)",
    "two": "Two runs (legal delivery)",
    "three": "Three runs (legal delivery)",
    "four": "Four (boundary, legal delivery)",
    "six": "Six (maximum, legal delivery)",
    "wicket": "Wicket (legal delivery)",
    "wide": "WIDE — an illegal delivery bowled out of the batter's reach; 1 extra run, no legal ball used. NOT a no-ball.",
    "noball": "NO-BALL — an illegal delivery (e.g. overstepping); 1 extra run, no legal ball used. NOT a wide.",
}

# Built lazily (and re-checked on every call) rather than at import time, so
# a missing/invalid GROQ_API_KEY doesn't take the whole module — and by
# extension the FastAPI app that imports it — down before it even starts.
_client = None


def _get_client():
    global _client
    if _client is None:
        # .strip() defensively — a .env file edited or saved on Windows
        # can leave a trailing \r or stray whitespace on the value, which
        # silently corrupts an Authorization header (this exact class of
        # bug already hit this project once, as CRLF line endings in
        # committed JSON files — see the .gitattributes fix). A malformed
        # Bearer token can produce a confusing error far removed from
        # "check your API key," so strip it before it ever gets used.
        api_key = (os.environ.get("GROQ_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        # Defaults are a 60s read timeout with 2 automatic retries — on
        # Groq's free tier (30 requests/minute), hitting that cap mid-
        # session means every call over the limit sits through a slow
        # retry-with-backoff cycle before finally giving up.
        # max_retries=0 (not 1): now that generate_explanation() has an
        # OpenRouter fallback, retrying the SAME rate-limited provider
        # before giving up is pure wasted time — a 429 means Groq is
        # full for this window regardless of how many more times this
        # client asks in the next few seconds. The real "retry" is
        # falling over to OpenRouter's separate, independent quota
        # immediately, not waiting through a backoff delay first. This
        # is what fixes the 5-6 second stall before commentary appeared:
        # that delay was this client's own retry-with-backoff cycle
        # running to completion before the fallback ever got a chance to
        # run. A short timeout still caps how long a single (non-retried)
        # attempt can hang if Groq is merely slow rather than rate-limited.
        _client = Groq(api_key=api_key, timeout=8.0, max_retries=0)
    return _client


EXPLANATION_SYSTEM_PROMPT = """You are a sharp T20 cricket commentator narrating
a win-probability swing after one ball, for an ordinary fan, not a data
analyst.

Write 1-2 flowing sentences (25-50 words) in a real commentator's voice.
Every fact needs its own connector ("and", "with", "as", "while") joining
it to what comes next — never comma-spliced fragments with nothing
linking them. If too many facts don't fit cleanly, keep only the 1-2 most
important match-state numbers (per "Must include" below) and cut the rest
rather than forcing everything into one overloaded clause.

The data given below has already resolved every judgment call for you —
which exact swing words are allowed, whether this is mid-over or an over
just completed, which numbers actually changed vs. stayed flat, how to
frame the required rate at this stage of the innings, and the exact
number formatting to use. Follow those inline instructions on each data
line exactly; do not re-derive, second-guess, or override them.

Two examples — different shapes on purpose, neither is "the template":

  Small swing: "A tidy single nudges the target down to 151 as the
  required rate ticks up just a touch to 8.1 an over — barely a ripple
  in a comfortable chase."

  Large swing: "That's a big blow — a wicket falls and 62% becomes 47%
  in a heartbeat, the batting side down to 5 wickets to rebuild the
  chase around."

Rules:
- Only cite a number as a cause if the data below marks it as CHANGED, or
  it's the direct subject of the event (e.g. wickets on a wicket ball).
  A number marked UNCHANGED caused nothing — never call it a cause.
- Match the DELIVERY's drama to what actually happened on the field, not
  to the size of the swing — a single or dot ball is routine even if a
  large swing is attached to it (usually the required rate biting, not
  the ball itself). Save "big blow"/"disaster"/"crucial" language for
  wickets or a genuinely decisive boundary.
- "Wickets down" means wickets LOST — never say "down to N wickets"
  (that means N REMAIN, the opposite of the data).
- "Level" means TIE — never use it for winning the chase outright; say
  "reach the target" or "get there" instead.
- Target is fixed all innings: mention it at most once if truly needed
  for stakes, and it never counts as one of your 1-2 required numbers.
- Use the specific delivery label given (wide / no-ball / single / two /
  etc.) exactly — never guess or default to "wide."
- Say "after that over" only if the data below marks the over as just
  completed AND this is a single delivery (not a multi-ball span).
- Use the exact probability numbers and swing words given below — they
  already account for rounding and intensity; never invent your own.

Must include: the probability change (using the given numbers/words),
one or two match-state numbers that genuinely changed, and a plain
implication for the chase (comfortable / tight / slipping away) — not
just a bare number.

Avoid overused phrasing: "the equation becomes/is a bit more
manageable", "a small mercy for the batting team", tacking "still
needing X off Y" onto every sentence as a rote closing tag, and opening
every response with "[Team]'s win probability slides from X% to Y%" —
vary the verb and where numbers land in the sentence.

Use team names naturally if given ("India's win probability slides");
otherwise "the batting/bowling team." Never invent a name, player, or
number that isn't in the data given below for THIS ball."""


def build_user_prompt(ball_context: dict) -> str:
    target = ball_context['cum_runs'] + ball_context.get('runs_required', 0)
    proba_before = ball_context['proba_before']
    proba_after = ball_context['proba_after']
    swing = ball_context['swing']
    before_pct = proba_before * 100
    after_pct = proba_after * 100
    swing_points = abs(swing) * 100

    # Explicit, unmissable direction label — don't make the model infer
    # up-vs-down from the sign of a "+0.15" style number, hand it the word
    # directly. Fix for commentary that said probability "slides" (a
    # decrease word) when it had actually gone UP.
    direction = 'UP' if swing > 1e-9 else 'DOWN' if swing < -1e-9 else 'UNCHANGED'

    # Which words are actually allowed for this swing — computed here,
    # not left for the model to work out from a table. Real generated
    # commentary showed the intensity-band table alone wasn't reliable:
    # a -10.8pt swing described as "edges lower" (that's the <3pt tier),
    # a +14.4pt swing as "edges higher", a -4.7pt swing as "ticks down"
    # — all three undersold a swing several times larger than the words
    # they used. Handing over the pre-resolved answer for this exact
    # swing removes the "check the sign, find the row, pick the column"
    # step entirely, the same fix that already made Direction (above)
    # and the rounding-collision case reliable.
    if swing_points < 3:
        band_words = "'ticks up' / 'edges higher' / 'nudges up a touch'" if direction == 'UP' else "'ticks down' / 'edges lower' / 'dips a touch'"
        band_note = "quietest tier — do not use \"mounts\", \"surges\", \"dramatic\", \"crumbles\" here, this swing is small"
    elif swing_points < 8:
        band_words = "'nudges up' / 'climbs a little' / 'eases higher'" if direction == 'UP' else "'eases back' / 'slips' / 'tightens against them'"
        band_note = "modest tier — do not use \"ticks\"/\"edges\" (too quiet) or \"crashes\"/\"surges\" (too loud) for this swing"
    elif swing_points < 15:
        band_words = "'swings up' / 'jumps to' / 'climbs sharply'" if direction == 'UP' else "'swings down' / 'slides' / 'drops'"
        band_note = "genuine-swing tier — \"ticks\"/\"edges\"/\"nudges\" are all too quiet for a swing this size"
    else:
        band_words = "'surges' / 'rockets up' / 'soars'" if direction == 'UP' else "'crashes' / 'plummets' / 'collapses'"
        band_note = "dramatic tier — this swing has earned strong language, don't undersell it with \"ticks\"/\"edges\"/\"eases\""

    # Three tiers, because two distinct real bugs showed up in generated
    # commentary:
    #   1. Negligible swing (<0.05 points): even one decimal place shows
    #      identical numbers ("46.4% from 46.4%") — forcing more decimals
    #      is the wrong fix for a swing that's genuinely not meaningfully
    #      different. Tell the model not to present a from/to PAIR at all.
    #   2. Small but real swing where 1 decimal still collides (0.05-0.09
    #      points): escalate to 2 decimals so the numbers are visibly
    #      different.
    #   3. Normal case: always format as a percentage with 1 decimal place
    #      — NEVER the raw 0-1 float truncated with :.2f (that was a real
    #      bug: 0.547 formatted as "0.55" gets read by the model as "55%"
    #      instead of the true 54.7%, which is exactly what produced
    #      commentary stating "55.0%" for a state that was actually 54.7%
    #      on screen — not a hallucination, a truncated input).
    if swing_points < 0.05:
        proba_before_line = f"Probability before: {before_pct:.1f}%"
        proba_after_line = (
            f"Probability after: {after_pct:.1f}% — NEGLIGIBLE SWING (<0.05 points), "
            "these will read as the same number. Do NOT write a 'from X% to Y%' "
            "sentence — that would show two identical-looking numbers as if "
            f"something changed. State the number ONCE instead (e.g. 'holds "
            f"steady around {after_pct:.0f}%', 'barely moves')."
        )
    elif round(before_pct, 1) == round(after_pct, 1):
        proba_before_line = f"Probability before: {before_pct:.2f}%"
        proba_after_line = f"Probability after: {after_pct:.2f}% (still round to the same number at 1 decimal — use 2 decimal places for both so they read as visibly different)"
    else:
        proba_before_line = f"Probability before: {before_pct:.1f}%"
        proba_after_line = f"Probability after: {after_pct:.1f}%"

    # Which framing convention applies — derived from balls_remaining,
    # which is already sent for every request, so this needs no new field.
    # Threshold is the last 10 overs (balls_remaining <= 60) for the
    # runs-off-balls framing; overs 1-10 comfortably sits inside the
    # "required rate is fine up to ~15-17 overs" allowance too.
    balls_remaining = ball_context.get('balls_remaining', 0)
    in_death_overs = balls_remaining <= 60
    overs_bowled = (120 - balls_remaining) / 6
    phase_line = (
        f"Innings phase: LAST 10 OVERS (over {overs_bowled:.1f} of 20) — do NOT state "
        "the required run rate number below; frame the equation as runs needed off "
        "balls remaining instead."
        if in_death_overs else
        f"Innings phase: overs 1-10 (over {overs_bowled:.1f} of 20) — required run rate "
        "is a fine, natural framing here."
    )

    def before_after(label, after_key, before_key, fmt="{}"):
        after_val = ball_context.get(after_key)
        before_val = ball_context.get(before_key)
        if before_val is None:
            return f"{label}: {fmt.format(after_val)} (before-ball value unknown — state current value only, don't claim it changed)"
        if before_val == after_val:
            return f"{label}: {fmt.format(after_val)} (UNCHANGED this ball — do not cite as a cause)"
        return f"{label}: {fmt.format(before_val)} -> {fmt.format(after_val)} (this is what changed)"

    required_rate_note = (
        " (DO NOT quote this number this response — see innings-phase framing rule)"
        if in_death_overs else ""
    )

    # How many legal deliveries this update actually spans — see the
    # multi-ball-span instructions above. Without this, the model has no
    # way to know it's describing five or six balls of play instead of
    # one, and confidently narrates "that wicket" as the sole cause of a
    # swing several other deliveries also contributed to.
    balls_elapsed = ball_context.get("balls_elapsed")
    if balls_elapsed is None:
        elapsed_line = "Balls since last commentary: unknown"
    elif balls_elapsed > 1:
        elapsed_line = (
            f"Balls since last commentary: {balls_elapsed} — MULTIPLE deliveries "
            "happened since the last update, not just one. Frame this as a stretch "
            "of play (e.g. \"across those last few balls\") rather than blaming the "
            "single named event for the whole swing, and do not say \"after that "
            "over\" even if the over line below says it just completed."
        )
    elif balls_elapsed == 0:
        elapsed_line = "Balls since last commentary: 0 legal deliveries (this delivery was a wide/no-ball — see Specific delivery below for which one)"
    else:
        elapsed_line = "Balls since last commentary: 1 (a single delivery — normal single-ball case, no \"flurry\" or plural language)"

    # The literal delivery clicked (distinct from the collapsed model
    # category below) — this is what fixes a no-ball being narrated as a
    # wide. Falls back to nothing if not provided (Begin Chase / Alter
    # State submissions aren't a literal single delivery).
    raw_event = ball_context.get("raw_event")
    specific_delivery_line = (
        f"Specific delivery: {RAW_EVENT_LABELS[raw_event]}"
        if raw_event in RAW_EVENT_LABELS
        else "Specific delivery: not provided — describe generally from Event category below, do not guess wide vs no-ball"
    )

    lines = [
        f"Event category (model feature, collapses wide/no-ball/single/two/three together): {ball_context['event_type']}",
        specific_delivery_line,
        elapsed_line,
        phase_line,
        f"Over just completed: {'yes' if ball_context.get('over_just_completed') else 'no'}",
        proba_before_line,
        proba_after_line,
        f"Direction: {direction} (pick your verb from the {direction} column — no exceptions)",
        f"Swing: {swing:+.2f} ({swing_points:.1f} points, {direction}). Allowed words for THIS swing ({band_note}): {band_words} — use one of these, not a word from a different tier.",
        before_after("Score (runs/wickets)", "cum_runs", "cum_runs_before"),
        f"Target: {target} (fixed all innings, never changes — do not restate this as if it's new, and it doesn't count as one of your required numbers)",
        f"Runs still needed: {ball_context.get('runs_required', 'unknown')}",
        before_after("Balls remaining", "balls_remaining", "balls_remaining_before"),
        before_after("Required run rate", "required_run_rate", "required_run_rate_before", fmt="{:.1f}") + required_rate_note,
        before_after("Wickets down", "cum_wickets", "cum_wickets_before"),
    ]
    batting_team = ball_context.get("batting_team")
    bowling_team = ball_context.get("bowling_team")
    if batting_team:
        lines.append(f"Batting team: {batting_team}")
    if bowling_team:
        lines.append(f"Bowling team: {bowling_team}")
    return "\n".join(lines) + "\n"


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# meta-llama/llama-3.3-70b-instruct:free — chosen deliberately as a plain
# INSTRUCT model, not a reasoning model. gpt-oss-20b on Groq (above) is a
# reasoning model that spends part of its token budget on invisible
# chain-of-thought before writing an answer, which is exactly what caused
# the empty-content bug this file already works around. A plain instruct
# model sidesteps that whole class of failure on the fallback path rather
# than reproducing it. Confirmed free and live on OpenRouter as of this
# writing; OpenRouter's free model catalog does rotate over time, so if
# this one disappears, check openrouter.ai/models?max_price=0 for a
# current replacement — same swap Groq's own deprecation just forced here.
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"


def _call_groq(user_prompt: str) -> str:
    """Raises on ANY failure — missing/invalid key, network error, rate
    limit, or empty content — rather than degrading to None itself.
    generate_explanation() below is what decides whether a fallback
    provider is worth trying; this function's only job is "did Groq
    actually produce usable text, yes or no."
    """
    client = _get_client()
    response = client.chat.completions.create(
        # Groq retired both llama-3.1-8b-instant and llama-3.3-70b-
        # versatile on 2026-08-16 (see console.groq.com/docs/deprecations)
        # — every call to either now 404s with model_not_found,
        # regardless of API key. openai/gpt-oss-20b is Groq's official
        # 1:1 replacement for llama-3.1-8b-instant, and — unlike the
        # old pairing, where 8b-instant had ~5x the free-tier token
        # budget of 70b-versatile — gpt-oss-20b and its bigger sibling
        # gpt-oss-120b currently share the same free-tier budget
        # (200,000 tokens/day), so this is a straight like-for-like
        # swap, not a budget-driven downgrade. If Groq changes pricing
        # again, check console.groq.com/docs/rate-limits before
        # switching models.
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        # gpt-oss-20b is a REASONING model — unlike llama-3.1-8b-instant
        # (plain instruct, no hidden thinking step), it spends some of
        # its completion-token budget on an internal chain-of-thought
        # before writing the actual answer. At the old max_tokens=140
        # (sized for a model with no thinking step), it was burning the
        # entire budget on reasoning and returning finish_reason="length"
        # with message.content == "" — a real, documented failure mode
        # for gpt-oss models on Groq when max_tokens is too tight, not
        # an error, just an empty string, which is exactly why this
        # failed silently (no exception, no log line) while
        # `explanation` came back as "". reasoning_effort="low" keeps
        # the thinking step short — this task doesn't need genuine
        # reasoning depth, the prompt already does the "thinking" via
        # its rules/examples — and max_tokens is raised to leave
        # headroom for that reasoning step plus the actual 25-50 word
        # answer.
        reasoning_effort="low",
        max_tokens=400,
        temperature=0.6,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        # Belt-and-suspenders: if the model still returns empty (e.g.
        # a future Groq-side default change), raise rather than return
        # an empty string — a caller checking `if explanation:` would
        # otherwise treat "" as success.
        raise RuntimeError(
            f"Groq returned empty content (finish_reason="
            f"{getattr(response.choices[0], 'finish_reason', 'unknown')})"
        )
    return content.strip()


def _call_openrouter(user_prompt: str) -> str:
    """Fallback provider, tried only when Groq fails — a separate
    account with its own, entirely independent free-tier quota
    (OpenRouter: 20 requests/minute on free models, its own daily cap),
    so it has real headroom left even at the exact moment Groq's 8000-
    tokens/minute limit has been hit. Optional: if OPENROUTER_API_KEY
    isn't set, this raises immediately and generate_explanation degrades
    to no commentary, same as if this function didn't exist — adding it
    is additive, never a new requirement. Same "raise on any failure"
    contract as _call_groq above.
    """
    # .strip() for the same reason as _get_client() above — a corrupted
    # key from a stray \r or trailing space produces a malformed
    # Authorization header, which is exactly the kind of thing that can
    # surface as a confusing, seemingly-unrelated HTTP error (a real
    # instance of this: OpenRouter returning 404 Not Found on a
    # perfectly valid, documented URL and model — not the "model doesn't
    # exist" or "bad auth" error you'd expect, because a malformed header
    # can get rejected by routing/edge infrastructure before it ever
    # reaches application-level auth or model-lookup code, which reports
    # errors differently and less clearly).
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
            "max_tokens": 140,
            "temperature": 0.6,
        },
        # Short timeout, no retries (unlike the Groq client's defaults) —
        # this is already the fallback path; if OpenRouter is also slow
        # or down, fail fast and let generate_explanation degrade to None
        # rather than compounding two providers' worth of retry delay.
        timeout=8,
    )
    if not response.ok:
        # Surface the actual response body, not just the status code —
        # response.raise_for_status() alone gives "404 Client Error: Not
        # Found for url: ..." with no indication of WHY, which is exactly
        # how a real OpenRouter-side rejection (bad model slug, malformed
        # auth header, account issue) turned into an unsolvable mystery
        # the first time this happened. Truncated to keep log lines sane
        # if the body is an unexpectedly large HTML error page rather
        # than the small JSON error OpenRouter normally returns.
        raise RuntimeError(
            f"OpenRouter returned {response.status_code}: {response.text[:500]}"
        )
    data = response.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    if not content or not content.strip():
        raise RuntimeError("OpenRouter returned empty content")
    return content.strip()


def generate_explanation(ball_context: dict) -> str | None:
    """
    ball_context expected keys: event_type, proba_before, proba_after, swing,
    cum_runs, cum_wickets, balls_remaining, required_run_rate. Optional keys
    (all improve commentary quality but degrade gracefully if omitted):
    cum_runs_before, cum_wickets_before, balls_remaining_before,
    required_run_rate_before, over_just_completed, balls_elapsed, raw_event,
    batting_team, bowling_team.

    Tries Groq first, then OpenRouter (only if OPENROUTER_API_KEY is set)
    if Groq fails for ANY reason — a missing/invalid key, a network error,
    or Groq's free-tier rate limit (30 requests/minute) being hit, which
    in practice is the single most common failure mode: clicking through
    balls quickly outruns it well before a real outage would. A single-
    provider design has no recourse at exactly that moment; OpenRouter is
    a separate account with its own, independent quota, so it still has
    headroom when Groq doesn't.

    Returns None (instead of raising) only if BOTH providers fail — a
    down/rate-limited explanation service should degrade the commentary,
    never take out a /predict response that already has a perfectly good
    win_probability. Callers should treat None as "no commentary
    available this time."
    """
    # A fresh structure directive every call — appended here (not baked
    # into build_user_prompt) so build_user_prompt stays pure/testable
    # while the actual request still gets call-to-call variety. Built
    # once and reused for both providers below, so a Groq->OpenRouter
    # fallback still gets the same structural instruction rather than a
    # second random one.
    user_prompt = build_user_prompt(ball_context) + "\n" + random.choice(STRUCTURE_DIRECTIVES)

    try:
        return _call_groq(user_prompt)
    except Exception:
        logger.warning(
            "Groq commentary call failed — trying OpenRouter fallback.",
            exc_info=True,
        )

    try:
        return _call_openrouter(user_prompt)
    except Exception:
        logger.exception(
            "Both Groq and OpenRouter commentary calls failed; continuing without commentary."
        )
        return None


if __name__ == "__main__":
    # Manual test scenarios — one per bug this file fixes, so a full
    # __main__ run doubles as a regression check against every real issue
    # found so far (though it still needs a working GROQ_API_KEY and
    # network access to actually see generated text, not just the prompt).
    scenarios = {
        "wicket (single ball)": {
            "event_type": "wicket", "raw_event": "wicket",
            "over_just_completed": False, "balls_elapsed": 1,
            "proba_before": 0.62, "proba_after": 0.47, "swing": -0.15,
            "cum_runs": 88, "cum_runs_before": 88,
            "cum_wickets": 5, "cum_wickets_before": 4,
            "balls_remaining": 24, "balls_remaining_before": 25,
            "required_run_rate": 11.25, "required_run_rate_before": 10.8,
            "runs_required": 40,
        },
        "quiet dot ball (single ball)": {
            "event_type": "dot_ball", "raw_event": "dot_ball",
            "over_just_completed": False, "balls_elapsed": 1,
            "proba_before": 0.49, "proba_after": 0.436, "swing": -0.054,
            "cum_runs": 11, "cum_runs_before": 11,
            "cum_wickets": 0, "cum_wickets_before": 0,
            "balls_remaining": 115, "balls_remaining_before": 116,
            "required_run_rate": 8.03, "required_run_rate_before": 7.94,
            "runs_required": 154,
        },
        "multi-ball jump (several balls compressed into one update)": {
            "event_type": "wicket", "raw_event": "wicket",
            "over_just_completed": True, "balls_elapsed": 6,
            "proba_before": 0.83, "proba_after": 0.55, "swing": -0.278,
            "cum_runs": 171, "cum_runs_before": 156,
            "cum_wickets": 7, "cum_wickets_before": 6,
            "balls_remaining": 6, "balls_remaining_before": 12,
            "required_run_rate": 9.0, "required_run_rate_before": 8.0,
            "runs_required": 9,
        },
        "negligible swing (rounding collision even at 1 decimal)": {
            "event_type": "dot_ball", "raw_event": "dot_ball",
            "over_just_completed": False, "balls_elapsed": 1,
            "proba_before": 0.3371, "proba_after": 0.3372, "swing": 0.0001,
            "cum_runs": 0, "cum_runs_before": 0,
            "cum_wickets": 0, "cum_wickets_before": 0,
            "balls_remaining": 119, "balls_remaining_before": 120,
            "required_run_rate": 8.3, "required_run_rate_before": 8.25,
            "runs_required": 165,
        },
        "precision check (54.7%, not truncated to 55.0%)": {
            "event_type": "six", "raw_event": "six",
            "over_just_completed": False, "balls_elapsed": 1,
            "proba_before": 0.451, "proba_after": 0.547, "swing": 0.096,
            "cum_runs": 12, "cum_runs_before": 6,
            "cum_wickets": 0, "cum_wickets_before": 0,
            "balls_remaining": 116, "balls_remaining_before": 117,
            "required_run_rate": 7.9, "required_run_rate_before": 8.2,
            "runs_required": 153,
        },
        "wide (must NOT be read as a no-ball)": {
            "event_type": "other_runs", "raw_event": "wide",
            "over_just_completed": False, "balls_elapsed": 0,
            "proba_before": 0.460, "proba_after": 0.464, "swing": 0.004,
            "cum_runs": 13, "cum_runs_before": 12,
            "cum_wickets": 1, "cum_wickets_before": 1,
            "balls_remaining": 115, "balls_remaining_before": 115,
            "required_run_rate": 7.9, "required_run_rate_before": 8.0,
            "runs_required": 152,
        },
        "no-ball (must NOT be read as a wide)": {
            "event_type": "other_runs", "raw_event": "noball",
            "over_just_completed": False, "balls_elapsed": 0,
            "proba_before": 0.464, "proba_after": 0.468, "swing": 0.004,
            "cum_runs": 14, "cum_runs_before": 13,
            "cum_wickets": 1, "cum_wickets_before": 1,
            "balls_remaining": 115, "balls_remaining_before": 115,
            "required_run_rate": 7.8, "required_run_rate_before": 7.9,
            "runs_required": 151,
        },
    }
    for name, example in scenarios.items():
        print(f"--- {name} ---")
        print(build_user_prompt(example))
        result = generate_explanation(example)
        print(result if result is not None else "(explanation unavailable)")
        print()