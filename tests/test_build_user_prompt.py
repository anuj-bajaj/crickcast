"""
Unit tests for the facts -> prompt -> validation pipeline in
phase6a_explanation.py. Split into three groups:
  - build_facts()/build_user_prompt(): pure-function tests of the
    deterministic fact computation, no network involved.
  - validate_output(): the code-level guardrail, tested directly against
    synthetic good and bad model outputs.
  - _deterministic_fallback_sentence(): the always-safe last resort.
"""

from src.phase6a_explanation import (
    build_facts,
    build_user_prompt,
    validate_output,
    _deterministic_fallback_sentence,
    _capitalize_first,
)


def _base_context(**overrides):
    context = dict(
        event_type="other_runs",
        proba_before=0.50,
        proba_after=0.52,
        swing=0.02,
        cum_runs=100,
        cum_wickets=3,
        balls_remaining=60,
        runs_required=50,
        required_run_rate=8.0,
    )
    context.update(overrides)
    return context


# --- build_facts / build_user_prompt -----------------------------------

def test_milestone_fires_when_crossed_this_ball():
    prompt = build_user_prompt(_base_context(cum_runs=102, cum_runs_before=98))
    assert "Milestone" in prompt
    assert "100" in prompt


def test_milestone_does_not_fire_when_not_crossed():
    prompt = build_user_prompt(_base_context(cum_runs=105, cum_runs_before=102))
    assert "Milestone" not in prompt


def test_milestone_picks_highest_threshold_crossed():
    prompt = build_user_prompt(_base_context(cum_runs=152, cum_runs_before=95))
    assert "150" in prompt


def test_risk_framing_overs_notation_correct_outside_death_overs():
    """An over only has 6 balls (0-5) — balls_remaining // 6 . balls_remaining % 6
    must never produce a fractional part of .6 through .9, for every
    balls_remaining value where Risk framing actually states it (outside
    the death overs — see test_risk_framing_omits_overs_when_chase_equation_already_states_balls
    for why it's omitted inside them)."""
    for balls_remaining in range(61, 121):  # outside death overs (balls_remaining <= 60)
        prompt = build_user_prompt(_base_context(
            balls_remaining=balls_remaining, cum_wickets=7,  # wickets_in_hand=3, triggers Risk framing
        ))
        if "Risk framing" in prompt:
            fraction = balls_remaining % 6
            assert f"{balls_remaining // 6}.{fraction}" in prompt
            assert fraction <= 5


def test_risk_framing_omits_overs_when_chase_equation_already_states_balls():
    """Inside the death overs, Chase equation already states the exact
    same remaining span in balls ('needs 31 more off 15 balls') — Risk
    framing restating it again as '2.3 overs left' in the same prompt
    is the same number said twice in two different units."""
    prompt = build_user_prompt(_base_context(
        balls_remaining=15, cum_wickets=7,  # wickets_in_hand=3, in death overs
        cum_runs=100, cum_runs_before=100,  # no Score fact, so it can't randomly
        # win the "one of Score/Chase equation" cap instead of Chase equation
        # (see test_supporting_facts_capped_to_one_when_both_present)
    ))
    assert "Risk framing" in prompt
    assert "overs left" not in prompt.split("Risk framing")[1].split("\n")[0]
    assert "Chase equation" in prompt
    assert "off 15 balls" in prompt


def test_risk_framing_states_overs_when_no_chase_equation_present():
    """Outside the death overs, nothing else in the prompt states the
    remaining time, so Risk framing's own overs-left figure is the only
    place it appears and must stay."""
    prompt = build_user_prompt(_base_context(
        balls_remaining=70, cum_wickets=7,  # wickets_in_hand=3, outside death overs
    ))
    assert "Risk framing" in prompt
    assert "overs left" in prompt.split("Risk framing")[1].split("\n")[0]
    assert "Chase equation" not in prompt


def test_risk_framing_does_not_fire_in_the_very_last_two_overs():
    """With only a ball or two of the innings left, wickets in hand
    stops being a real constraint on risk-taking (there's barely any
    time left to lose more of them regardless) — 'only N wickets in
    hand' with 1.1 overs left is misleading, not a genuine warning."""
    prompt = build_user_prompt(_base_context(balls_remaining=7, cum_wickets=7))  # 1.1 overs left
    assert "Risk framing" not in prompt


def test_risk_framing_still_fires_with_a_meaningful_number_of_overs_left():
    prompt = build_user_prompt(_base_context(balls_remaining=30, cum_wickets=7))  # 5 overs left
    assert "Risk framing" in prompt


def test_chase_equation_names_the_batting_side_not_bowling_side():
    """Only the batting side ever 'needs' runs in a chase. Score and
    Chase equation are both capped to "at most one shown" when both are
    true (see test_supporting_facts_capped_to_one_when_both_present), so
    this retries until Chase equation is the one drawn rather than
    asserting on a single, possibly-dropped call."""
    ctx = _base_context(
        balls_remaining=30, batting_team="West Indies", bowling_team="India",
        runs_required=17,
    )
    chase_line = None
    for _ in range(30):
        facts = build_facts(ctx)
        chase_line = next((l for l in facts["lines"] if l.startswith("Chase equation")), None)
        if chase_line:
            break
    assert chase_line is not None, "Chase equation fact never appeared across 30 tries"
    assert "West Indies needs" in chase_line
    assert "India needs" not in chase_line


def test_delivery_never_uses_a_team_name():
    """A single ball's action is one player's doing, not the team's —
    Delivery always uses a person-level generic ('the batsman'/'the
    bowler'), even when team names are known."""
    for _ in range(10):
        facts = build_facts(_base_context(
            raw_event="four", batting_team="Pakistan", bowling_team="England",
        ))
        delivery_line = next(l for l in facts["lines"] if l.startswith("Delivery"))
        assert "Pakistan" not in delivery_line
        assert "England" not in delivery_line
        assert "the batsman" in delivery_line


def test_delivery_never_misattributes_action_to_the_team_as_a_whole():
    facts = build_facts(_base_context(raw_event="wicket"))
    delivery_line = next(l for l in facts["lines"] if l.startswith("Delivery"))
    assert "the bowler" in delivery_line
    assert "the bowling side" not in delivery_line


def test_previous_ball_continuity_fires_for_wicket_then_boundary():
    """Recovering after a wicket is a genuine narrative thread worth
    calling back to."""
    facts = build_facts(_base_context(raw_event="four", previous_event="wicket"))
    delivery_line = next(l for l in facts["lines"] if l.startswith("Delivery"))
    assert delivery_line.startswith("Delivery: after the wicket on the previous ball,")
    # Current ball's action must still follow, after the comma, with the
    # batsman as its subject and "four" as the outcome.
    tail = delivery_line.split(",", 1)[1].strip()
    assert tail.startswith("the batsman")
    assert "four" in tail


def test_previous_ball_continuity_fires_for_a_wide():
    """A wide on the previous delivery is worth a mention regardless of
    what this ball is — it didn't count toward the over."""
    facts = build_facts(_base_context(raw_event="single", previous_event="wide"))
    delivery_line = next(l for l in facts["lines"] if l.startswith("Delivery"))
    assert "after the wide on the previous ball," in delivery_line


def test_previous_ball_after_a_noball_states_the_free_hit_instead():
    """A no-ball's very next delivery is automatically a free hit — the
    lead-in states that directly instead of a generic 'after the
    no-ball' callback, and there's no separate standalone Free-hit fact
    saying the same thing again."""
    facts = build_facts(_base_context(raw_event="four", previous_event="noball", is_free_hit=True))
    delivery_line = next(l for l in facts["lines"] if l.startswith("Delivery"))
    assert delivery_line.startswith("Delivery: on the free hit that followed the no-ball,")
    assert not any(l.startswith("Free hit") for l in facts["lines"])


def test_previous_ball_continuity_omitted_for_unrelated_sequence():
    """Most consecutive balls have no genuine thread — a dot ball
    followed by an ordinary single shouldn't drag in a callback."""
    facts = build_facts(_base_context(raw_event="single", previous_event="dot_ball"))
    delivery_line = next(l for l in facts["lines"] if l.startswith("Delivery"))
    assert "on the previous ball" not in delivery_line


def test_previous_ball_continuity_fires_for_back_to_back_boundaries():
    facts = build_facts(_base_context(raw_event="four", previous_event="six"))
    delivery_line = next(l for l in facts["lines"] if l.startswith("Delivery"))
    assert "after the six on the previous ball," in delivery_line


def test_previous_ball_lead_in_names_no_actor():
    """The lead-in names the event itself, not who did it — 'after the
    wicket', not 'after the bowler's wicket'."""
    facts = build_facts(_base_context(raw_event="four", previous_event="wicket"))
    delivery_line = next(l for l in facts["lines"] if l.startswith("Delivery"))
    assert "bowler's" not in delivery_line
    assert "batsman's" not in delivery_line


# --- consecutive-event streaks --------------------------------------------

def test_streak_fact_for_fours_on_the_trot():
    facts = build_facts(_base_context(raw_event="four", streak_type="four", streak_count=3))
    streak_line = next(l for l in facts["lines"] if l.startswith("Streak"))
    assert "3 fours on the trot" in streak_line


def test_streak_fact_for_consecutive_sixes():
    facts = build_facts(_base_context(raw_event="six", streak_type="six", streak_count=2))
    streak_line = next(l for l in facts["lines"] if l.startswith("Streak"))
    assert "2 consecutive sixes" in streak_line


def test_streak_fact_for_wickets_in_a_cluster():
    facts = build_facts(_base_context(raw_event="wicket", streak_type="wicket", streak_count=2))
    streak_line = next(l for l in facts["lines"] if l.startswith("Streak"))
    assert "2 wickets in 2 balls" in streak_line


def test_streak_fact_for_consecutive_wides():
    facts = build_facts(_base_context(raw_event="wide", streak_type="wide", streak_count=2))
    streak_line = next(l for l in facts["lines"] if l.startswith("Streak"))
    assert "2 wides in a row" in streak_line


def test_streak_fact_for_consecutive_noballs():
    facts = build_facts(_base_context(raw_event="noball", streak_type="noball", streak_count=2))
    streak_line = next(l for l in facts["lines"] if l.startswith("Streak"))
    assert "2 no-balls in a row" in streak_line


def test_previous_ball_callback_suppressed_when_wicket_streak_is_reported():
    """The third wicket in a row should be framed by the Streak fact
    ('3 wickets in 3 balls'), not also by a generic 'after the wicket
    on the previous ball' callback saying the same thing a second way."""
    facts = build_facts(_base_context(
        raw_event="wicket", previous_event="wicket", streak_type="wicket", streak_count=3,
    ))
    delivery_line = next(l for l in facts["lines"] if l.startswith("Delivery"))
    assert "on the previous ball" not in delivery_line
    streak_line = next(l for l in facts["lines"] if l.startswith("Streak"))
    assert "3 wickets in 3 balls" in streak_line


def test_wicket_streak_still_reported_and_suppresses_callback_when_match_just_ended():
    """Regression test: a real, observed bug — the match-ending wicket
    (the one that takes the side all out) is exactly the ball where a
    wicket streak is most informative ('2 wickets in 2 balls' explains
    WHY the innings just folded), but it was being silently dropped
    because match-decided facts got nulled too broadly. That, in turn,
    let the generic 'after the wicket on the previous ball' callback
    fire instead, producing a sentence that described the same wicket
    twice ('after the wicket on the previous ball, the bowler takes a
    big wicket')."""
    facts = build_facts(_base_context(
        raw_event="wicket", previous_event="wicket", streak_type="wicket", streak_count=2,
        cum_wickets=10, cum_wickets_before=9, runs_required=5,
        batting_team="Pakistan", bowling_team="New Zealand",
    ))
    streak_line = next(l for l in facts["lines"] if l.startswith("Streak"))
    assert "2 wickets in 2 balls" in streak_line
    delivery_line = next(l for l in facts["lines"] if l.startswith("Delivery"))
    assert "on the previous ball" not in delivery_line
    chase_line = next(l for l in facts["lines"] if l.startswith("Chase state"))
    assert "New Zealand" in chase_line


def test_dot_and_boundary_streaks_still_suppressed_once_match_is_decided():
    """Unlike a wicket streak, 'pressure building' framing for dots and
    boundaries implies an ongoing trend that stops mattering once
    there's no more play left — these stay suppressed."""
    facts = build_facts(_base_context(
        raw_event="four", streak_type="boundary", streak_count=3,
        cum_wickets=10, cum_wickets_before=9, runs_required=5,
    ))
    assert not any(l.startswith("Streak") for l in facts["lines"])


def test_previous_ball_callback_suppressed_when_wide_streak_is_reported():
    facts = build_facts(_base_context(
        raw_event="wide", previous_event="wide", streak_type="wide", streak_count=2,
    ))
    delivery_line = next(l for l in facts["lines"] if l.startswith("Delivery"))
    assert "on the previous ball" not in delivery_line


def test_previous_ball_callback_suppressed_when_noball_streak_is_reported():
    facts = build_facts(_base_context(
        raw_event="noball", previous_event="noball", streak_type="noball", streak_count=2,
    ))
    delivery_line = next(l for l in facts["lines"] if l.startswith("Delivery"))
    assert "on the previous ball" not in delivery_line
    # And definitely not the free-hit framing either — that's a no-ball
    # streak, not a free hit following a single no-ball.
    assert "free hit" not in delivery_line


def test_previous_ball_callback_suppressed_when_four_streak_is_reported():
    facts = build_facts(_base_context(
        raw_event="four", previous_event="four", streak_type="four", streak_count=3,
    ))
    delivery_line = next(l for l in facts["lines"] if l.startswith("Delivery"))
    assert "on the previous ball" not in delivery_line


def test_previous_ball_callback_still_fires_without_a_matching_streak():
    """The suppression is specific to a genuine streak of the SAME
    type — an isolated single wicket recovery must still get its
    callback."""
    facts = build_facts(_base_context(raw_event="four", previous_event="wicket", streak_type=None))
    delivery_line = next(l for l in facts["lines"] if l.startswith("Delivery"))
    assert "after the wicket on the previous ball," in delivery_line


def test_supporting_facts_capped_to_one_when_both_present():
    """Score and the rate/chase-equation fact are both routinely true on
    the same ball — only one should make it into the prompt at a time,
    so the commentary doesn't always recite the full 'event + score +
    equation' combination."""
    facts = build_facts(_base_context(
        balls_remaining=90,  # outside death overs, so the rate fact can appear
        required_run_rate=8.5, required_run_rate_before=8.0,  # rate changed -> fires
        cum_runs=105, cum_runs_before=100,  # score changed -> fires
    ))
    supporting_labels = {"Score", "Required run rate"}
    present = [l.split(":")[0] for l in facts["lines"] if l.split(":")[0] in supporting_labels]
    assert len(present) <= 1


def test_chase_state_has_five_distinct_tiers():
    """Enough tiers, each with enough variants, that a long spell inside
    one broad probability band doesn't keep repeating the same couple of
    lines."""
    from src.phase6a_explanation import _CHASE_STATE_TIERS
    assert len(_CHASE_STATE_TIERS) == 5
    for _threshold, phrases in _CHASE_STATE_TIERS:
        assert len(phrases) >= 3


def test_chase_state_tier_boundaries_produce_different_wording():
    from src.phase6a_explanation import _CHASE_STATE_TIERS
    tier_phrase_sets = [set(phrases) for _t, phrases in _CHASE_STATE_TIERS]
    for pct, tier_index in ((95, 0), (75, 1), (55, 2), (30, 3), (10, 4)):
        facts = build_facts(_base_context(proba_before=pct / 100, proba_after=pct / 100, swing=0.0))
        chase_line = next(l for l in facts["lines"] if l.startswith("Chase state"))
        content = chase_line.split(":", 1)[1].strip()
        # The chosen phrase (before team-name substitution) must belong
        # to the expected tier and no other.
        matched = any(
            content == p or content == p.replace("{team}", "the batting side")
            for p in tier_phrase_sets[tier_index]
        )
        assert matched, f"{pct}% produced unexpected tier phrasing: {content}"


def test_wide_and_noball_are_distinct():
    wide_prompt = build_user_prompt(_base_context(raw_event="wide"))
    noball_prompt = build_user_prompt(_base_context(raw_event="noball"))
    assert "wide" in wide_prompt.lower()
    assert "no-ball" in noball_prompt.lower()
    assert wide_prompt != noball_prompt


def test_target_line_present_and_fixed():
    prompt = build_user_prompt(_base_context(cum_runs=100, runs_required=50))
    assert "Target: 150" in prompt


def test_probability_rounding_collision_escalates_to_two_decimals():
    # 33.66% and 33.74% both round to "33.7%" at 1 decimal, but the swing
    # (0.08 points) is large enough to clear the "negligible swing" floor
    # — this must escalate to 2 decimals so the two numbers don't
    # visibly contradict each other ("33.7% to 33.7%").
    prompt = build_user_prompt(_base_context(proba_before=0.3366, proba_after=0.3374, swing=0.0008))
    assert "33.66%" in prompt
    assert "33.74%" in prompt


def test_negligible_swing_uses_steady_framing():
    prompt = build_user_prompt(_base_context(proba_before=0.50, proba_after=0.5001, swing=0.0001))
    assert "holds steady" in prompt


def test_flavor_facts_capped_at_two():
    """Even when every optional flavor fact is simultaneously true, at
    most 2 make it into the prompt."""
    facts = build_facts(_base_context(
        cum_runs=102, cum_runs_before=98,  # milestone
        cum_wickets=7,  # risk framing eligible
        streak_type="boundary", streak_count=3,
        rate_momentum="rising", balls_remaining=70,  # not death overs, so momentum can fire
    ))
    flavor_labels = ("Milestone", "Streak", "Momentum", "Risk framing")
    flavor_count = sum(1 for line in facts["lines"] if line.split(":")[0] in flavor_labels)
    assert flavor_count <= 2


def test_score_fact_omitted_when_runs_unchanged():
    prompt = build_user_prompt(_base_context(cum_runs=100, cum_runs_before=100))
    assert "Score:" not in prompt


def test_wickets_fact_only_present_when_a_wicket_actually_fell():
    prompt = build_user_prompt(_base_context(cum_wickets=3, cum_wickets_before=3))
    assert "Wickets:" not in prompt
    prompt2 = build_user_prompt(_base_context(cum_wickets=4, cum_wickets_before=3))
    assert "Wickets:" in prompt2


def test_no_standalone_free_hit_fact_ever_appears():
    """Free-hit context now lives entirely inside the Delivery lead-in
    for a no-ball — there's no separate 'Free hit: ...' line to say the
    same thing twice."""
    prompt = build_user_prompt(_base_context(raw_event="four", previous_event="noball", is_free_hit=True))
    assert "Free hit" not in prompt


def test_no_span_fact_for_a_wide_or_noball():
    """'0 legal deliveries' is internal bookkeeping, not something a
    real commentator ever says — the Delivery fact's own wording
    ('wide called' / 'no-ball called') already covers it."""
    prompt = build_user_prompt(_base_context(raw_event="wide", balls_elapsed=0))
    assert "Span:" not in prompt
    assert "0 legal deliveries" not in prompt


def test_span_fact_still_fires_for_a_genuine_multi_ball_jump():
    prompt = build_user_prompt(_base_context(balls_elapsed=4))
    assert "Span:" in prompt


# --- match already decided ------------------------------------------------

def test_chase_state_gives_a_win_confirmation_when_target_is_reached():
    facts = build_facts(_base_context(
        proba_before=0.609, proba_after=1.0, swing=0.391,
        runs_required=0, batting_team="Netherlands",
    ))
    chase_line = next(l for l in facts["lines"] if l.startswith("Chase state"))
    assert "Netherlands" in chase_line
    assert any(kw in chase_line for kw in ("won", "seal the chase", "complete the chase", "winning moment", "close out the win"))


def test_chase_state_gives_a_loss_confirmation_when_all_out_short_of_target():
    facts = build_facts(_base_context(
        proba_before=0.253, proba_after=0.0, swing=-0.253,
        cum_wickets=10, cum_wickets_before=9, runs_required=5,
        batting_team="Netherlands", bowling_team="Australia",
    ))
    chase_line = next(l for l in facts["lines"] if l.startswith("Chase state"))
    assert "Australia" in chase_line


def test_chase_state_gives_a_loss_confirmation_when_overs_run_out():
    facts = build_facts(_base_context(
        proba_before=0.4, proba_after=0.0, swing=-0.4,
        balls_remaining=0, balls_remaining_before=1, runs_required=5,
        batting_team="Netherlands", bowling_team="Australia",
    ))
    chase_line = next(l for l in facts["lines"] if l.startswith("Chase state"))
    assert "Australia" in chase_line


def test_chase_equation_suppressed_once_match_is_decided():
    """'Needs 5 more off 0 balls' is nonsensical once there are no
    balls left — the confirmative Chase state line covers the result
    instead."""
    prompt = build_user_prompt(_base_context(
        balls_remaining=0, balls_remaining_before=1, runs_required=5,
        cum_wickets=10, cum_wickets_before=9,
    ))
    assert "Chase equation" not in prompt


def test_risk_and_streak_and_momentum_suppressed_once_match_is_decided():
    prompt = build_user_prompt(_base_context(
        cum_wickets=10, cum_wickets_before=9, runs_required=5,
        streak_type="boundary", streak_count=3, rate_momentum="rising",
        balls_remaining=30,  # would otherwise be eligible for Risk framing
    ))
    assert "Risk framing" not in prompt
    assert "Streak" not in prompt
    assert "Momentum" not in prompt


def test_ordinary_ball_still_uses_probability_band_chase_state():
    """An ongoing, undecided chase must not accidentally trip the new
    won/lost branches."""
    facts = build_facts(_base_context(proba_before=0.5, proba_after=0.7, runs_required=50, balls_remaining=60, cum_wickets=3))
    chase_line = next(l for l in facts["lines"] if l.startswith("Chase state"))
    assert "won" not in chase_line
    assert "seal the chase" not in chase_line


def test_mathematically_impossible_chase_is_declared_lost_immediately():
    """Regression test: needing 14 off the last 2 balls can't be done
    even by hitting a six off every remaining ball (6 x 2 = 12 max) —
    this must be treated as decided the moment it becomes true, not
    only once balls_remaining literally reaches 0 or all wickets fall.
    The raw model probability can still show a small nonzero number
    here (it's a statistical estimate, not a hard rule); Chase state
    must not defer to it."""
    facts = build_facts(_base_context(
        proba_before=0.15, proba_after=0.02, swing=-0.13,
        runs_required=14, balls_remaining=2, required_run_rate=42.0,
        cum_wickets=7, cum_wickets_before=7,
        batting_team="Nepal", bowling_team="Australia",
    ))
    chase_line = next(l for l in facts["lines"] if l.startswith("Chase state"))
    assert "Australia" in chase_line
    # No probability-band phrasing ("up against it", "would need
    # something special") should leak through once it's genuinely
    # decided — only the confirmative loss wording.
    assert "Nepal are up against it" not in chase_line
    # Chase equation is nonsensical once decided ("needs 14 more off 2
    # balls" reads like it's still live) and must not appear either.
    assert not any(l.startswith("Chase equation") for l in facts["lines"])


def test_required_run_rate_past_36_alone_is_enough_to_declare_lost():
    """A required run rate above 36 (more than a six a ball) is
    mathematically unreachable on its own, even if runs_required/
    balls_remaining weren't also checked."""
    facts = build_facts(_base_context(
        runs_required=20, balls_remaining=3, required_run_rate=40.0,
        batting_team="Nepal", bowling_team="Australia",
    ))
    chase_line = next(l for l in facts["lines"] if l.startswith("Chase state"))
    assert "Australia" in chase_line


# --- validate_output -----------------------------------------------------

def _sample_facts():
    return build_facts(_base_context(
        proba_before=0.62, proba_after=0.47, swing=-0.15,
        cum_runs=88, cum_wickets=5, balls_remaining=24, runs_required=40,
        batting_team="India", bowling_team="Australia", raw_event="wicket",
    ))


def test_validate_accepts_clean_output_using_only_given_facts():
    facts = _sample_facts()
    text = f"India slides from {facts['proba_before_fmt']}% to {facts['proba_after_fmt']}% as Australia strikes."
    assert validate_output(text, facts) == []


def test_validate_flags_all_caps_leak():
    facts = _sample_facts()
    text = "WICKET falls and the win probability drops sharply here."
    issues = validate_output(text, facts)
    assert "all_caps_leak" in issues


def test_validate_flags_hallucinated_number():
    facts = _sample_facts()
    text = "The win probability crashes to 12.34% after that strike."
    issues = validate_output(text, facts)
    assert any(i.startswith("unrecognized_number") for i in issues)


def test_validate_tolerates_small_connective_numbers():
    facts = _sample_facts()
    text = "Two balls left in the over as pressure builds on the batting side."
    assert validate_output(text, facts) == []


def test_validate_flags_unrecognized_team_name():
    facts = _sample_facts()
    text = "The strike leaves Pakistan celebrating wildly out there."
    issues = validate_output(text, facts)
    assert any(i.startswith("unrecognized_name") for i in issues)


def test_validate_allows_known_team_names():
    facts = _sample_facts()
    text = "India will need to rebuild as Australia celebrates the breakthrough."
    assert validate_output(text, facts) == []


def test_validate_does_not_flag_sentence_initial_capitalization():
    facts = _sample_facts()
    text = "Massive moment right there as the wicket falls."
    assert validate_output(text, facts) == []


def test_validate_flags_empty_output():
    facts = _sample_facts()
    assert validate_output("", facts) == ["empty_output"]
    assert validate_output("   ", facts) == ["empty_output"]


def test_validate_flags_incomplete_output():
    facts = _sample_facts()
    text = "Australia strikes as the win probability slides sharply and the"
    issues = validate_output(text, facts)
    assert "incomplete_output" in issues


def test_validate_accepts_complete_output():
    facts = _sample_facts()
    text = "Australia strikes as the win probability slides sharply here."
    assert "incomplete_output" not in validate_output(text, facts)


def test_validate_flags_label_leak():
    """A real, observed bug: the model echoed a fact's own category
    label back verbatim ('Delivery: after the wide on the previous
    ball, ...') instead of just using the content after the colon."""
    facts = _sample_facts()
    text = "Delivery: Australia strikes as the win probability slides sharply."
    issues = validate_output(text, facts)
    assert any(i.startswith("label_leak") for i in issues)


def test_validate_does_not_flag_output_with_no_labels():
    facts = _sample_facts()
    text = "Australia strikes as the win probability slides sharply here."
    assert validate_output(text, facts) == []


# --- deterministic fallback ----------------------------------------------

def test_deterministic_fallback_contains_correct_probability_numbers():
    facts = _sample_facts()
    sentence = _deterministic_fallback_sentence(facts)
    assert facts["proba_before_fmt"] in sentence
    assert facts["proba_after_fmt"] in sentence


def test_deterministic_fallback_never_raises_without_delivery_phrase():
    facts = build_facts(_base_context())  # no raw_event -> no delivery phrase
    sentence = _deterministic_fallback_sentence(facts)
    assert isinstance(sentence, str) and sentence


def test_deterministic_fallback_passes_its_own_validation():
    """The fallback sentence is what the user sees when both providers
    fail twice in a row — it must itself be clean by the same rules used
    to judge the model's own output."""
    facts = _sample_facts()
    sentence = _deterministic_fallback_sentence(facts)
    assert validate_output(sentence, facts) == []


# --- capitalization ------------------------------------------------------

def test_capitalize_first_fixes_a_lowercase_opening():
    """A provider following the 'use this fact's exact wording' rule can
    open its sentence with a lowercase fact string verbatim (facts like
    Delivery are deliberately lowercase for normal mid-sentence use) —
    the final text must never reach the user starting with a lowercase
    letter."""
    assert _capitalize_first("the batsman finds the gap for four.") == "The batsman finds the gap for four."


def test_capitalize_first_leaves_already_capitalized_text_unchanged():
    assert _capitalize_first("Australia strikes as the total holds.") == "Australia strikes as the total holds."


def test_capitalize_first_handles_leading_punctuation():
    """A sentence that legitimately opens with punctuation (an
    exclamation lead-in, a quote) should still get its first letter
    capitalized, not the punctuation itself skipped incorrectly."""
    assert _capitalize_first("\"the bowler strikes!\"") == "\"The bowler strikes!\""


def test_capitalize_first_handles_empty_string():
    assert _capitalize_first("") == ""