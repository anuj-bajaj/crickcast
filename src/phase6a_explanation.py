"""
Phase 6a — LLM explanation layer

Given a ball's before/after match state and the model's probability swing,
generate a short natural-language explanation of why the win probability
moved. Reuses the same Groq client pattern as CodeSage.

Set your GROQ_API_KEY as an environment variable before running:
  Windows (PowerShell):  $env:GROQ_API_KEY="your_key_here"
"""

import logging
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)

# Built lazily (and re-checked on every call) rather than at import time, so
# a missing/invalid GROQ_API_KEY doesn't take the whole module — and by
# extension the FastAPI app that imports it — down before it even starts.
_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        _client = Groq(api_key=api_key)
    return _client


EXPLANATION_SYSTEM_PROMPT = """You are a sharp T20 cricket commentator calling
the win-probability swing after a ball, for a viewer watching live.

Write 1-2 sentences (35-55 words total) in a real commentator's voice — not
a dry stats readout. Convey the stakes and momentum, the way a broadcaster
would, while staying strictly factual.

You MUST include, in your own words:
  - the probability change (e.g. "win probability slides from 62% to 47%")
  - at least TWO concrete match-state numbers that explain WHY (pick from:
    wickets down, balls/overs remaining, required run rate, runs still
    needed, target) — weave them in naturally, don't just list them
  - a sense of what this means for the chase (e.g. pressure mounting,
    the equation easing, the required rate becoming unrealistic)

If team names are given in the prompt, use them naturally in place of "the
batting team" / "the bowling team" (e.g. "India's win probability slides"
rather than "the batting team's win probability slides"). If no team names
are given, fall back to "the batting team" / "the bowling team" — never
invent a team name that wasn't provided.

Do not invent player names, do not invent events beyond what's given, do
not use vague filler ("a lot has changed"). Every number you use must come
from what's provided. Vary your sentence construction — don't default to
the same "Probability dropped from X to Y" template every time."""


def build_user_prompt(ball_context: dict) -> str:
    target = ball_context['cum_runs'] + ball_context.get('runs_required', 0)
    lines = [
        f"Event: {ball_context['event_type']}",
        f"Probability before: {ball_context['proba_before']:.2f}",
        f"Probability after: {ball_context['proba_after']:.2f}",
        f"Swing: {ball_context['swing']:+.2f}",
        f"Score: {ball_context['cum_runs']}/{ball_context['cum_wickets']}",
        f"Target: {target}",
        f"Runs still needed: {ball_context.get('runs_required', 'unknown')}",
        f"Balls remaining: {ball_context['balls_remaining']}",
        f"Required run rate: {ball_context['required_run_rate']:.1f}",
    ]
    batting_team = ball_context.get("batting_team")
    bowling_team = ball_context.get("bowling_team")
    if batting_team:
        lines.append(f"Batting team: {batting_team}")
    if bowling_team:
        lines.append(f"Bowling team: {bowling_team}")
    return "\n".join(lines) + "\n"


def generate_explanation(ball_context: dict) -> str | None:
    """
    ball_context expected keys: event_type, proba_before, proba_after, swing,
    cum_runs, cum_wickets, balls_remaining, required_run_rate

    Returns None (instead of raising) on any failure — a down/rate-limited
    Groq API or a missing key should degrade the explanation, not take out
    a /predict response that already has a perfectly good win_probability.
    Callers should treat None as "no commentary available this time."
    """
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(ball_context)},
            ],
            max_tokens=140,
            temperature=0.6,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        logger.exception("Explanation generation failed; continuing without commentary.")
        return None


if __name__ == "__main__":
    # Quick manual test with a made-up wicket scenario
    example = {
        "event_type": "wicket",
        "proba_before": 0.62,
        "proba_after": 0.47,
        "swing": -0.15,
        "cum_runs": 88,
        "cum_wickets": 5,
        "balls_remaining": 24,
        "required_run_rate": 11.25,
    }
    result = generate_explanation(example)
    print(result if result is not None else "(explanation unavailable)")