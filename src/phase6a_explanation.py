"""
Phase 6a — LLM explanation layer

Given a ball's before/after match state and the model's probability swing,
generate a short natural-language explanation of why the win probability
moved. Reuses the same Groq client pattern as CodeSage.

Set your GROQ_API_KEY as an environment variable before running:
  Windows (PowerShell):  $env:GROQ_API_KEY="your_key_here"
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

EXPLANATION_SYSTEM_PROMPT = """You are a concise cricket commentator explaining
win-probability swings in a T20 match. Given the match state before and after
a ball, write ONE sentence (under 35 words) explaining the swing.

You MUST include, in your own words:
  - the probability change itself (e.g. "probability dropped from 62% to 47%")
  - at least one concrete match-state number that drove it (wickets down,
    balls remaining, or required run rate)

Do not write a vague summary — cite the actual numbers given to you."""


def build_user_prompt(ball_context: dict) -> str:
    return (
        f"Event: {ball_context['event_type']}\n"
        f"Probability before: {ball_context['proba_before']:.2f}\n"
        f"Probability after: {ball_context['proba_after']:.2f}\n"
        f"Swing: {ball_context['swing']:+.2f}\n"
        f"Score: {ball_context['cum_runs']}/{ball_context['cum_wickets']}\n"
        f"Balls remaining: {ball_context['balls_remaining']}\n"
        f"Required run rate: {ball_context['required_run_rate']:.1f}\n"
    )


def generate_explanation(ball_context: dict) -> str:
    """
    ball_context expected keys: event_type, proba_before, proba_after, swing,
    cum_runs, cum_wickets, balls_remaining, required_run_rate
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(ball_context)},
        ],
        max_tokens=80,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


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
    print(generate_explanation(example))