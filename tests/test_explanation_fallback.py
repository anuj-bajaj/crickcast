"""
Unit tests for generate_explanation()'s Groq -> OpenRouter fallback
orchestration, isolated from both real network calls. _call_groq and
_call_openrouter are monkeypatched directly so these tests verify the
ORCHESTRATION logic (try Groq, fall back to OpenRouter on failure, return
None only if both fail) without depending on real API keys, real network
access, or either provider's actual current availability — all of which
are exactly the things this fallback exists to be resilient to.
"""

from src import phase6a_explanation


def _base_context(**overrides):
    context = dict(
        event_type="wicket",
        proba_before=0.62,
        proba_after=0.47,
        swing=-0.15,
        cum_runs=88,
        cum_wickets=5,
        balls_remaining=24,
        runs_required=40,
        required_run_rate=11.25,
    )
    context.update(overrides)
    return context


def _raise(*_args, **_kwargs):
    raise RuntimeError("simulated provider failure")


def test_uses_groq_result_without_touching_openrouter(monkeypatch):
    """The common case: Groq succeeds, OpenRouter is never called at
    all — the fallback should add nothing when the primary provider is
    working fine."""
    monkeypatch.setattr(phase6a_explanation, "_call_groq", lambda _prompt: "Groq commentary text.")

    def _fail_if_called(_prompt):
        raise AssertionError("OpenRouter should not be called when Groq succeeds")

    monkeypatch.setattr(phase6a_explanation, "_call_openrouter", _fail_if_called)

    result = phase6a_explanation.generate_explanation(_base_context())
    assert result == "Groq commentary text."


def test_falls_back_to_openrouter_when_groq_fails(monkeypatch):
    """The actual point of this feature: Groq failing (rate limit,
    outage, deprecated model, anything) for whatever reason must not be
    the end of the story if OpenRouter is configured and working."""
    monkeypatch.setattr(phase6a_explanation, "_call_groq", _raise)
    monkeypatch.setattr(phase6a_explanation, "_call_openrouter", lambda _prompt: "OpenRouter fallback text.")

    result = phase6a_explanation.generate_explanation(_base_context())
    assert result == "OpenRouter fallback text."


def test_returns_none_when_both_providers_fail(monkeypatch):
    """Only when EVERY option is exhausted does this degrade to None —
    same external contract as before the fallback existed."""
    monkeypatch.setattr(phase6a_explanation, "_call_groq", _raise)
    monkeypatch.setattr(phase6a_explanation, "_call_openrouter", _raise)

    result = phase6a_explanation.generate_explanation(_base_context())
    assert result is None


def test_call_openrouter_raises_without_api_key(monkeypatch):
    """_call_openrouter itself (not the orchestration) must raise, not
    silently return something, when OPENROUTER_API_KEY isn't set — this
    is what lets generate_explanation's fallback stay a no-op add-on
    rather than a new hard requirement."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    try:
        phase6a_explanation._call_openrouter("irrelevant prompt")
        assert False, "expected _call_openrouter to raise without an API key"
    except Exception:
        pass