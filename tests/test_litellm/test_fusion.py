"""
Tests for litellm.fusion and litellm.afusion.

Uses unittest.mock to avoid real API calls.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import litellm
from litellm.fusion.main import (
    FusionStrategy,
    _build_judge_messages,
    _merge_usage,
    _sum_usage,
    afusion,
    fusion,
)
from litellm.types.utils import Choices, Message, ModelResponse, Usage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(content: str, model: str = "gpt-4o", prompt_tokens: int = 10) -> ModelResponse:
    resp = ModelResponse(
        id=f"mock-{model}",
        choices=[
            Choices(
                index=0,
                message=Message(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        model=model,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=len(content.split()),
            total_tokens=prompt_tokens + len(content.split()),
        ),
    )
    return resp


MESSAGES = [{"role": "user", "content": "What is 2+2?"}]
PANEL_MODELS = ["gpt-4o", "claude-3-5-sonnet", "gemini-2.0-flash"]
JUDGE_MODEL = "gpt-4o-judge"  # distinct from panel to avoid index confusion


def _panel_mock(panel_responses, judge_response=None):
    """Return an async mock that routes by model name."""
    async def mock_acompletion(**kwargs):
        m = kwargs["model"]
        if judge_response is not None and m == JUDGE_MODEL:
            return judge_response
        if m in PANEL_MODELS:
            return panel_responses[PANEL_MODELS.index(m)]
        return panel_responses[0]
    return mock_acompletion


# ---------------------------------------------------------------------------
# Unit: _build_judge_messages
# ---------------------------------------------------------------------------

class TestBuildJudgeMessages:
    def _panel(self):
        return [
            _make_response("The answer is 4.", "gpt-4o"),
            _make_response("4", "claude-3-5-sonnet"),
        ]

    def test_contains_user_question(self):
        msgs = _build_judge_messages(MESSAGES, self._panel(), PANEL_MODELS[:2], "single_judge")
        combined = " ".join(m["content"] for m in msgs)
        assert "2+2" in combined

    def test_contains_panel_responses(self):
        msgs = _build_judge_messages(MESSAGES, self._panel(), PANEL_MODELS[:2], "single_judge")
        combined = " ".join(m["content"] for m in msgs)
        assert "The answer is 4." in combined
        assert "4" in combined

    def test_strategy_majority_vote(self):
        msgs = _build_judge_messages(MESSAGES, self._panel(), PANEL_MODELS[:2], "majority_vote")
        system_msg = next(m for m in msgs if m["role"] == "system")
        assert "Select the single best response" in system_msg["content"]

    def test_strategy_best_of_n(self):
        msgs = _build_judge_messages(MESSAGES, self._panel(), PANEL_MODELS[:2], "best_of_n")
        system_msg = next(m for m in msgs if m["role"] == "system")
        assert "Score each response" in system_msg["content"]

    def test_ends_with_user_message(self):
        msgs = _build_judge_messages(MESSAGES, self._panel(), PANEL_MODELS[:2], "single_judge")
        assert msgs[-1]["role"] == "user"


# ---------------------------------------------------------------------------
# Unit: _sum_usage / _merge_usage
# ---------------------------------------------------------------------------

class TestUsageHelpers:
    def test_sum_usage(self):
        responses = [
            _make_response("hello", prompt_tokens=10),       # completion=1
            _make_response("world more words", prompt_tokens=20),  # completion=3
        ]
        u = _sum_usage(responses)
        assert u.prompt_tokens == 30
        assert u.completion_tokens == 4

    def test_merge_usage_sums_panel_and_judge(self):
        panel = [
            _make_response("hello", prompt_tokens=10),
            _make_response("world more words", prompt_tokens=20),
        ]
        judge = _make_response("final answer here today", prompt_tokens=5)
        merged = _merge_usage(panel, judge)
        assert merged.prompt_tokens == 35          # 10+20+5
        assert merged.completion_tokens == 1 + 3 + 4

    def test_handles_none_usage(self):
        panel = [_make_response("hi")]
        panel[0].usage = None
        judge = _make_response("answer", prompt_tokens=5)
        merged = _merge_usage(panel, judge)
        assert merged.prompt_tokens == 5


# ---------------------------------------------------------------------------
# afusion — panel only (no judge)
# ---------------------------------------------------------------------------

class TestAFusionPanelOnly:
    @pytest.fixture
    def panel_responses(self):
        return [_make_response(f"Answer {m}", m) for m in PANEL_MODELS]

    @pytest.mark.asyncio
    async def test_returns_list_of_model_responses(self, panel_responses):
        with patch("litellm.acompletion", side_effect=_panel_mock(panel_responses)):
            result = await afusion(models=PANEL_MODELS, messages=MESSAGES)

        assert isinstance(result, list)
        assert len(result) == len(PANEL_MODELS)
        assert all(isinstance(r, ModelResponse) for r in result)

    @pytest.mark.asyncio
    async def test_returns_all_panel_contents(self, panel_responses):
        with patch("litellm.acompletion", side_effect=_panel_mock(panel_responses)):
            result = await afusion(models=PANEL_MODELS, messages=MESSAGES)

        contents = {r.choices[0].message.content for r in result}
        assert contents == {f"Answer {m}" for m in PANEL_MODELS}

    @pytest.mark.asyncio
    async def test_no_judge_call_when_judge_model_omitted(self, panel_responses):
        called_models: list[str] = []

        async def mock_acompletion(**kwargs):
            called_models.append(kwargs["model"])
            return panel_responses[PANEL_MODELS.index(kwargs["model"])]

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            await afusion(models=PANEL_MODELS, messages=MESSAGES)

        # Exactly the panel models, no judge
        assert sorted(called_models) == sorted(PANEL_MODELS)

    @pytest.mark.asyncio
    async def test_partial_failure_returns_remaining(self, panel_responses):
        async def mock_acompletion(**kwargs):
            if kwargs["model"] == "gemini-2.0-flash":
                raise RuntimeError("API error")
            return panel_responses[PANEL_MODELS.index(kwargs["model"])]

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            result = await afusion(models=PANEL_MODELS, messages=MESSAGES)

        assert isinstance(result, list)
        assert len(result) == 2  # gemini dropped


# ---------------------------------------------------------------------------
# afusion — with judge
# ---------------------------------------------------------------------------

class TestAFusionWithJudge:
    @pytest.fixture
    def panel_responses(self):
        return [_make_response(f"Answer {m}", m) for m in PANEL_MODELS]

    @pytest.fixture
    def judge_response(self):
        return _make_response("Synthesized final answer", JUDGE_MODEL)

    @pytest.mark.asyncio
    async def test_returns_single_model_response(self, panel_responses, judge_response):
        with patch("litellm.acompletion", side_effect=_panel_mock(panel_responses, judge_response)):
            result = await afusion(models=PANEL_MODELS, judge_model=JUDGE_MODEL, messages=MESSAGES)

        assert isinstance(result, ModelResponse)
        assert result.choices[0].message.content == "Synthesized final answer"

    @pytest.mark.asyncio
    async def test_calls_all_panel_and_judge(self, panel_responses, judge_response):
        called: list[str] = []

        async def mock_acompletion(**kwargs):
            called.append(kwargs["model"])
            return judge_response if kwargs["model"] == JUDGE_MODEL else panel_responses[PANEL_MODELS.index(kwargs["model"])]

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            await afusion(models=PANEL_MODELS, judge_model=JUDGE_MODEL, messages=MESSAGES)

        assert set(PANEL_MODELS).issubset(set(called))
        assert JUDGE_MODEL in called

    @pytest.mark.asyncio
    async def test_include_panel_true_by_default(self, panel_responses, judge_response):
        with patch("litellm.acompletion", side_effect=_panel_mock(panel_responses, judge_response)):
            result = await afusion(models=PANEL_MODELS, judge_model=JUDGE_MODEL, messages=MESSAGES)

        fusion_meta = result._hidden_params["fusion"]
        assert "panel_responses" in fusion_meta
        assert len(fusion_meta["panel_responses"]) == len(PANEL_MODELS)
        assert all(isinstance(r, ModelResponse) for r in fusion_meta["panel_responses"])

    @pytest.mark.asyncio
    async def test_include_panel_false_excludes_panel(self, panel_responses, judge_response):
        with patch("litellm.acompletion", side_effect=_panel_mock(panel_responses, judge_response)):
            result = await afusion(
                models=PANEL_MODELS,
                judge_model=JUDGE_MODEL,
                messages=MESSAGES,
                include_panel=False,
            )

        fusion_meta = result._hidden_params["fusion"]
        assert "panel_responses" not in fusion_meta

    @pytest.mark.asyncio
    async def test_fusion_metadata_always_present(self, panel_responses, judge_response):
        with patch("litellm.acompletion", side_effect=_panel_mock(panel_responses, judge_response)):
            result = await afusion(models=PANEL_MODELS, judge_model=JUDGE_MODEL, messages=MESSAGES)

        meta = result._hidden_params["fusion"]
        assert meta["judge_model"] == JUDGE_MODEL
        assert set(meta["panel_models"]) == set(PANEL_MODELS)

    @pytest.mark.asyncio
    async def test_usage_merged(self, panel_responses, judge_response):
        with patch("litellm.acompletion", side_effect=_panel_mock(panel_responses, judge_response)):
            result = await afusion(models=PANEL_MODELS, judge_model=JUDGE_MODEL, messages=MESSAGES)

        assert result.usage is not None
        assert result.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_partial_panel_failure_continues(self, panel_responses, judge_response):
        async def mock_acompletion(**kwargs):
            if kwargs["model"] == "gemini-2.0-flash":
                raise RuntimeError("API error")
            return judge_response if kwargs["model"] == JUDGE_MODEL else panel_responses[PANEL_MODELS.index(kwargs["model"])]

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            result = await afusion(models=PANEL_MODELS, judge_model=JUDGE_MODEL, messages=MESSAGES)

        assert isinstance(result, ModelResponse)
        # Only 2 panel models survived
        assert len(result._hidden_params["fusion"]["panel_models"]) == 2

    @pytest.mark.asyncio
    async def test_all_panel_fail_raises(self):
        async def mock_acompletion(**kwargs):
            raise RuntimeError("All failed")

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            with pytest.raises(RuntimeError, match="all panel models failed"):
                await afusion(models=PANEL_MODELS, judge_model=JUDGE_MODEL, messages=MESSAGES)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("strategy", ["single_judge", "majority_vote", "best_of_n"])
    async def test_all_strategies(self, strategy, panel_responses, judge_response):
        with patch("litellm.acompletion", side_effect=_panel_mock(panel_responses, judge_response)):
            result = await afusion(
                models=PANEL_MODELS,
                judge_model=JUDGE_MODEL,
                messages=MESSAGES,
                strategy=strategy,
            )
        assert isinstance(result, ModelResponse)

    @pytest.mark.asyncio
    async def test_invalid_strategy_raises(self, panel_responses, judge_response):
        with patch("litellm.acompletion", side_effect=_panel_mock(panel_responses, judge_response)):
            with pytest.raises(ValueError, match="unknown strategy"):
                await afusion(
                    models=PANEL_MODELS,
                    judge_model=JUDGE_MODEL,
                    messages=MESSAGES,
                    strategy="nonexistent",  # type: ignore
                )


# ---------------------------------------------------------------------------
# Common error cases
# ---------------------------------------------------------------------------

class TestAFusionKwargs:
    """Cover temperature/max_tokens/timeout forwarding and system-prompt preservation."""

    @pytest.fixture
    def panel_responses(self):
        return [_make_response(f"Answer {m}", m) for m in PANEL_MODELS]

    @pytest.fixture
    def judge_response(self):
        return _make_response("Synthesized final answer", JUDGE_MODEL)

    @pytest.mark.asyncio
    async def test_temperature_max_tokens_timeout_forwarded(self, panel_responses, judge_response):
        """Lines 309, 311, 313, 336: temperature/max_tokens/timeout set on panel_kwargs and judge_kwargs."""
        captured: list[dict] = []

        async def mock_acompletion(**kwargs):
            captured.append(dict(kwargs))
            return judge_response if kwargs["model"] == JUDGE_MODEL else panel_responses[PANEL_MODELS.index(kwargs["model"])]

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            await afusion(
                models=PANEL_MODELS,
                judge_model=JUDGE_MODEL,
                messages=MESSAGES,
                temperature=0.5,
                max_tokens=128,
                timeout=10.0,
            )

        panel_calls = [c for c in captured if c["model"] in PANEL_MODELS]
        judge_calls = [c for c in captured if c["model"] == JUDGE_MODEL]
        assert all(c.get("temperature") == 0.5 for c in panel_calls)
        assert all(c.get("max_tokens") == 128 for c in panel_calls)
        assert all(c.get("timeout") == 10.0 for c in panel_calls)
        assert all(c.get("timeout") == 10.0 for c in judge_calls)

    @pytest.mark.asyncio
    async def test_existing_system_prompt_preserved(self, panel_responses, judge_response):
        """Lines 126-127: system message in original_messages is carried into judge prompt."""
        messages_with_system = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        captured_judge_messages: list = []

        async def mock_acompletion(**kwargs):
            if kwargs["model"] == JUDGE_MODEL:
                captured_judge_messages.extend(kwargs["messages"])
            return judge_response if kwargs["model"] == JUDGE_MODEL else panel_responses[PANEL_MODELS.index(kwargs["model"])]

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            await afusion(
                models=PANEL_MODELS,
                judge_model=JUDGE_MODEL,
                messages=messages_with_system,
            )

        system_contents = [m["content"] for m in captured_judge_messages if m.get("role") == "system"]
        assert any("helpful assistant" in c for c in system_contents)


class TestCommonErrors:
    @pytest.mark.asyncio
    async def test_empty_models_raises(self):
        with pytest.raises(ValueError, match="non-empty list"):
            await afusion(models=[], messages=MESSAGES)

    @pytest.mark.asyncio
    async def test_empty_models_with_judge_raises(self):
        with pytest.raises(ValueError, match="non-empty list"):
            await afusion(models=[], judge_model=JUDGE_MODEL, messages=MESSAGES)


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------

class TestFusionSync:
    def test_callable_via_litellm(self):
        assert callable(litellm.fusion)
        assert callable(litellm.afusion)

    def test_panel_only_returns_list(self):
        panel_resps = [_make_response(f"Panel {m}", m) for m in PANEL_MODELS]

        with patch("litellm.acompletion", side_effect=_panel_mock(panel_resps)):
            result = fusion(models=PANEL_MODELS, messages=MESSAGES)

        assert isinstance(result, list)
        assert len(result) == len(PANEL_MODELS)

    def test_with_judge_returns_model_response(self):
        panel_resps = [_make_response(f"Panel {m}", m) for m in PANEL_MODELS]
        judge_resp = _make_response("Sync result", JUDGE_MODEL)

        with patch("litellm.acompletion", side_effect=_panel_mock(panel_resps, judge_resp)):
            result = fusion(models=PANEL_MODELS, judge_model=JUDGE_MODEL, messages=MESSAGES)

        assert isinstance(result, ModelResponse)
        assert result.choices[0].message.content == "Sync result"

    def test_include_panel_false(self):
        panel_resps = [_make_response(f"Panel {m}", m) for m in PANEL_MODELS]
        judge_resp = _make_response("Sync result", JUDGE_MODEL)

        with patch("litellm.acompletion", side_effect=_panel_mock(panel_resps, judge_resp)):
            result = fusion(
                models=PANEL_MODELS,
                judge_model=JUDGE_MODEL,
                messages=MESSAGES,
                include_panel=False,
            )

        assert "panel_responses" not in result._hidden_params["fusion"]
