from types import SimpleNamespace
from typing import Any, AsyncGenerator

import pytest
from fastapi import HTTPException

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.llm_as_a_judge import (
    LLMAsAJudgeGuardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail import (
    unified_guardrail as unified_module,
)
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
    UnifiedLLMGuardrails,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CallTypes, Delta, ModelResponseStream, StreamingChoices


def _llm_judge_guardrail() -> LLMAsAJudgeGuardrail:
    return LLMAsAJudgeGuardrail(
        guardrail_name="judge-quality",
        judge_model="gpt-4o-mini",
        criteria=[
            {
                "name": "quality",
                "description": "Response must be high quality",
                "weight": 100,
            }
        ],
        overall_threshold=80,
        on_failure="block",
    )


def test_llm_as_a_judge_selected_guardrail_matches_post_call_event() -> None:
    guardrail = _llm_judge_guardrail()

    assert guardrail.should_run_guardrail(
        data={"metadata": {"guardrails": ["judge-quality"]}},
        event_type=GuardrailEventHooks.post_call,
    )


@pytest.mark.asyncio
async def test_llm_as_a_judge_uses_proxy_router_for_judge_model_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MockRouter:
        def __init__(self) -> None:
            self.calls = 0

        async def acompletion(self, **kwargs: Any) -> Any:
            self.calls += 1
            assert kwargs["model"] == "proxy-judge-alias"
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"overall_score": 10}')
                    )
                ]
            )

    async def direct_litellm_call_should_not_be_used(**kwargs: Any) -> Any:
        raise RuntimeError("direct provider call cannot resolve proxy model alias")

    router = MockRouter()
    guardrail = LLMAsAJudgeGuardrail(
        guardrail_name="judge-quality",
        judge_model="proxy-judge-alias",
        criteria=[
            {
                "name": "quality",
                "description": "Response must be high quality",
                "weight": 100,
            }
        ],
        overall_threshold=80,
        on_failure="block",
        llm_router=router,
    )
    monkeypatch.setattr(litellm, "acompletion", direct_litellm_call_should_not_be_used)

    with pytest.raises(HTTPException) as exc_info:
        await guardrail.apply_guardrail(
            inputs={"texts": ["unsafe response"]},
            request_data={"messages": [{"role": "user", "content": "hello"}]},
            input_type="response",
        )

    assert exc_info.value.status_code == 422
    assert router.calls == 1


@pytest.mark.asyncio
async def test_llm_as_a_judge_blocks_selected_streaming_chat_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guardrail = _llm_judge_guardrail()
    judge_calls = 0

    async def mock_run_judge(
        messages: list[dict[str, Any]], response_text: str
    ) -> dict:
        nonlocal judge_calls
        judge_calls += 1
        assert messages == [{"role": "user", "content": "hello"}]
        assert "unsafe response" in response_text
        return {
            "overall_score": 10,
            "verdicts": [
                {
                    "criterion_name": "quality",
                    "score": 10,
                    "reasoning": "low quality",
                    "passed": False,
                    "weight": 100,
                }
            ],
        }

    monkeypatch.setattr(guardrail, "_run_judge", mock_run_judge)
    original_mappings = unified_module.endpoint_guardrail_translation_mappings
    unified_module.endpoint_guardrail_translation_mappings = {
        CallTypes.acompletion: unified_module.load_guardrail_translation_mappings()[
            CallTypes.acompletion
        ],
    }

    async def mock_stream() -> AsyncGenerator[ModelResponseStream, None]:
        yield ModelResponseStream(
            choices=[
                StreamingChoices(
                    delta=Delta(content="unsafe response", role="assistant"),
                    finish_reason=None,
                )
            ],
            model="gpt-4o-mini",
        )
        yield ModelResponseStream(
            choices=[
                StreamingChoices(
                    delta=Delta(content=None),
                    finish_reason="stop",
                )
            ],
            model="gpt-4o-mini",
        )

    request_data = {
        "guardrail_to_apply": guardrail,
        "metadata": {"guardrails": ["judge-quality"]},
        "messages": [{"role": "user", "content": "hello"}],
        "model": "gpt-4o-mini",
    }
    emitted_chunks: list[ModelResponseStream] = []

    try:
        with pytest.raises(HTTPException) as exc_info:
            async for (
                chunk
            ) in UnifiedLLMGuardrails().async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(
                    api_key="test-key",
                    request_route="/v1/chat/completions",
                ),
                response=mock_stream(),
                request_data=request_data,
            ):
                emitted_chunks.append(chunk)
    finally:
        unified_module.endpoint_guardrail_translation_mappings = original_mappings

    assert exc_info.value.status_code == 422
    assert judge_calls == 1
    assert emitted_chunks == []
