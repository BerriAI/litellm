from typing import Final, Literal
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.chat_completion_agentic_loop import (
    _execute_chat_completion_agentic_plan,
)
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.integrations.custom_logger import (
    AgenticLoopPlan,
    AgenticLoopRequestPatch,
)
from litellm.types.utils import ModelResponse


@pytest.mark.asyncio
@pytest.mark.parametrize("execution_path", ("http", "sdk"))
@pytest.mark.parametrize("use_patch_model", (False, True), ids=("original-model", "patch-model"))
@pytest.mark.parametrize(
    ("model_name", "custom_llm_provider", "expected_model"),
    (
        ("zai-org/GLM-5.3-Flash", "hosted_vllm", "hosted_vllm/zai-org/GLM-5.3-Flash"),
        ("hosted_vllm/zai-org/GLM-5.3-Flash", "hosted_vllm", "hosted_vllm/zai-org/GLM-5.3-Flash"),
        ("hosted_vllm/zai-org/GLM-5.3-Flash", "", "hosted_vllm/zai-org/GLM-5.3-Flash"),
        ("openai/gpt-4o", "hosted_vllm", "openai/gpt-4o"),
    ),
    ids=("organization-model", "already-prefixed", "no-provider", "cross-provider"),
)
async def test_agentic_followup_preserves_provider_prefix(
    execution_path: Literal["http", "sdk"],
    use_patch_model: bool,
    model_name: str,
    custom_llm_provider: str,
    expected_model: str,
) -> None:
    model: Final = "original-model" if use_patch_model else model_name
    plan: Final = AgenticLoopPlan(
        run_agentic_loop=True,
        request_patch=AgenticLoopRequestPatch(
            model=model_name if use_patch_model else None,
            messages=[{"role": "user", "content": "Continue"}],
        ),
    )
    followup: Final = AsyncMock(wraps=litellm.acompletion)

    with patch("litellm.acompletion", new=followup):
        response: Final = (
            await BaseLLMHTTPHandler()._execute_chat_completion_agentic_plan(
                plan=plan,
                model=model,
                messages=[],
                optional_params={"mock_response": "Follow-up complete"},
                kwargs={"custom_llm_provider": custom_llm_provider},
                custom_llm_provider=custom_llm_provider,
                depth=0,
                max_loops=3,
                fingerprints=[],
                fingerprint="followup",
            )
            if execution_path == "http"
            else await _execute_chat_completion_agentic_plan(
                plan=plan,
                callback=CustomLogger(),
                model=model,
                optional_params={"mock_response": "Follow-up complete"},
                kwargs={"custom_llm_provider": custom_llm_provider},
                logging_obj=None,
                custom_llm_provider=custom_llm_provider,
                depth=0,
                max_loops=3,
                fingerprints=[],
                fingerprint="followup",
            )
        )

    followup.assert_awaited_once()
    assert followup.await_args is not None
    assert followup.await_args.kwargs["model"] == expected_model
    assert isinstance(response, ModelResponse)
    assert response.model == expected_model.split("/", 1)[1]
