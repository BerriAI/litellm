"""Pin ``ProxyLogging.post_call_success_hook``."""

from __future__ import annotations

from typing import Any, Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import CallTypes, UserAPIKeyAuth
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs, LLMResponseTypes, ModelResponse
from litellm.utils import async_post_call_success_deployment_hook


@pytest.fixture(autouse=True)
def _clear_caps_cache():
    ProxyLogging._callback_capabilities_cache.clear()
    yield
    ProxyLogging._callback_capabilities_cache.clear()


def _make_guardrail(name="g", should_run=True, override=None):
    cb = MagicMock(spec=CustomGuardrail)
    cb.__class__ = CustomGuardrail
    cb.guardrail_name = name
    cb.event_hook = GuardrailEventHooks.post_call
    cb.should_run_guardrail = MagicMock(return_value=should_run)
    cb.async_post_call_success_hook = AsyncMock(return_value=override)
    cb.run_in_parallel = False
    cb.post_call_deployment_hook_ran = CustomGuardrail(guardrail_name=name).post_call_deployment_hook_ran
    return cb


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
@pytest.mark.parametrize("scenario", ["model", "model-and-request", "cache-hit", "request-only", "discarded", "no-id"])
async def test_post_call_moderates_returned_response_once(
    proxy_logging: ProxyLogging, monkeypatch: pytest.MonkeyPatch, parallel: bool, metadata_key: str, scenario: str
) -> None:
    class Guardrail(CustomGuardrail):
        def __init__(self) -> None:
            super().__init__(guardrail_name="post-mod", event_hook=GuardrailEventHooks.post_call, default_on=False)
            self.evaluations = 0
            self.run_in_parallel = parallel

        async def apply_guardrail(
            self,
            inputs: GenericGuardrailAPIInputs,
            request_data: dict[str, object],
            input_type: str,
            logging_obj: object = None,
        ) -> GenericGuardrailAPIInputs:
            self.evaluations += 1
            return inputs

    guardrail: Final = Guardrail()
    monkeypatch.setattr(litellm, "callbacks", [guardrail])
    metadata: Final[dict[str, object]] = {"guardrails": ["post-mod"]}
    data: Final = {"model": "m", metadata_key: metadata}
    response: Final = ModelResponse(
        id="" if scenario == "no-id" else "chatcmpl-returned",
        choices=[{"message": {"role": "assistant", "content": "Hello"}}],
    )
    deployment_response: Final = (
        response.model_copy(update={"id": "chatcmpl-discarded"}) if scenario == "discarded" else response
    )
    if scenario not in ("cache-hit", "request-only"):
        await guardrail.async_post_call_success_deployment_hook(
            request_data={**data, "guardrails": ["post-mod"]},
            response=deployment_response,
            call_type=CallTypes.acompletion,
        )
    returned: Final = response.model_copy(deep=True)
    out: Final = await proxy_logging.post_call_success_hook(
        data={**data, **({"guardrails": ["post-mod"]} if scenario == "model-and-request" else {})},
        response=returned,
        user_api_key_dict=UserAPIKeyAuth(),
    )
    assert out is returned
    assert guardrail.evaluations == (2 if scenario in ("discarded", "no-id") else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("rewrite", [False, True])
async def test_post_call_rechecks_output_after_later_deployment_callback(
    proxy_logging: ProxyLogging, monkeypatch: pytest.MonkeyPatch, parallel: bool, rewrite: bool
) -> None:
    class Guardrail(CustomGuardrail):
        def __init__(self) -> None:
            super().__init__(guardrail_name="post-mod", event_hook=GuardrailEventHooks.post_call)
            self.evaluations = 0
            self.run_in_parallel = parallel

        async def async_post_call_success_hook(
            self, data: dict[str, object], response: LLMResponseTypes, user_api_key_dict: UserAPIKeyAuth
        ) -> None:
            self.evaluations += 1
            if isinstance(response, ModelResponse) and "restored" in response.model_dump_json():
                raise HTTPException(status_code=400, detail="blocked restored output")

    class Rewriter(CustomLogger):
        async def async_post_call_success_deployment_hook(
            self, request_data: dict[str, object], response: LLMResponseTypes, call_type: CallTypes | None
        ) -> ModelResponse | None:
            if not rewrite or not isinstance(response, ModelResponse):
                return None
            return ModelResponse(
                id=response.id, choices=[{"message": {"role": "assistant", "content": "restored"}}]
            )

    guardrail: Final = Guardrail()
    monkeypatch.setattr(litellm, "callbacks", [guardrail, Rewriter()])
    data: Final = {"model": "m", "guardrails": ["post-mod"], "metadata": {}}
    response: Final = ModelResponse(
        id="chatcmpl-same", choices=[{"message": {"role": "assistant", "content": "placeholder"}}]
    )
    returned: Final = await async_post_call_success_deployment_hook(data, response, CallTypes.acompletion)
    if rewrite:
        with pytest.raises(HTTPException, match="blocked restored output"):
            await proxy_logging.post_call_success_hook(data=data, user_api_key_dict=UserAPIKeyAuth(), response=returned)
    else:
        assert await proxy_logging.post_call_success_hook(
            data=data, user_api_key_dict=UserAPIKeyAuth(), response=returned
        ) is returned
    assert guardrail.evaluations == (2 if rewrite else 1)


@pytest.mark.asyncio
async def test_post_call_success_hook_returns_response_when_no_callbacks(proxy_logging, make_user_api_key_auth, mock_callbacks_disabled):
    response = {"original": True, "model": "m", "choices": []}
    out = await proxy_logging.post_call_success_hook(
        data={}, response=response, user_api_key_dict=make_user_api_key_auth()
    )
    assert out == {"original": True, "model": "m", "choices": []}


@pytest.mark.asyncio
async def test_post_call_success_hook_runs_other_callback_and_replaces_response(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    new_response = {"modified": True, "kept": "yes", "final": "v"}

    class _CL(CustomLogger):
        async def async_post_call_success_hook(self, **kwargs):  # type: ignore[override]
            return new_response

    monkeypatch.setattr(litellm, "callbacks", [_CL()])
    out = await proxy_logging.post_call_success_hook(
        data={}, response={"original": True}, user_api_key_dict=make_user_api_key_auth()
    )
    assert out == new_response


@pytest.mark.asyncio
async def test_post_call_success_hook_guardrail_should_not_run_skipped(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    g = _make_guardrail(should_run=False)
    monkeypatch.setattr(litellm, "callbacks", [g])
    response = MagicMock()
    out = await proxy_logging.post_call_success_hook(
        data={}, response=response, user_api_key_dict=make_user_api_key_auth()
    )
    g.async_post_call_success_hook.assert_not_called()
    assert out is response


@pytest.mark.asyncio
async def test_post_call_success_hook_guardrail_error_raises(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    g = _make_guardrail()
    g.async_post_call_success_hook = AsyncMock(side_effect=RuntimeError("blocked"))
    monkeypatch.setattr(litellm, "callbacks", [g])
    with pytest.raises(RuntimeError):
        await proxy_logging.post_call_success_hook(
            data={}, response=MagicMock(), user_api_key_dict=make_user_api_key_auth()
        )


@pytest.mark.asyncio
async def test_post_call_success_hook_guardrail_returns_modified_response(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    modified = {"a": 1, "b": 2, "c": 3}
    g = _make_guardrail(override=modified)
    monkeypatch.setattr(litellm, "callbacks", [g])
    out = await proxy_logging.post_call_success_hook(
        data={}, response={"orig": True}, user_api_key_dict=make_user_api_key_auth()
    )
    assert out == modified


@pytest.mark.asyncio
@pytest.mark.parametrize("run_in_parallel", [False, True], ids=["sequential", "parallel"])
async def test_post_call_block_names_the_blocking_guardrail_in_applied_guardrails(
    proxy_logging, make_user_api_key_auth, monkeypatch, run_in_parallel
):
    def _passer_that_records(data, user_api_key_dict, response):
        data["metadata"]["applied_guardrails"] = ["passer"]

    passer = _make_guardrail("passer")
    passer.async_post_call_success_hook = AsyncMock(side_effect=_passer_that_records)
    passer.run_in_parallel = run_in_parallel
    blocker = _make_guardrail("blocker")
    blocker.async_post_call_success_hook = AsyncMock(side_effect=HTTPException(status_code=400, detail="blocked"))
    blocker.run_in_parallel = run_in_parallel
    monkeypatch.setattr(litellm, "callbacks", [passer, blocker])
    data = {"model": "m", "metadata": {}}
    with pytest.raises(HTTPException):
        await proxy_logging.post_call_success_hook(
            data=data, response=MagicMock(), user_api_key_dict=make_user_api_key_auth()
        )
    assert data["metadata"]["applied_guardrails"] == ["passer", "blocker"]
