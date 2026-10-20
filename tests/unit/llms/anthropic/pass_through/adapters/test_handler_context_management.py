from collections.abc import Mapping
from typing import Final
from unittest.mock import patch

import pytest

import litellm
from litellm.llms.anthropic.pass_through.adapters.handler import LiteLLMMessagesToCompletionTransformationHandler
from litellm.llms.anthropic.pass_through.context_management import AnthropicContextManagementError
from litellm.router import Router
from litellm.types.utils import Choices, Message, ModelResponse, Usage

_TOOL_OUTPUT: Final = "line of the file under review " * 40
_INVALID_COMPACT_EDIT: Final = {
    "edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 1000}}]
}


def _response() -> ModelResponse:
    return ModelResponse(
        choices=[Choices(index=0, finish_reason="stop", message=Message(content="done"))],
        usage=Usage(prompt_tokens=10, completion_tokens=1, total_tokens=11),
    )


def _messages() -> list[dict[str, object]]:
    return [
        {"role": "user", "content": "review a.py"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.py"}}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": _TOOL_OUTPUT}]},
    ]


class _Backend:
    def __init__(self) -> None:
        self.calls: list[Mapping[str, object]] = []

    async def __call__(self, **kwargs: object) -> ModelResponse:
        self.calls.append(kwargs)
        return _response()


class _SyncBackend:
    def __init__(self) -> None:
        self.calls: list[Mapping[str, object]] = []

    def __call__(self, **kwargs: object) -> ModelResponse:
        self.calls.append(kwargs)
        return _response()


def _router(deployment_params: Mapping[str, object], router_defaults: Mapping[str, object]) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "gemini-agent",
                "litellm_params": {"model": "gemini/gemini-2.5-flash", "api_key": "test-key", **deployment_params},
            }
        ],
        default_litellm_params=dict(router_defaults),
    )


async def _send(router: Router, request_options: Mapping[str, object], backend: _Backend) -> object:
    with patch("litellm.acompletion", new=backend):
        return await router.aanthropic_messages(
            model="gemini-agent",
            max_tokens=16,
            messages=_messages(),
            tools=[{"name": "read_file", "input_schema": {"type": "object", "properties": {}}}],
            context_management=_INVALID_COMPACT_EDIT,
            **request_options,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deployment_params", "router_defaults", "request_options", "litellm_drop_params"),
    (
        ({"drop_params": True}, {}, {}, False),
        ({}, {"drop_params": True}, {}, False),
        ({}, {}, {"drop_params": True}, False),
        ({}, {}, {"drop_params": "true"}, False),
        ({}, {}, {}, True),
    ),
    ids=(
        "deployment-drop-params",
        "router-default-drop-params",
        "request-drop-params",
        "request-drop-params-string",
        "litellm-drop-params",
    ),
)
async def test_messages_sends_the_unedited_body_when_drop_params_drops_an_invalid_edit(
    deployment_params: Mapping[str, object],
    router_defaults: Mapping[str, object],
    request_options: Mapping[str, object],
    litellm_drop_params: bool,
) -> None:
    backend: Final = _Backend()
    with patch.object(litellm, "drop_params", litellm_drop_params):
        response: Final = await _send(_router(deployment_params, router_defaults), request_options, backend)

    assert isinstance(response, dict)
    assert response["content"] == [{"type": "text", "text": "done"}]
    assert len(backend.calls) == 1
    assert _TOOL_OUTPUT in str(backend.calls[0]["messages"])


@pytest.mark.asyncio
async def test_messages_rejects_an_invalid_edit_without_drop_params() -> None:
    backend: Final = _Backend()
    with patch.object(litellm, "drop_params", False), pytest.raises(AnthropicContextManagementError) as error:
        await _send(_router({}, {}), {}, backend)

    assert error.value.status_code == 400
    assert backend.calls == []


@pytest.mark.parametrize("drop_params", (True, "true"), ids=("drop-params", "drop-params-string"))
def test_sync_messages_sends_the_unedited_body_when_drop_params_drops_an_invalid_edit(drop_params: bool | str) -> None:
    backend: Final = _SyncBackend()
    with patch.object(litellm, "drop_params", False), patch("litellm.completion", new=backend):
        response: Final = LiteLLMMessagesToCompletionTransformationHandler.anthropic_messages_handler(
            max_tokens=16,
            messages=_messages(),
            model="gemini/gemini-2.5-flash",
            api_key="test-key",
            context_management=_INVALID_COMPACT_EDIT,
            drop_params=drop_params,
        )

    assert isinstance(response, dict)
    assert response["content"] == [{"type": "text", "text": "done"}]
    assert len(backend.calls) == 1
    assert _TOOL_OUTPUT in str(backend.calls[0]["messages"])
