import asyncio
from typing import Final, Literal, TypedDict

from typing_extensions import ReadOnly

import httpx
import pytest
import respx
from pydantic import TypeAdapter

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.router import Router

_MODERATIONS_URL: Final = "https://api.openai.com/v1/moderations"
_MODEL_GROUP: Final = "internal-moderation-model"
_INPUT: Final = "Hello, how are you?"
_CATEGORIES: Final = ("harassment", "hate", "self-harm", "sexual", "violence")
_MODERATION_RESPONSE: Final = {
    "id": "modr-logging",
    "model": "omni-moderation-latest",
    "results": [
        {
            "flagged": False,
            "categories": {name: False for name in _CATEGORIES},
            "category_scores": {name: 0.001 for name in _CATEGORIES},
        }
    ],
}


class _LoggedModeration(TypedDict):
    call_type: ReadOnly[str]
    status: ReadOnly[str]
    custom_llm_provider: ReadOnly[str | None]
    messages: ReadOnly[object]
    response: ReadOnly[object]
    model_group: ReadOnly[str | None]


class _ModerationRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.payloads: Final[list[_LoggedModeration]] = []
        self.logged: Final = asyncio.Event()

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.payloads.append(TypeAdapter(_LoggedModeration).validate_python(kwargs["standard_logging_object"]))
        self.logged.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", ["default-model", "named-model", "router-group"])
async def test_moderation_call_is_logged_as_an_amoderation_standard_payload(
    caller: Literal["default-model", "named-model", "router-group"],
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-unit-test")
    recorder: Final = _ModerationRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    respx_mock.post(_MODERATIONS_URL).mock(return_value=httpx.Response(200, json=_MODERATION_RESPONSE))
    router: Final = Router(
        model_list=[{"model_name": _MODEL_GROUP, "litellm_params": {"model": "openai/omni-moderation-latest"}}]
    )

    response: Final = (
        await router.amoderation(input=_INPUT, model=_MODEL_GROUP)
        if caller == "router-group"
        else await litellm.amoderation(
            input=_INPUT, model=None if caller == "default-model" else "omni-moderation-latest"
        )
    )
    await asyncio.wait_for(recorder.logged.wait(), timeout=10)

    payload: Final = recorder.payloads[-1]
    assert payload["call_type"] == litellm.utils.CallTypes.amoderation.value
    assert payload["status"] == "success"
    assert payload["custom_llm_provider"] == litellm.LlmProviders.OPENAI.value
    assert TypeAdapter(tuple[dict[str, str], ...]).validate_python(payload["messages"])[0]["content"] == _INPUT
    assert dict(TypeAdapter(dict[str, object]).validate_python(payload["response"])) == response.model_dump()
    if caller == "router-group":
        assert payload["model_group"] == _MODEL_GROUP
    else:
        assert not payload["model_group"]
