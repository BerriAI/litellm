from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import pytest
import respx
from respx.models import Call

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.types.decisions import DecisionsResponse, OpenAIDecisionResponse

_STATE: Final = "Customer asked for a refund"
_SYSTEMONE_QUESTIONS: Final[Mapping[str, object]] = MappingProxyType(
    {"is_refund": {"type": "noul", "instructions": "Is this a refund request?"}}
)
_OPENAI_QUESTIONS: Final[tuple[Mapping[str, object], ...]] = (
    {"type": "predicate", "name": "is_refund", "instructions": "Is this a refund request?"},
)
_SYSTEMONE_RESPONSE: Final[Mapping[str, object]] = {
    "model": "jev-1.13",
    "answers": {"is_refund": {"type": "noul", "noul": 0.93}},
    "usage": {"input_tokens": 12, "output_tokens": 1},
}
_TYPESAFE_URL: Final = "https://api.typesafe.ai/v1/systemone"


@pytest.fixture(autouse=True)
def _httpx_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()


class _CallTypeRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.call_types: list[str] = []  # mutable-ok: test recorder collects logger callbacks

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self.call_types.append(kwargs["call_type"])


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", (True, False), ids=("asystemone", "systemone"))
async def test_systemone_sends_the_systemone_wire_format_and_returns_a_systemone_response(
    use_async: bool, respx_mock: respx.MockRouter
) -> None:
    route: Final = respx_mock.post(_TYPESAFE_URL).respond(json=_SYSTEMONE_RESPONSE)
    kwargs: Final[Mapping[str, object]] = MappingProxyType(
        {"model": "typesafe/jev-1.13", "state": _STATE, "questions": _SYSTEMONE_QUESTIONS, "api_key": "caller-key"}
    )

    response: Final = await litellm.asystemone(**kwargs) if use_async else litellm.systemone(**kwargs)

    assert route.called
    call: Final[Call] = respx_mock.calls.last
    assert json.loads(call.request.content) == {
        "model": "jev-1.13",
        "state": _STATE,
        "questions": {"is_refund": {"type": "noul", "instructions": "Is this a refund request?"}},
    }
    assert isinstance(response, DecisionsResponse)
    assert response.answers["is_refund"].model_dump(mode="json") == {"type": "noul", "noul": 0.93}


@pytest.mark.asyncio
async def test_systemone_rejects_openai_input_and_points_at_decisions(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(_TYPESAFE_URL).respond(json=_SYSTEMONE_RESPONSE)

    with pytest.raises(litellm.BadRequestError, match=r"litellm\.decisions") as caught:
        await litellm.asystemone(
            model="typesafe/jev-1.13", input=_STATE, questions=_SYSTEMONE_QUESTIONS, api_key="caller-key"
        )

    assert caught.value.status_code == 400
    assert not route.called


@pytest.mark.asyncio
async def test_decisions_rejects_systemone_state_and_points_at_systemone(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://api.openai.com/v1/decisions").respond(json={})

    with pytest.raises(litellm.BadRequestError, match=r"litellm\.systemone") as caught:
        await litellm.adecisions(
            model="openai/gpt-6-luna", state=_STATE, questions=_OPENAI_QUESTIONS, api_key="caller-key"
        )

    assert caught.value.status_code == 400
    assert not route.called


@pytest.mark.asyncio
async def test_systemone_rejects_openai_question_lists_before_http(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(_TYPESAFE_URL).respond(json=_SYSTEMONE_RESPONSE)

    with pytest.raises(litellm.BadRequestError, match="Invalid Decisions request"):
        await litellm.asystemone(
            model="typesafe/jev-1.13", state=_STATE, questions=_OPENAI_QUESTIONS, api_key="caller-key"
        )

    assert not route.called


@pytest.mark.asyncio
async def test_router_asystemone_logs_the_systemone_call_type(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    respx_mock.post(_TYPESAFE_URL).respond(json=_SYSTEMONE_RESPONSE)
    recorder: Final = _CallTypeRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    router: Final = litellm.Router(
        model_list=[{"model_name": "jev", "litellm_params": {"model": "typesafe/jev-1.13", "api_key": "caller-key"}}]
    )

    response: Final = await router.asystemone(model="jev", state=_STATE, questions=_SYSTEMONE_QUESTIONS)
    await asyncio.sleep(0)
    GLOBAL_LOGGING_WORKER.start()
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)

    assert isinstance(response, DecisionsResponse)
    assert recorder.call_types == ["asystemone"]
    assert not isinstance(response, OpenAIDecisionResponse)
