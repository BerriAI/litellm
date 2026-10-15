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
from tests.unit.conftest import AsyncOnlyImageFetch

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
_CLOUDFLARE_API_BASE: Final = "https://api.cloudflare.com/client/v4/accounts/acct/ai/run"
_CLOUDFLARE_CLEF_URL: Final = f"{_CLOUDFLARE_API_BASE}/@cf/cloudflare/clef"
_PNG_DATA_URL: Final = "data:image/png;base64,AA=="


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
@pytest.mark.parametrize("use_async", (True, False), ids=("adecisions", "decisions"))
async def test_decisions_with_state_still_sends_the_systemone_format_and_warns(
    use_async: bool, respx_mock: respx.MockRouter
) -> None:
    route: Final = respx_mock.post(_TYPESAFE_URL).respond(json=_SYSTEMONE_RESPONSE)
    kwargs: Final[Mapping[str, object]] = MappingProxyType(
        {"model": "typesafe/jev-1.13", "state": _STATE, "questions": _SYSTEMONE_QUESTIONS, "api_key": "caller-key"}
    )

    with pytest.warns(UserWarning, match=r"litellm\.systemone"):
        response: Final = await litellm.adecisions(**kwargs) if use_async else litellm.decisions(**kwargs)

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
async def test_decisions_with_state_still_sends_its_images(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(_CLOUDFLARE_CLEF_URL).respond(json=_SYSTEMONE_RESPONSE)

    with pytest.warns(UserWarning, match=r"litellm\.systemone"):
        await litellm.adecisions(
            model="cloudflare/clef",
            state=_STATE,
            questions=_SYSTEMONE_QUESTIONS,
            images=[_PNG_DATA_URL],
            api_key="caller-key",
            api_base=_CLOUDFLARE_API_BASE,
        )

    sent: Final = json.loads(respx_mock.calls.last.request.content)
    assert sent["images"] == [_PNG_DATA_URL]
    assert route.called


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


_BOX_RESPONSE: Final[Mapping[str, object]] = {
    "model": "jev-1.14.0",
    "answers": {
        "has_mug": {"type": "noul", "noul": 0.97},
        "mug": {"type": "box", "box": {"top": 0.1, "bottom": 0.5, "left": 0.2, "right": 0.6}, "confidence": 0.84},
    },
    "usage": {"input_tokens": 299, "output_tokens": 37},
}
_MUG_BOX: Final[Mapping[str, float]] = MappingProxyType({"top": 0.1, "bottom": 0.5, "left": 0.2, "right": 0.6})


@pytest.mark.asyncio
async def test_systemone_sends_assets_and_box_questions_to_typesafe(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(_TYPESAFE_URL).respond(json=_BOX_RESPONSE)
    questions: Final[Mapping[str, object]] = {
        "has_mug": {"type": "noul", "instructions": "Is there a mug in `photo-1`?"},
        "mug": {"type": "box", "instructions": "the mug in `photo-1`"},
    }

    response: Final = await litellm.asystemone(
        model="typesafe/jev-1.14.0",
        state={"photo": "photo-1"},
        assets={"photo-1": {"type": "image", "format": "png", "data_base64": "AA=="}},
        questions=questions,
        api_key="caller-key",
    )

    assert json.loads(route.calls.last.request.content) == {
        "model": "jev-1.14.0",
        "state": {"photo": "photo-1"},
        "questions": questions,
        "assets": {"photo-1": {"type": "image", "format": "png", "data_base64": "AA=="}},
    }
    assert isinstance(response, DecisionsResponse)
    assert response.answers["mug"].model_dump(mode="json") == {"type": "box", "box": _MUG_BOX, "confidence": 0.84}


@pytest.mark.asyncio
async def test_decisions_input_images_reach_typesafe_as_assets_referenced_in_the_state(
    respx_mock: respx.MockRouter,
) -> None:
    route: Final = respx_mock.post(_TYPESAFE_URL).respond(json=_BOX_RESPONSE)

    response: Final = await litellm.adecisions(
        model="typesafe/jev-1.14.0",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Desk photo:"},
                    {"type": "input_image", "image_url": _PNG_DATA_URL},
                ],
            }
        ],
        questions=[
            {"type": "predicate", "name": "has_mug", "instructions": "Is there a mug on the desk?"},
            {"type": "box", "name": "mug", "instructions": "the mug on the desk"},
        ],
        api_key="caller-key",
    )

    sent: Final = json.loads(route.calls.last.request.content)
    (reference,) = sent["assets"]
    assert sent["state"] == f"Desk photo:\n\n{reference}"
    assert sent["assets"][reference] == {"type": "image", "format": "png", "data_base64": "AA=="}
    assert sent["questions"]["mug"] == {"type": "box", "instructions": "the mug on the desk"}
    assert isinstance(response, OpenAIDecisionResponse)
    assert [answer.model_dump(mode="json") for answer in response.answers] == [
        {"type": "predicate", "name": "has_mug", "probability": 0.97},
        {"type": "box", "name": "mug", "box": _MUG_BOX, "confidence": 0.84},
    ]


@pytest.mark.asyncio
async def test_decisions_remote_images_reach_typesafe_inlined_as_assets(
    respx_mock: respx.MockRouter, async_only_image_fetch: AsyncOnlyImageFetch
) -> None:
    route: Final = respx_mock.post(_TYPESAFE_URL).respond(json=_BOX_RESPONSE)

    await litellm.adecisions(
        model="typesafe/jev-1.14.0",
        input=[{"role": "user", "content": [{"type": "input_image", "image_url": "https://images.example/desk.png"}]}],
        questions=[{"type": "predicate", "name": "has_mug", "instructions": "Is there a mug on the desk?"}],
        api_key="caller-key",
    )

    sent: Final = json.loads(route.calls.last.request.content)
    assert async_only_image_fetch.fetched == ["https://images.example/desk.png"]
    assert sent["assets"] == {
        sent["state"]: {"type": "image", "format": "png", "data_base64": async_only_image_fetch.base64_png}
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scoped_provider", "expected_headers"),
    (("typesafe", {"typesafe-danger-zone-long-context": "true"}), ("perplexity", {})),
    ids=("typesafe", "another_provider"),
)
async def test_provider_specific_headers_reach_typesafe_only_when_scoped_to_it(
    scoped_provider: str, expected_headers: Mapping[str, str], respx_mock: respx.MockRouter
) -> None:
    route: Final = respx_mock.post(_TYPESAFE_URL).respond(json=_SYSTEMONE_RESPONSE)

    await litellm.asystemone(
        model="typesafe/jev-1.14.0",
        state=_STATE,
        questions=_SYSTEMONE_QUESTIONS,
        api_key="caller-key",
        provider_specific_header={
            "custom_llm_provider": scoped_provider,
            "extra_headers": {"TypeSafe-Danger-Zone-Long-Context": "true"},
        },
    )

    sent_headers: Final = route.calls.last.request.headers
    assert {
        name: value for name, value in sent_headers.items() if name.startswith("typesafe-danger-zone-")
    } == expected_headers
