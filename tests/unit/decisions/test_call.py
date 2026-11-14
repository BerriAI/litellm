import base64
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Final

import httpx
import pytest
import respx
from pydantic import TypeAdapter

import litellm
from litellm.decisions.main import _prepare_decisions_call
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.prompt_templates.image_handling import in_memory_cache
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.systemone.main import _prepare_systemone_call

QUESTION_TEXT: Final = "Is this about a billing problem?"
CONNECTION: Final = {
    "model": "hosted_vllm/jev-fake",
    "safety_identifier": None,
    "api_key": "fake-provider-key",
    "api_base": "http://127.0.0.1:4010",
    "timeout": None,
    "custom_llm_provider": None,
    "extra_headers": None,
}


def _logging_obj(call_type: str) -> Logging:
    return Logging(
        model="hosted_vllm/jev-fake",
        messages="placeholder",
        stream=False,
        call_type=call_type,
        start_time=datetime.now(timezone.utc),
        litellm_call_id="lit-9659",
        function_id="lit-9659",
    )


def _logged_request(logging_obj: Logging) -> Mapping[str, object]:
    messages: Final = logging_obj.model_call_details["messages"]
    assert len(messages) == 1 and messages[0]["role"] == "user"
    return json.loads(messages[0]["content"])


def test_systemone_logs_every_request_field_the_caller_sent():
    logging_obj: Final = _logging_obj("asystemone")

    _prepare_systemone_call(
        **CONNECTION,
        state={"message": "My card was charged twice."},
        questions={"billing": {"type": "noul", "instructions": QUESTION_TEXT}},
        images=None,
        assets=None,
        audio=None,
        videos=None,
        kwargs={"litellm_logging_obj": logging_obj},
    )

    logged: Final = _logged_request(logging_obj)
    assert logged["state"] == {"message": "My card was charged twice."}
    assert logged["questions"]["billing"]["instructions"] == QUESTION_TEXT
    assert set(logged) == {"state", "questions"}


def test_decisions_logs_every_request_field_the_caller_sent():
    logging_obj: Final = _logging_obj("adecisions")

    _prepare_decisions_call(
        **CONNECTION,
        decision_input="My card was charged twice.",
        questions=[{"type": "predicate", "instructions": QUESTION_TEXT}],
        audio=None,
        videos=None,
        kwargs={"litellm_logging_obj": logging_obj},
    )

    logged: Final = _logged_request(logging_obj)
    assert logged["input"] == "My card was charged twice."
    assert logged["questions"][0]["instructions"] == QUESTION_TEXT
    assert set(logged) == {"input", "questions"}


def test_request_is_logged_even_when_it_is_rejected_before_the_provider_call():
    logging_obj: Final = _logging_obj("asystemone")

    with pytest.raises(litellm.BadRequestError):
        _prepare_systemone_call(
            **CONNECTION,
            state={"message": "My card was charged twice."},
            questions={"billing": {"type": "noul", "instructions": QUESTION_TEXT}},
            images=None,
            assets=None,
            audio=None,
            videos=None,
            kwargs={"litellm_logging_obj": logging_obj, "input": "sent to the wrong function"},
        )

    logged: Final = _logged_request(logging_obj)
    assert logged["questions"]["billing"]["instructions"] == QUESTION_TEXT


def test_decisions_request_is_logged_even_when_its_questions_fail_validation():
    logging_obj: Final = _logging_obj("adecisions")

    with pytest.raises(litellm.BadRequestError):
        _prepare_decisions_call(
            **CONNECTION,
            decision_input="My card was charged twice.",
            questions=[{"type": "not-a-question-type", "instructions": QUESTION_TEXT}],
            audio=None,
            videos=None,
            kwargs={"litellm_logging_obj": logging_obj},
        )

    logged: Final = _logged_request(logging_obj)
    assert logged["questions"][0]["instructions"] == QUESTION_TEXT


def test_non_utf8_bytes_in_the_request_are_logged_and_rejected_with_a_400():
    logging_obj: Final = _logging_obj("adecisions")
    raw_input: Final = b"\xff\xfe not utf-8"

    with pytest.raises(litellm.BadRequestError):
        _prepare_decisions_call(
            **CONNECTION,
            decision_input=raw_input,
            questions=[{"type": "predicate", "instructions": QUESTION_TEXT}],
            audio=None,
            videos=None,
            kwargs={"litellm_logging_obj": logging_obj},
        )

    logged: Final = _logged_request(logging_obj)
    assert base64.urlsafe_b64decode(logged["input"]) == raw_input
    assert logged["questions"][0]["instructions"] == QUESTION_TEXT


@pytest.mark.usefixtures("gemini_httpx_transport")
@pytest.mark.asyncio
@pytest.mark.parametrize("sync", (False, True))
async def test_gemini_rejects_excess_images_before_fetching(sync: bool, respx_mock: respx.MockRouter) -> None:
    limit: Final = 16
    images: Final = [f"https://images.example/{index}.png" for index in range(limit + 1)]
    image_route: Final = respx_mock.get(url__startswith="https://images.example/").mock(
        return_value=httpx.Response(200, content=b"image")
    )
    with pytest.raises(litellm.BadRequestError, match="at most"):
        await _call_gemini(sync, images)
    assert image_route.call_count == 0


@pytest.mark.usefixtures("gemini_httpx_transport")
@pytest.mark.asyncio
@pytest.mark.parametrize("sync", (False, True))
async def test_gemini_stops_image_downloads_at_the_request_budget(
    sync: bool, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", False)
    monkeypatch.setattr(litellm, "module_level_aclient", AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport()))
    budget: Final = 20 * 1024 * 1024
    image: Final = b"x" * (budget // 2 + 1)
    first: Final = respx_mock.get("https://images.example/first.png").mock(
        return_value=httpx.Response(200, content=image, headers={"content-type": "image/png"})
    )
    second: Final = respx_mock.get("https://images.example/second.png").mock(
        return_value=httpx.Response(200, content=image, headers={"content-type": "image/png"})
    )
    last: Final = respx_mock.get("https://images.example/last.png").mock(
        return_value=httpx.Response(200, content=b"unused")
    )
    provider: Final = respx_mock.post("https://generativelanguage.googleapis.com/v1beta/decisions").mock(
        return_value=httpx.Response(200, json={"answers": {"ok": {"type": "noul", "noul": 0.5}}})
    )
    urls: Final = [
        "https://images.example/first.png",
        "https://images.example/second.png",
        "https://images.example/last.png",
    ]
    with pytest.raises(litellm.ImageFetchError, match="remaining request budget"):
        await _call_gemini(sync, urls)
    assert first.call_count == second.call_count == 1
    assert last.call_count == provider.call_count == 0


@pytest.mark.usefixtures("gemini_httpx_transport")
@pytest.mark.asyncio
@pytest.mark.parametrize("sync", (False, True))
async def test_gemini_accepts_remote_images_within_the_request_budget(
    sync: bool, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", False)
    monkeypatch.setattr(litellm, "module_level_aclient", AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport()))
    image: Final = b"\x89PNG\r\n\x1a\n"
    image_route: Final = respx_mock.get("https://images.example/small.png").mock(
        return_value=httpx.Response(200, content=image, headers={"content-type": "image/png"})
    )
    provider: Final = respx_mock.post("https://generativelanguage.googleapis.com/v1beta/decisions").mock(
        return_value=httpx.Response(200, json={"answers": {"ok": {"type": "noul", "noul": 0.5}}})
    )
    await _call_gemini(sync, ("https://images.example/small.png",))
    await _call_gemini(sync, ("https://images.example/small.png",))
    sent: Final = TypeAdapter(dict[str, object]).validate_json(provider.calls.last.request.content)
    assert sent["input"] == [
        {"type": "text", "text": "Inspect"},
        {"type": "image", "data": base64.b64encode(image).decode(), "mime_type": "image/png"},
    ]
    assert image_route.call_count == 1
    inline_image: Final = "data:image/png;base64," + base64.b64encode(b"x" * (20 * 1024 * 1024 - 4)).decode()
    with pytest.raises(litellm.ImageFetchError, match="remaining request budget"):
        await _call_gemini(sync, (inline_image, "https://images.example/small.png"))
    assert image_route.call_count == 1


@pytest.fixture
def gemini_httpx_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    in_memory_cache.flush_cache()
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.mark.usefixtures("gemini_httpx_transport")
@pytest.mark.asyncio
@pytest.mark.parametrize("sync", (False, True))
async def test_gemini_remote_images_can_be_logged_alongside_audio_and_video(
    sync: bool, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", False)
    monkeypatch.setattr(litellm, "module_level_aclient", AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport()))
    image: Final = b"\x89PNG\r\n\x1a\n"
    respx_mock.get("https://images.example/multimodal.png").respond(
        content=image, headers={"content-type": "image/png"}
    )
    provider: Final = respx_mock.post("https://generativelanguage.googleapis.com/v1beta/decisions").respond(
        json={"answers": {"ok": {"type": "noul", "noul": 0.5}}}
    )
    options: Final = {
        "model": "gemini/gemini-decision-1",
        "state": "Inspect",
        "questions": {"ok": {"type": "noul", "instructions": "Is it damaged?"}},
        "images": ["https://images.example/multimodal.png"],
        "audio": ["data:audio/wav;base64,YQ=="],
        "videos": ["data:video/mp4;base64,Yg=="],
        "api_key": "test-key",
    }

    response: Final = litellm.systemone(**options) if sync else await litellm.asystemone(**options)

    assert response.answers["ok"].noul == 0.5
    sent: Final = TypeAdapter(dict[str, object]).validate_json(provider.calls.last.request.content)
    assert sent["input"] == [
        {"type": "text", "text": "Inspect"},
        {"type": "image", "data": base64.b64encode(image).decode(), "mime_type": "image/png"},
        {"type": "audio", "data": "YQ==", "mime_type": "audio/wav"},
        {"type": "video", "data": "Yg==", "mime_type": "video/mp4"},
    ]


@pytest.mark.usefixtures("gemini_httpx_transport")
@pytest.mark.asyncio
@pytest.mark.parametrize("sync", (False, True))
@pytest.mark.parametrize("base_parameter", ("api_base", "base_url"))
async def test_gemini_custom_base_cannot_receive_environment_key(
    sync: bool, base_parameter: str, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "shared-google-key")
    destination: Final = respx_mock.post("https://gateway.example/v1beta/decisions")
    options: Final = {base_parameter: "https://gateway.example"}
    if sync:
        with pytest.raises(litellm.BadRequestError, match="explicit api_key"):
            litellm.systemone(
                model="gemini/gemini-decision-1",
                state="Inspect",
                questions={"ok": {"type": "noul", "instructions": "Is it damaged?"}},
                **options,
            )
    else:
        with pytest.raises(litellm.BadRequestError, match="explicit api_key"):
            await litellm.adecisions(
                model="gemini/gemini-decision-1",
                input="Inspect",
                questions=[{"name": "ok", "type": "predicate", "instructions": "Is it damaged?"}],
                **options,
            )
    assert destination.call_count == 0


@pytest.mark.usefixtures("gemini_httpx_transport")
@pytest.mark.asyncio
@pytest.mark.parametrize("systemone", (False, True))
@pytest.mark.parametrize("base_parameter", ("api_base", "base_url"))
async def test_router_gemini_custom_base_cannot_receive_deployment_key(
    systemone: bool, base_parameter: str, respx_mock: respx.MockRouter
) -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "gemini-decisions",
                "litellm_params": {"model": "gemini/gemini-decision-1", "api_key": "deployment-key"},
            }
        ],
        num_retries=0,
    )
    destination: Final = respx_mock.post("https://gateway.example/v1beta/decisions").respond(
        json={"answers": {"ok": {"type": "noul", "noul": 0.5}}}
    )
    options: Final = {base_parameter: "https://gateway.example"}
    if systemone:
        with pytest.raises(litellm.BadRequestError, match="explicit api_key"):
            await router.asystemone(
                model="gemini-decisions",
                state="Inspect",
                questions={"ok": {"type": "noul", "instructions": "Is it damaged?"}},
                **options,
            )
    else:
        with pytest.raises(litellm.BadRequestError, match="explicit api_key"):
            await router.adecisions(
                model="gemini-decisions",
                input="Inspect",
                questions=[{"name": "ok", "type": "predicate", "instructions": "Is it damaged?"}],
                **options,
            )
    assert destination.call_count == 0


@pytest.mark.usefixtures("gemini_httpx_transport")
@pytest.mark.asyncio
@pytest.mark.parametrize("systemone", (False, True))
@pytest.mark.parametrize("overrides", (False, True))
@pytest.mark.parametrize("sync", (False, True))
@pytest.mark.parametrize("base_parameter", ("api_base", "base_url"))
async def test_router_gemini_keeps_keys_with_their_configured_or_explicit_destination(
    systemone: bool, overrides: bool, sync: bool, base_parameter: str, respx_mock: respx.MockRouter
) -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "gemini-decisions",
                "litellm_params": {
                    "model": "gemini/gemini-decision-1",
                    "api_key": "deployment-key",
                    "api_base": "https://deployment.example",
                },
            }
        ],
        num_retries=0,
    )
    target: Final = "https://gateway.example" if overrides else "https://deployment.example"
    destination: Final = respx_mock.post(f"{target}/v1beta/decisions").respond(
        json={"answers": {"ok": {"type": "noul", "noul": 0.5}}}
    )
    options: Final = {base_parameter: target, "api_key": "caller-key"} if overrides else {}
    payload: Final = (
        {"state": "Inspect", "questions": {"ok": {"type": "noul", "instructions": "Is it damaged?"}}}
        if systemone
        else {"input": "Inspect", "questions": [{"name": "ok", "type": "predicate", "instructions": "Is it damaged?"}]}
    )
    request: Final = router.systemone if systemone else router.decisions
    async_request: Final = router.asystemone if systemone else router.adecisions
    response: Final = (
        request(model="gemini-decisions", **payload, **options)
        if sync
        else await async_request(model="gemini-decisions", **payload, **options)
    )
    if systemone:
        assert response.answers["ok"].noul == 0.5
    else:
        assert response.answers[0].probability == 0.5
    assert destination.call_count == 1
    assert destination.calls[0].request.headers["x-goog-api-key"] == ("caller-key" if overrides else "deployment-key")


async def _call_gemini(sync: bool, images: Sequence[str]) -> None:
    if sync:
        litellm.systemone(
            model="gemini/gemini-decision-1",
            state="Inspect",
            questions={"ok": {"type": "noul", "instructions": "Is it damaged?"}},
            images=images,
            api_key="test",
        )
        return
    await litellm.asystemone(
        model="gemini/gemini-decision-1",
        state="Inspect",
        questions={"ok": {"type": "noul", "instructions": "Is it damaged?"}},
        images=images,
        api_key="test",
    )
