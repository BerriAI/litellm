import json
from collections.abc import Mapping
from typing import Final

import httpx
import pytest
import respx
from openai import OpenAI

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.proxy.auth.model_checks import get_provider_models

MERAGPT_API_BASE: Final = "https://meragpt.com/v1"

MERAGPT_MODELS: Final = (
    "extractor-1",
    "fanout-1",
    "restyler-1",
)


class _RecordingMeraGPT:
    def __init__(self, usage: Mapping[str, object]) -> None:
        self.usage: Final = usage
        self.requests: tuple[httpx.Request, ...] = ()

    def respond(self, request: httpx.Request) -> httpx.Response:
        self.requests = (*self.requests, request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-meragpt",
                "object": "chat.completion",
                "created": 1234567890,
                "model": json.loads(request.content)["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                "usage": dict(self.usage),
            },
        )

    def client(self) -> OpenAI:
        return OpenAI(
            api_key="meragpt-test-key",
            base_url=MERAGPT_API_BASE,
            http_client=httpx.Client(transport=httpx.MockTransport(self.respond)),
        )


def test_prefixed_model_reaches_meragpt_chat_completions_without_the_prefix():
    meragpt: Final = _RecordingMeraGPT({"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4})

    with meragpt.client() as client:
        litellm.completion(
            model="meragpt/restyler-1",
            messages=[{"role": "user", "content": "hello"}],
            client=client,
        )

    assert len(meragpt.requests) == 1
    assert str(meragpt.requests[0].url) == f"{MERAGPT_API_BASE}/chat/completions"
    assert json.loads(meragpt.requests[0].content) == {
        "model": "restyler-1",
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_anthropic_messages_request_is_bridged_to_meragpt_chat_completions():
    meragpt: Final = _RecordingMeraGPT({"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4})

    with meragpt.client() as client:
        response: Final = litellm.anthropic.messages.create(
            model="meragpt/fanout-1",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=10,
            client=client,
        )

    assert [str(request.url) for request in meragpt.requests] == [f"{MERAGPT_API_BASE}/chat/completions"]
    assert response["content"] == [{"type": "text", "text": "hi"}]


def test_extractor_json_schema_and_image_parts_reach_meragpt_unchanged():
    meragpt: Final = _RecordingMeraGPT({"prompt_tokens": 90, "completion_tokens": 12, "total_tokens": 102})
    response_format: Final = {
        "type": "json_schema",
        "json_schema": {
            "name": "receipt",
            "schema": {"type": "object", "properties": {"total": {"type": "number"}, "po": {"type": "string"}}},
        },
    }
    content: Final = [
        {"type": "text", "text": "Extract the receipt."},
        {"type": "image_url", "image_url": {"url": "https://example.com/receipt.jpg"}},
    ]

    with meragpt.client() as client:
        litellm.completion(
            model="meragpt/extractor-1",
            messages=[{"role": "user", "content": content}],
            response_format=response_format,
            client=client,
        )

    sent: Final = json.loads(meragpt.requests[0].content)
    assert str(meragpt.requests[0].url) == f"{MERAGPT_API_BASE}/chat/completions"
    assert sent["model"] == "extractor-1"
    assert sent["response_format"] == response_format
    assert sent["messages"] == [{"role": "user", "content": content}]


def test_extractor_output_tokens_are_free():
    rates: Final = litellm.model_cost["meragpt/extractor-1"]

    assert rates["input_cost_per_token"] == pytest.approx(4e-08)
    assert rates["output_cost_per_token"] == 0


@pytest.mark.parametrize("model", MERAGPT_MODELS)
def test_meragpt_completion_is_charged_at_the_cost_map_rates(model: str):
    meragpt: Final = _RecordingMeraGPT({"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200})

    with meragpt.client() as client:
        response: Final = litellm.completion(
            model=f"meragpt/{model}",
            messages=[{"role": "user", "content": "hello"}],
            client=client,
        )

    rates: Final = litellm.model_cost[f"meragpt/{model}"]
    expected_cost: Final = 1000 * rates["input_cost_per_token"] + 200 * rates["output_cost_per_token"]
    assert response._hidden_params["response_cost"] == pytest.approx(expected_cost)
    assert expected_cost > 0


def test_meragpt_key_and_base_come_from_meragpt_env_vars(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MERAGPT_API_KEY", "meragpt-env-key")
    monkeypatch.delenv("MERAGPT_API_BASE", raising=False)

    assert get_llm_provider(model="meragpt/restyler-1") == (
        "restyler-1",
        "meragpt",
        "meragpt-env-key",
        MERAGPT_API_BASE,
    )

    monkeypatch.setenv("MERAGPT_API_BASE", "https://meragpt.internal.example/v1")

    assert get_llm_provider(model="meragpt/restyler-1")[3] == "https://meragpt.internal.example/v1"


def test_meragpt_env_key_is_sent_as_the_bearer_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MERAGPT_API_KEY", "meragpt-env-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-key")
    monkeypatch.delenv("MERAGPT_API_BASE", raising=False)

    with respx.mock() as upstream:
        route: Final = upstream.post(f"{MERAGPT_API_BASE}/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-meragpt",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "restyler-1",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        )
        litellm.completion(model="meragpt/restyler-1", messages=[{"role": "user", "content": "hello"}])

    assert route.calls.last.request.headers["authorization"] == "Bearer meragpt-env-key"


@pytest.mark.parametrize("model", MERAGPT_MODELS)
def test_missing_meragpt_key_fails_locally_instead_of_sending_the_openai_key(
    monkeypatch: pytest.MonkeyPatch, model: str
):
    monkeypatch.delenv("MERAGPT_API_KEY", raising=False)
    monkeypatch.delenv("MERAGPT_API_BASE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-key")
    monkeypatch.setattr(litellm, "openai_key", "sk-openai-global-key")

    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(f"{MERAGPT_API_BASE}/chat/completions").respond(200, json={})
        with pytest.raises(litellm.AuthenticationError, match="MERAGPT_API_KEY"):
            litellm.completion(model=f"meragpt/{model}", messages=[{"role": "user", "content": "hello"}])

    assert route.calls.call_count == 0


def test_meragpt_wildcard_lists_every_meragpt_catalog_model():
    catalog_models: Final = sorted(
        key for key, row in litellm.model_cost.items() if row.get("litellm_provider") == "meragpt"
    )

    assert catalog_models
    assert sorted(get_provider_models("meragpt") or []) == catalog_models


@pytest.fixture
def openai_key_but_no_meragpt_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MERAGPT_API_KEY", raising=False)
    monkeypatch.delenv("MERAGPT_API_BASE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-key")
    monkeypatch.setattr(litellm, "openai_key", "sk-openai-global-key")


@pytest.mark.usefixtures("openai_key_but_no_meragpt_key")
def test_missing_meragpt_key_fails_speech_locally_instead_of_sending_the_openai_key():
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(f"{MERAGPT_API_BASE}/audio/speech").respond(200, content=b"audio")
        with pytest.raises(litellm.AuthenticationError, match="MERAGPT_API_KEY"):
            litellm.speech(model="meragpt/restyler-1", input="hello", voice="alloy")

    assert route.calls.call_count == 0


def test_meragpt_speech_sends_the_meragpt_key_not_the_openai_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MERAGPT_API_KEY", "meragpt-env-key")
    monkeypatch.delenv("MERAGPT_API_BASE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-key")
    monkeypatch.setattr(litellm, "openai_key", "sk-openai-global-key")

    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(f"{MERAGPT_API_BASE}/audio/speech").respond(200, content=b"audio")
        litellm.speech(model="meragpt/restyler-1", input="hello", voice="alloy")

    assert route.calls.last.request.headers["authorization"] == "Bearer meragpt-env-key"


@pytest.mark.usefixtures("openai_key_but_no_meragpt_key")
def test_missing_meragpt_key_fails_transcription_locally_instead_of_sending_the_openai_key():
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(f"{MERAGPT_API_BASE}/audio/transcriptions").respond(200, json={"text": "hi"})
        with pytest.raises(litellm.AuthenticationError, match="MERAGPT_API_KEY"):
            litellm.transcription(model="meragpt/restyler-1", file=("a.wav", b"RIFF0000WAVE", "audio/wav"))

    assert route.calls.call_count == 0


@pytest.mark.usefixtures("openai_key_but_no_meragpt_key")
async def test_missing_meragpt_key_fails_moderation_locally_instead_of_sending_the_openai_key():
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(f"{MERAGPT_API_BASE}/moderations").respond(200, json={})
        with pytest.raises(litellm.AuthenticationError, match="MERAGPT_API_KEY"):
            await litellm.amoderation(model="meragpt/restyler-1", input="hello")

    assert route.calls.call_count == 0


@pytest.mark.parametrize("base_llm_http_handler", ["true", "false"])
@pytest.mark.usefixtures("openai_key_but_no_meragpt_key")
def test_missing_meragpt_key_fails_locally_with_a_bare_transport_client(
    monkeypatch: pytest.MonkeyPatch, base_llm_http_handler: str
):
    monkeypatch.setenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", base_llm_http_handler)

    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(f"{MERAGPT_API_BASE}/chat/completions").respond(200, json={})
        with pytest.raises(litellm.AuthenticationError, match="MERAGPT_API_KEY"):
            litellm.completion(
                model="meragpt/restyler-1",
                messages=[{"role": "user", "content": "hello"}],
                client=HTTPHandler(),
            )

    assert route.calls.call_count == 0
