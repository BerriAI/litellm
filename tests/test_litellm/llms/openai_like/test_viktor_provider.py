import json
from pathlib import Path

import httpx
import pytest
import respx

import litellm
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

CHAT_COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "viktor",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

RESPONSE = {
    "id": "resp-1",
    "object": "response",
    "created_at": 1,
    "status": "completed",
    "model": "viktor",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hello", "annotations": []}],
        }
    ],
    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
}

MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture
def base_url() -> str:
    provider = JSONProviderRegistry.get("viktor")
    assert provider is not None
    return provider.base_url


@pytest.fixture(autouse=True)
def viktor_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIKTOR_API_KEY", "env-key")
    monkeypatch.delenv("VIKTOR_API_BASE", raising=False)


@pytest.fixture
def viktor_http():
    with respx.mock(assert_all_called=False) as router:
        yield router


def test_completion_posts_to_viktor_with_the_env_key(viktor_http: respx.MockRouter, base_url: str) -> None:
    route = viktor_http.post(f"{base_url}/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_COMPLETION)
    )

    response = litellm.completion(model="viktor/viktor", messages=MESSAGES, max_completion_tokens=256)

    assert response.choices[0].message.content == "hello"
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer env-key"
    body = json.loads(request.content)
    assert body["model"] == "viktor"
    assert body["max_completion_tokens"] == 256


def test_responses_posts_to_viktor(viktor_http: respx.MockRouter, base_url: str) -> None:
    route = viktor_http.post(f"{base_url}/responses").mock(return_value=httpx.Response(200, json=RESPONSE))

    response = litellm.responses(model="viktor/viktor", input="hi")

    assert response.output[-1].content[0].text == "hello"
    assert json.loads(route.calls.last.request.content)["model"] == "viktor"


def test_api_base_env_override_is_used(viktor_http: respx.MockRouter, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIKTOR_API_BASE", "https://gateway.example/api/compat/v1")
    route = viktor_http.post("https://gateway.example/api/compat/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_COMPLETION)
    )

    litellm.completion(model="viktor/viktor", messages=MESSAGES)

    assert route.called


def test_explicit_api_base_and_key_win(viktor_http: respx.MockRouter) -> None:
    route = viktor_http.post("https://explicit.example/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_COMPLETION)
    )

    litellm.completion(
        model="viktor/viktor", messages=MESSAGES, api_base="https://explicit.example/v1", api_key="explicit-key"
    )

    assert route.calls.last.request.headers["authorization"] == "Bearer explicit-key"


def test_unprefixed_model_on_the_viktor_base_url_uses_the_viktor_key(
    viktor_http: respx.MockRouter, base_url: str
) -> None:
    route = viktor_http.post(f"{base_url}/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_COMPLETION)
    )

    litellm.completion(model="agent", messages=MESSAGES, api_base=base_url)

    assert route.calls.last.request.headers["authorization"] == "Bearer env-key"


def test_router_deployment_reaches_viktor(viktor_http: respx.MockRouter, base_url: str) -> None:
    route = viktor_http.post(f"{base_url}/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_COMPLETION)
    )
    router = litellm.Router(model_list=[{"model_name": "agent", "litellm_params": {"model": "viktor/viktor"}}])

    response = router.completion(model="agent", messages=MESSAGES)

    assert response.choices[0].message.content == "hello"
    assert route.called


def test_transcription_is_refused_without_calling_viktor(viktor_http: respx.MockRouter) -> None:
    catch_all = viktor_http.route().mock(return_value=httpx.Response(500))

    with pytest.raises(ValueError):
        litellm.transcription(model="viktor/viktor", file=("a.wav", b"RIFF", "audio/wav"))

    assert not catch_all.called


def test_endpoint_matrix_matches_its_backup() -> None:
    def endpoints(path: Path) -> dict:
        return json.loads(path.read_text())["providers"]["viktor"]["endpoints"]

    root = Path(__file__).parents[4] / "provider_endpoints_support.json"
    backup = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"

    assert endpoints(root) == endpoints(backup)
    assert endpoints(root)["responses"] is JSONProviderRegistry.supports_responses_api("viktor")
