import json
from collections.abc import Iterator
from typing import Final

import anthropic
import openai
import pytest
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MESSAGES: Final[list[JsonValue]] = [{"role": "user", "content": "Count this synthetic request."}]
_PROVIDER_KEY_VARIABLES: Final = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "AZURE_AI_API_KEY",
    "OPENAI_API_KEY",
)


@pytest.fixture(scope="module")
def keyless_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedProxy]:
    with (
        gateway_from_environment() as gateway,
        owned_proxy_process(
            gateway,
            tmp_path_factory.mktemp("count-tokens-error-shapes"),
            {},
            remove_environment=_PROVIDER_KEY_VARIABLES,
            workers=1,
        ) as owned,
    ):
        yield owned


def _anthropic(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key=gateway.key, base_url=str(gateway.client.base_url), max_retries=0, http_client=gateway.client
    )


def _never_called(request: Request) -> Reply:
    raise AssertionError(f"count reached the provider without a key: {request.target}")


@pytest.mark.parametrize(
    ("deployment", "setting"),
    (
        ({"model": "anthropic/claude-sonnet-4-6"}, "Anthropic API key"),
        ({"model": "azure_ai/claude-sonnet-4-6"}, "Azure AI API key"),
    ),
    ids=("anthropic", "azure_ai_anthropic"),
)
def test_anthropic_count_without_a_provider_key_is_an_anthropic_401(
    keyless_proxy: OwnedProxy, deployment: dict[str, JsonValue], setting: str
) -> None:
    gateway: Final = keyless_proxy.gateway
    with wire_server(_never_called) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(**deployment, api_base=wire.url, api_key=None, num_retries=0)
        with pytest.raises(anthropic.AuthenticationError) as raised:
            _anthropic(gateway).messages.count_tokens(model=model, messages=_MESSAGES)
        assert raised.value.body == {
            "type": "error",
            "error": {"type": "authentication_error", "message": raised.value.body["error"]["message"]},
        }
        assert f"No {setting} is set for model {model}" in raised.value.message
        assert wire.drain() == ()


def test_anthropic_count_request_errors_use_the_anthropic_envelope(keyless_proxy: OwnedProxy) -> None:
    gateway: Final = keyless_proxy.gateway
    upstream_error: Final = {"type": "error", "error": {"type": "invalid_request_error", "message": "scripted reject"}}
    with (
        wire_server(lambda _request: Reply(status=400, body=json.dumps(upstream_error).encode())) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-4-6", api_base=wire.url, api_key="synthetic-key", num_retries=0
        )
        missing_messages: Final = gateway.request("POST", "/v1/messages/count_tokens", {"model": model})
        assert missing_messages.status_code == 400, missing_messages.text
        assert _JSON_OBJECT.validate_json(missing_messages.content) == {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "messages parameter is required"},
        }

        rejected: Final = gateway.request(
            "POST", "/v1/messages/count_tokens", {"model": model, "messages": _MESSAGES}
        )
        assert rejected.status_code == 400, rejected.text
        body: Final = _JSON_OBJECT.validate_json(rejected.content)
        assert set(body) <= {"type", "error", "request_id"}, body
        assert body["error"] == upstream_error["error"], body
        assert len(wire.drain()) == 1


def test_openai_input_tokens_without_a_provider_key_is_an_openai_401(keyless_proxy: OwnedProxy) -> None:
    gateway: Final = keyless_proxy.gateway
    with wire_server(_never_called) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4o-mini", api_base=f"{wire.url}/v1", api_key=None, num_retries=0
        )
        client: Final = openai.OpenAI(
            api_key=gateway.key, base_url=str(gateway.client.base_url), max_retries=0, http_client=gateway.client
        )
        with pytest.raises(openai.AuthenticationError) as raised:
            client.responses.input_tokens.count(model=model, input="Count this synthetic request.")
        assert f"No OpenAI API key is set for model {model}" in raised.value.message
        assert wire.drain() == ()


def test_google_count_tokens_errors_use_the_google_envelope(keyless_proxy: OwnedProxy) -> None:
    gateway: Final = keyless_proxy.gateway
    upstream_error: Final = {
        "error": {"code": 400, "message": "API key not valid. Please pass a valid API key.", "status": "INVALID_ARGUMENT"}
    }
    with (
        wire_server(lambda _request: Reply(status=400, body=json.dumps(upstream_error).encode())) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model="gemini/gemini-2.5-flash", api_base=wire.url, api_key="synthetic-key", num_retries=0
        )
        rejected: Final = gateway.request(
            "POST",
            f"/v1beta/models/{model}:countTokens",
            {"contents": [{"role": "user", "parts": [{"text": "Count this synthetic request."}]}]},
        )
        assert rejected.status_code == 400, rejected.text
        error: Final = _JSON_OBJECT.validate_json(rejected.content)["error"]
        assert isinstance(error, dict) and error["code"] == 400 and error["status"] == "INVALID_ARGUMENT", error
        assert "API key not valid" in str(error["message"]), error
        assert len(wire.drain()) == 1

        malformed: Final = gateway.request("POST", f"/v1beta/models/{model}:countTokens", {"contents": "not-a-list"})
        assert malformed.status_code == 400, malformed.text
        assert _JSON_OBJECT.validate_json(malformed.content) == {
            "error": {"code": 400, "message": "Invalid Google countTokens request body", "status": "INVALID_ARGUMENT"}
        }
