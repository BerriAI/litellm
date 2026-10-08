import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Final, cast
from urllib.parse import parse_qs

import httpx
import pytest
from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.microsoft_365_copilot.chat.handler import acompletion, completion
from litellm.llms.microsoft_365_copilot.common_utils import Microsoft365CopilotError
from litellm.types.llms.openai import AllMessageValues
from litellm.types.proxy.litellm_pre_call_utils import SecretFields
from litellm.types.utils import ModelResponse
from litellm.utils import token_counter

_MODEL: Final = "microsoft_365_copilot/chat"
_MESSAGES_ADAPTER: Final = TypeAdapter(list[AllMessageValues])
_DEFAULT_CHAT_RESPONSE: Final = {"messages": [{"text": "prompt echo"}, {"text": "Copilot response"}]}
_DEFAULT_OAUTH_RESPONSE: Final = {"access_token": "graph-access-token", "expires_in": 3600}


class _AsyncCaptureTransport(httpx.AsyncBaseTransport):
    def __init__(self, responder: Callable[[httpx.Request], httpx.Response]) -> None:
        self._responder = responder

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return self._responder(request)


def _secret_fields(assertion: str) -> SecretFields:
    return SecretFields(raw_headers={"Authorization": f"bEaReR {assertion}"})


def _messages() -> list[AllMessageValues]:
    return _MESSAGES_ADAPTER.validate_python([{"role": "user", "content": "hello Copilot"}])


def _sync_client(
    oauth_status: int = 200,
    oauth_payload: object = _DEFAULT_OAUTH_RESPONSE,
    graph_status: int = 200,
    graph_payload: object = _DEFAULT_CHAT_RESPONSE,
    conversation_id: str = "conversation/42",
) -> tuple[HTTPHandler, list[httpx.Request]]:
    requests: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "identity.example.com":
            return httpx.Response(
                status_code=oauth_status,
                json=oauth_payload,
                request=request,
            )
        if request.url.path == "/beta/copilot/conversations":
            return httpx.Response(
                status_code=201,
                json={"id": conversation_id, "state": "active"},
                request=request,
            )
        if request.url.path.endswith("/chat"):
            return httpx.Response(
                status_code=graph_status,
                json=graph_payload,
                request=request,
            )
        return httpx.Response(status_code=404, json={}, request=request)

    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    return client, requests


def _async_client(
    oauth_status: int = 200,
    oauth_payload: object = _DEFAULT_OAUTH_RESPONSE,
    graph_status: int = 200,
    graph_payload: object = _DEFAULT_CHAT_RESPONSE,
    conversation_id: str = "conversation/42",
) -> tuple[AsyncHTTPHandler, list[httpx.Request]]:
    requests: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "identity.example.com":
            return httpx.Response(
                status_code=oauth_status,
                json=oauth_payload,
                request=request,
            )
        if request.url.path == "/beta/copilot/conversations":
            return httpx.Response(
                status_code=201,
                json={"id": conversation_id, "state": "active"},
                request=request,
            )
        if request.url.path.endswith("/chat"):
            return httpx.Response(
                status_code=graph_status,
                json=graph_payload,
                request=request,
            )
        return httpx.Response(status_code=404, json={}, request=request)

    client: Final = AsyncHTTPHandler(transport=_AsyncCaptureTransport(respond))
    return client, requests


def _token_exchange_params(
    exchange_name: str,
    client_id: str,
    client_secret: str,
) -> Mapping[str, object]:
    return {
        "token_exchange_endpoint": f"https://identity.example.com/{exchange_name}/oauth2/v2.0/token",
        "client_id": client_id,
        "client_secret": client_secret,
        "token_exchange_profile": "jwt_bearer_obo",
        "token_exchange_scope": "https://graph.microsoft.com/.default",
    }


def _token_requests(requests: list[httpx.Request]) -> tuple[httpx.Request, ...]:
    return tuple(request for request in requests if request.url.host == "identity.example.com")


def test_sync_jwt_bearer_obo_uses_exact_exchange_and_sends_graph_token() -> None:
    assertion: Final = "header-sync-9306.payload-sync-9306.signature-sync-9306"
    params: Final = _token_exchange_params(
        "sync-handler-9306",
        "client-sync-handler-9306",
        "secret-sync-handler-9306",
    )
    client, requests = _sync_client()

    response: Final = completion(
        model=_MODEL,
        messages=_messages(),
        api_key=None,
        litellm_params=params,
        optional_params={},
        secret_fields=_secret_fields(assertion),
        client=client,
    )

    token_request: Final = _token_requests(requests)[0]
    token_form: Final = {name: values[0] for name, values in parse_qs(token_request.content.decode("utf-8")).items()}
    graph_requests: Final = tuple(request for request in requests if request.url.host == "graph.microsoft.com")
    assert str(token_request.url) == "https://identity.example.com/sync-handler-9306/oauth2/v2.0/token"
    assert token_form == {
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "client_id": "client-sync-handler-9306",
        "client_secret": "secret-sync-handler-9306",
        "assertion": assertion,
        "scope": "https://graph.microsoft.com/.default",
        "requested_token_use": "on_behalf_of",
    }
    assert graph_requests[0].method == "POST"
    assert graph_requests[0].url.path == "/beta/copilot/conversations"
    assert json.loads(graph_requests[0].content) == {}
    assert graph_requests[0].headers["Authorization"] == "Bearer graph-access-token"
    assert graph_requests[1].url.raw_path == b"/beta/copilot/conversations/conversation%2F42/chat"
    assert json.loads(graph_requests[1].content) == {
        "message": {"text": "hello Copilot"},
        "locationHint": {"timeZone": "UTC"},
    }
    assert graph_requests[1].headers["Authorization"] == "Bearer graph-access-token"
    assert assertion not in graph_requests[0].headers["Authorization"]
    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "Copilot response"
    assert response.model == _MODEL


@pytest.mark.asyncio
async def test_async_jwt_bearer_obo_uses_exact_exchange_and_sends_graph_token() -> None:
    assertion: Final = "header-async-9306.payload-async-9306.signature-async-9306"
    params: Final = _token_exchange_params(
        "async-handler-9306",
        "client-async-handler-9306",
        "secret-async-handler-9306",
    )
    client, requests = _async_client()

    async with client.client:
        response: Final = await acompletion(
            model=_MODEL,
            messages=_messages(),
            api_key=None,
            litellm_params=params,
            optional_params={},
            secret_fields=_secret_fields(assertion),
            client=client,
        )

    token_request: Final = _token_requests(requests)[0]
    token_form: Final = {name: values[0] for name, values in parse_qs(token_request.content.decode("utf-8")).items()}
    graph_requests: Final = tuple(request for request in requests if request.url.host == "graph.microsoft.com")
    assert str(token_request.url) == "https://identity.example.com/async-handler-9306/oauth2/v2.0/token"
    assert token_form == {
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "client_id": "client-async-handler-9306",
        "client_secret": "secret-async-handler-9306",
        "assertion": assertion,
        "scope": "https://graph.microsoft.com/.default",
        "requested_token_use": "on_behalf_of",
    }
    assert graph_requests[0].headers["Authorization"] == "Bearer graph-access-token"
    assert graph_requests[1].url.raw_path == b"/beta/copilot/conversations/conversation%2F42/chat"
    assert graph_requests[1].headers["Authorization"] == "Bearer graph-access-token"
    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "Copilot response"


def test_token_exchange_cache_hits_and_keys_by_subject_token_and_client_secret() -> None:
    client, requests = _sync_client()
    exchange_name: Final = "cache-handler-9306"
    client_id: Final = "client-cache-handler-9306"
    client_secret: Final = "secret-cache-handler-original-9306"
    first_assertion: Final = "header-cache-one.payload-cache-one.signature-cache-one"
    second_assertion: Final = "header-cache-two.payload-cache-two.signature-cache-two"

    def call(assertion: str, secret: str) -> None:
        completion(
            model=_MODEL,
            messages=_messages(),
            api_key=None,
            litellm_params=_token_exchange_params(exchange_name, client_id, secret),
            optional_params={},
            secret_fields=_secret_fields(assertion),
            client=client,
        )

    call(first_assertion, client_secret)
    call(first_assertion, client_secret)
    call(second_assertion, client_secret)
    call(second_assertion, "secret-cache-handler-changed-9306")

    assert len(_token_requests(requests)) == 3


def test_direct_delegated_graph_token_skips_token_exchange() -> None:
    client, requests = _sync_client()

    response: Final = completion(
        model=_MODEL,
        messages=_messages(),
        api_key="delegated-graph-token-9306",
        litellm_params={},
        optional_params={},
        client=client,
    )

    graph_requests: Final = tuple(request for request in requests if request.url.host == "graph.microsoft.com")
    assert len(_token_requests(requests)) == 0
    assert len(graph_requests) == 2
    assert all(request.headers["Authorization"] == "Bearer delegated-graph-token-9306" for request in graph_requests)
    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "Copilot response"


def test_token_exchange_credentials_take_precedence_over_direct_api_key() -> None:
    assertion: Final = "header-priority.payload-priority.signature-priority"
    client, requests = _sync_client()
    messages: Final = _messages()
    litellm_params: Final = _token_exchange_params(
        "priority-9306",
        "client-priority-9306",
        "secret-priority-9306",
    )
    optional_params: Final = {"time_zone": "UTC"}
    secret_fields: Final = _secret_fields(assertion)
    original_messages: Final = deepcopy(messages)
    original_litellm_params: Final = dict(litellm_params)
    original_optional_params: Final = dict(optional_params)
    original_headers: Final = dict(cast(Mapping[str, str], secret_fields["raw_headers"]))

    completion(
        model=_MODEL,
        messages=messages,
        api_key="direct-token-must-not-win-9306",
        litellm_params=litellm_params,
        optional_params=optional_params,
        secret_fields=secret_fields,
        client=client,
    )

    graph_requests: Final = tuple(request for request in requests if request.url.host == "graph.microsoft.com")
    assert len(_token_requests(requests)) == 1
    assert len(graph_requests) == 2
    assert all(request.headers["Authorization"] == "Bearer graph-access-token" for request in graph_requests)
    assert messages == original_messages
    assert litellm_params == original_litellm_params
    assert optional_params == original_optional_params
    assert secret_fields["raw_headers"] == original_headers


def test_partial_token_exchange_configuration_fails_without_http_calls() -> None:
    client, requests = _sync_client()

    with pytest.raises(Microsoft365CopilotError) as error:
        completion(
            model=_MODEL,
            messages=_messages(),
            api_key="direct-token-must-not-fallback-9306",
            litellm_params={
                "token_exchange_endpoint": "https://identity.example.com/partial-9306/token",
                "client_id": "client-partial-9306",
            },
            optional_params={},
            client=client,
        )

    assert error.value.status_code == 400
    assert len(requests) == 0


def test_api_base_override_clears_exchange_config_and_fails_closed_without_token_post() -> None:
    from litellm.router_utils.clientside_credential_handler import get_dynamic_litellm_params

    assertion: Final = "header-base-override.payload-base-override.signature-base-override"
    oauth_fields: Final = (
        "token_exchange_audience",
        "token_exchange_endpoint",
        "token_exchange_profile",
        "token_exchange_scope",
    )
    litellm_params: Final = get_dynamic_litellm_params(
        litellm_params={
            "model": _MODEL,
            "api_base": "https://graph.microsoft.com/beta",
            **_token_exchange_params(
                "base-override-9306",
                "client-base-override-9306",
                "secret-base-override-9306",
            ),
            "token_exchange_audience": "https://graph.microsoft.com",
        },
        request_kwargs={"api_base": "https://caller-controlled.example"},
    )
    assert litellm_params["api_base"] == "https://caller-controlled.example"
    assert all(field not in litellm_params for field in oauth_fields)
    assert "client_id" not in litellm_params
    assert "client_secret" not in litellm_params

    client, requests = _sync_client()
    with pytest.raises(Microsoft365CopilotError) as error:
        completion(
            model=_MODEL,
            messages=_messages(),
            api_key=None,
            litellm_params=litellm_params,
            optional_params={},
            secret_fields=_secret_fields(assertion),
            client=client,
        )

    assert error.value.status_code == 401
    assert _token_requests(requests) == ()


def test_unknown_token_exchange_profile_fails_without_http_calls() -> None:
    client, requests = _sync_client()
    litellm_params: Final = {
        **_token_exchange_params("unknown-profile-9306", "client-unknown-profile-9306", "secret-unknown-profile-9306"),
        "token_exchange_profile": "unknown",
    }

    with pytest.raises(Microsoft365CopilotError) as error:
        completion(
            model=_MODEL,
            messages=_messages(),
            api_key=None,
            litellm_params=litellm_params,
            optional_params={},
            secret_fields=_secret_fields("header.unknown.profile"),
            client=client,
        )

    assert error.value.status_code == 400
    assert error.value.message == ("microsoft_365_copilot token_exchange_profile must be 'rfc8693' or 'jwt_bearer_obo'")
    assert len(requests) == 0


def test_missing_auth_configuration_fails_without_http_calls() -> None:
    client, requests = _sync_client()

    with pytest.raises(Microsoft365CopilotError) as error:
        completion(
            model=_MODEL,
            messages=_messages(),
            api_key=None,
            litellm_params={},
            optional_params={},
            client=client,
        )

    assert error.value.status_code == 401
    assert len(requests) == 0


@pytest.mark.parametrize(
    "secret_fields",
    [
        None,
        SecretFields(raw_headers={"Authorization": "Bearer caller-api-key"}),
    ],
)
def test_token_exchange_requires_a_caller_bearer_token(secret_fields: SecretFields | None) -> None:
    client, requests = _sync_client()

    with pytest.raises(Microsoft365CopilotError) as error:
        completion(
            model=_MODEL,
            messages=_messages(),
            api_key="direct-token-must-not-fallback-9306",
            litellm_params=_token_exchange_params(
                "no-assertion-9306",
                "client-no-assertion-9306",
                "secret-no-assertion-9306",
            ),
            optional_params={},
            secret_fields=secret_fields,
            client=client,
        )

    assert error.value.status_code == 401
    assert str(error.value) == (
        "microsoft_365_copilot with OAuth token exchange requires the caller's IdP-issued access token "
        "in the Authorization header"
    )
    assert len(requests) == 0


def test_token_exchange_invalid_grant_maps_to_401_without_leaking_credentials() -> None:
    assertion: Final = "header-invalid-grant.payload-invalid-grant.signature-invalid-grant"
    client_secret: Final = "secret-invalid-grant-9306"
    client, requests = _sync_client(
        oauth_status=400,
        oauth_payload={
            "error": "invalid_grant",
            "error_description": "The assertion is invalid",
        },
    )

    with pytest.raises(Microsoft365CopilotError) as error:
        completion(
            model=_MODEL,
            messages=_messages(),
            api_key=None,
            litellm_params=_token_exchange_params(
                "invalid-grant-9306",
                "client-invalid-grant-9306",
                client_secret,
            ),
            optional_params={},
            secret_fields=_secret_fields(assertion),
            client=client,
        )

    assert error.value.status_code == 401
    assert str(error.value) == "OAuth token exchange failed: invalid_grant: The assertion is invalid"
    assert assertion not in str(error.value)
    assert client_secret not in str(error.value)
    assert len(requests) == 1


def test_graph_403_extracts_last_reply_from_stringified_conversation() -> None:
    license_message: Final = "It looks like you do not have a valid license"
    graph_error: Final = {
        "error": {
            "code": "UnknownError",
            "message": json.dumps({"messages": [{"text": "prompt echo"}, {"text": license_message}]}),
        }
    }
    client, requests = _sync_client(graph_status=403, graph_payload=graph_error)

    with pytest.raises(Microsoft365CopilotError) as error:
        completion(
            model=_MODEL,
            messages=_messages(),
            api_key="delegated-graph-token-403",
            litellm_params={},
            optional_params={},
            client=client,
        )

    assert error.value.status_code == 403
    assert str(error.value) == license_message
    assert len(requests) == 2


def test_graph_error_redacts_caller_token_client_secret_and_access_token() -> None:
    assertion: Final = "header-redaction.payload-redaction.signature-redaction"
    client_secret: Final = "secret-redaction-9306"
    graph_error: Final = {
        "error": {
            "message": f"token=graph-access-token assertion={assertion} secret={client_secret}",
        }
    }
    client, _ = _sync_client(graph_status=403, graph_payload=graph_error)

    with pytest.raises(Microsoft365CopilotError) as error:
        completion(
            model=_MODEL,
            messages=_messages(),
            api_key=None,
            litellm_params=_token_exchange_params(
                "redaction-9306",
                "client-redaction-9306",
                client_secret,
            ),
            optional_params={},
            secret_fields=_secret_fields(assertion),
            client=client,
        )

    assert "graph-access-token" not in str(error.value)
    assert assertion not in str(error.value)
    assert client_secret not in str(error.value)


def test_streaming_wraps_full_reply_as_fake_stream() -> None:
    client, _ = _sync_client()

    response: Final = litellm.completion(
        model=_MODEL,
        messages=_messages(),
        api_key="delegated-stream-token-9306",
        stream=True,
        stream_options={"include_usage": True},
        client=client,
    )

    assert isinstance(response, CustomStreamWrapper)
    chunks: Final = tuple(response)
    content: Final = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
    assert content == "Copilot response"
    usage_chunk: Final = chunks[-1]
    assert usage_chunk.usage.prompt_tokens == token_counter(
        model=_MODEL,
        messages=_messages(),
    )
    assert usage_chunk.usage.completion_tokens == token_counter(
        model=_MODEL,
        text="Copilot response",
        count_response_tokens=True,
    )
    assert usage_chunk.usage.total_tokens == (usage_chunk.usage.prompt_tokens + usage_chunk.usage.completion_tokens)


@pytest.mark.asyncio
async def test_async_streaming_wraps_full_reply_as_fake_stream() -> None:
    client, _ = _async_client()

    async with client.client:
        response: Final = await litellm.acompletion(
            model=_MODEL,
            messages=_messages(),
            api_key="delegated-async-stream-token-9306",
            stream=True,
            stream_options={"include_usage": True},
            client=client,
        )

    assert isinstance(response, CustomStreamWrapper)
    chunks: Final = tuple([chunk async for chunk in response])
    contents: Final = tuple(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
    assert "".join(contents) == "Copilot response"
    usage_chunk: Final = chunks[-1]
    assert usage_chunk.usage.prompt_tokens == token_counter(
        model=_MODEL,
        messages=_messages(),
    )
    assert usage_chunk.usage.completion_tokens == token_counter(
        model=_MODEL,
        text="Copilot response",
        count_response_tokens=True,
    )
    assert usage_chunk.usage.total_tokens == (usage_chunk.usage.prompt_tokens + usage_chunk.usage.completion_tokens)


def test_litellm_completion_forwards_token_exchange_settings() -> None:
    assertion: Final = "header-dispatch-9306.payload-dispatch-9306.signature-dispatch-9306"
    client, requests = _sync_client()
    messages: Final[list[AllMessageValues]] = cast(
        list[AllMessageValues],
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "dispatch"},
                    {"type": "text", "text": "path"},
                ],
            }
        ],
    )
    original_messages: Final = deepcopy(messages)
    headers: Final[dict[str, str]] = {"X-Caller-Header": "unchanged"}
    original_headers: Final = dict(headers)

    response: Final = litellm.completion(
        model=_MODEL,
        messages=messages,
        api_key="direct-dispatch-token-must-not-win-9306",
        api_base="https://caller-controlled.example",
        extra_headers=headers,
        token_exchange_endpoint="https://identity.example.com/dispatch-9306/oauth2/v2.0/token",
        token_exchange_profile="rfc8693",
        token_exchange_scope="scope-dispatch-9306",
        token_exchange_audience="audience-dispatch-9306",
        client_id="client-dispatch-9306",
        client_secret="secret-dispatch-9306",
        secret_fields=_secret_fields(assertion),
        time_zone="America/New_York",
        client=client,
    )

    graph_requests: Final = tuple(request for request in requests if request.url.host == "graph.microsoft.com")
    token_requests: Final = _token_requests(requests)
    assert isinstance(response, ModelResponse)
    assert len(graph_requests) == 2
    assert len(token_requests) == 1
    assert str(token_requests[0].url) == "https://identity.example.com/dispatch-9306/oauth2/v2.0/token"
    assert {name: values[0] for name, values in parse_qs(token_requests[0].content.decode("utf-8")).items()} == {
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "client_id": "client-dispatch-9306",
        "client_secret": "secret-dispatch-9306",
        "subject_token": assertion,
        "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        "scope": "scope-dispatch-9306",
        "audience": "audience-dispatch-9306",
    }
    assert all(request.headers["Authorization"] == "Bearer graph-access-token" for request in graph_requests)
    assert assertion not in graph_requests[0].headers["Authorization"]
    assert json.loads(graph_requests[1].content) == {
        "message": {"text": "dispatch\npath"},
        "locationHint": {"timeZone": "America/New_York"},
    }
    assert messages == original_messages
    assert headers == original_headers


def test_litellm_completion_accepts_max_tokens_without_sending_it_to_graph() -> None:
    client, requests = _sync_client()

    response: Final = litellm.completion(
        model=_MODEL,
        messages=_messages(),
        api_key="delegated-max-token",
        max_tokens=16,
        max_completion_tokens=32,
        client=client,
    )

    graph_chat_requests: Final = tuple(request for request in requests if request.url.path.endswith("/chat"))
    assert isinstance(response, ModelResponse)
    assert json.loads(graph_chat_requests[-1].content) == {
        "message": {"text": "hello Copilot"},
        "locationHint": {"timeZone": "UTC"},
    }


def test_litellm_completion_still_rejects_temperature() -> None:
    client, _ = _sync_client()

    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.completion(
            model=_MODEL,
            messages=_messages(),
            api_key="delegated-temperature",
            temperature=0.2,
            client=client,
        )
