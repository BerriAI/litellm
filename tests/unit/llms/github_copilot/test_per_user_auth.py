"""Unit tests for per-user GitHub Copilot OAuth connections (per_user_auth)."""

import asyncio
import datetime
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm
from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    CallerCredentialAuthenticationError,
    CallerCredentialRateLimitError,
    ServiceUnavailableError,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.github_copilot.per_user_auth import (
    _SESSION_CACHE,
    GITHUB_COPILOT_USER_SESSION_KWARG_KEY,
    GithubCopilotUserSession,
    _session_cache_key,
    aattach_github_copilot_user_session,
    aexchange_github_token,
    apoll_device_flow,
    astart_device_flow,
    attach_github_copilot_user_session,
    evict_copilot_user_session,
    exchange_github_token,
    github_copilot_auth_mode,
    github_copilot_user_session_from,
    validated_copilot_api_base,
)
from litellm.types.utils import CredentialItem, ModelResponse

EXCHANGE_URL = "https://api.github.com/copilot_internal/v2/token"
DEVICE_CODE_URL = "https://github.com/login/device/code"
ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"


def _credential(auth_type="per_user_oauth"):
    values = {} if auth_type is None else {"github_copilot_auth_type": auth_type}
    return CredentialItem(credential_name="copilot-cred", credential_values=values, credential_info={})


def _exchange_payload(token="copilot-token", expires_at=None, api_base="https://api.githubcopilot.com"):
    return {
        "token": token,
        "expires_at": expires_at if expires_at is not None else int(time.time()) + 3600,
        "endpoints": {"api": api_base},
    }


@contextmanager
def github_http(respond: Callable[[httpx.Request], httpx.Response]):
    """Swap both module-level HTTP clients for httpx MockTransport versions and
    collect every request, so tests assert on the real HTTP edge (URL, headers)
    while no socket is ever opened."""
    requests: list[httpx.Request] = []

    def capture(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return respond(request)

    async_client = AsyncHTTPHandler(transport=httpx.MockTransport(capture))
    sync_client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(capture)))
    with (
        patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client", return_value=async_client),
        patch("litellm.llms.custom_httpx.llm_http_handler.get_async_httpx_client", return_value=async_client),
        patch("litellm.llms.custom_httpx.llm_http_handler.get_httpx_client", return_value=sync_client),
        patch.object(litellm, "module_level_client", sync_client),
    ):
        yield requests


def _exchange_responder(payload=None, status=200):
    def respond(request: httpx.Request) -> httpx.Response:
        if str(request.url) == EXCHANGE_URL:
            return httpx.Response(status, json=payload or _exchange_payload(), request=request)
        pytest.fail(f"unexpected request to {request.url}")

    return respond


@pytest.fixture(autouse=True)
def _clear_session_cache():
    _SESSION_CACHE.flush_cache()
    yield
    _SESSION_CACHE.flush_cache()


@pytest.fixture
def per_user_credential():
    with patch.object(litellm, "credential_list", [_credential()]):
        yield


def test_named_per_user_credential_selects_per_user_mode(per_user_credential):
    assert github_copilot_auth_mode("copilot-cred", None) is True


def test_named_shared_credential_stays_shared_even_if_kwargs_ask_for_per_user():
    with patch.object(litellm, "credential_list", [_credential(auth_type="shared")]):
        assert github_copilot_auth_mode("copilot-cred", "per_user_oauth") is False


def test_named_credential_without_flag_stays_shared():
    with patch.object(litellm, "credential_list", [_credential(auth_type=None)]):
        assert github_copilot_auth_mode("copilot-cred", None) is False


def test_per_user_kwargs_without_credential_name_raise_400():
    with pytest.raises(BadRequestError):
        github_copilot_auth_mode(None, "per_user_oauth")


def test_unknown_named_credential_falls_back_to_kwargs():
    with patch.object(litellm, "credential_list", []):
        assert github_copilot_auth_mode("missing-cred", "per_user_oauth") is True
        assert github_copilot_auth_mode("missing-cred", None) is False


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("https://api.githubcopilot.com", "https://api.githubcopilot.com"),
        ("https://api.githubcopilot.com/", "https://api.githubcopilot.com"),
        ("https://tenant.githubcopilot.com:443", "https://tenant.githubcopilot.com:443"),
        ("githubcopilot.com", "https://api.githubcopilot.com"),  # no scheme -> default
        ("http://api.githubcopilot.com", "https://api.githubcopilot.com"),  # http rejected
        ("https://evil.com", "https://api.githubcopilot.com"),
        ("https://githubcopilot.com.evil.com", "https://api.githubcopilot.com"),
        ("https://user:pw@api.githubcopilot.com", "https://api.githubcopilot.com"),
        ("https://api.githubcopilot.com:8080", "https://api.githubcopilot.com"),
        (None, "https://api.githubcopilot.com"),
        (1234, "https://api.githubcopilot.com"),
        ("https://API.GITHUBCOPILOT.COM", "https://API.GITHUBCOPILOT.COM"),
    ],
)
def test_validated_copilot_api_base(raw, expected):
    assert validated_copilot_api_base(raw) == expected


def test_session_repr_and_str_hide_the_token(caplog):
    session = GithubCopilotUserSession(token="secret-copilot-token", api_base="https://api.githubcopilot.com")
    assert "secret-copilot-token" not in repr(session)
    assert "secret-copilot-token" not in str(session)
    with caplog.at_level("DEBUG"):
        import logging

        logging.getLogger("litellm").debug("session: %s", session)
    assert "secret-copilot-token" not in caplog.text


def test_exchange_returns_session_and_caches_per_user():
    with github_http(_exchange_responder()) as requests:
        s1 = exchange_github_token("user-a", "gh-token", "copilot-cred")
        assert len(requests) == 1
        assert requests[0].headers["Authorization"] == "token gh-token"
        s2 = exchange_github_token("user-a", "gh-token", "copilot-cred")
        assert len(requests) == 1
        assert s2.token == s1.token == "copilot-token"
        # different user -> separate exchange
        exchange_github_token("user-b", "gh-token", "copilot-cred")
        assert len(requests) == 2
        # different github token -> separate exchange
        exchange_github_token("user-a", "gh-token-2", "copilot-cred")
        assert len(requests) == 3


@pytest.mark.parametrize("status", [401, 403, 404])
def test_exchange_auth_failures_raise_caller_auth_error_and_evict(status):
    with github_http(_exchange_responder(status=status)):
        with pytest.raises(CallerCredentialAuthenticationError) as exc:
            exchange_github_token("user-a", "gh-token", "copilot-cred")
        assert "Reconnect GitHub Copilot for credential 'copilot-cred' in the LiteLLM UI (LLM Credentials)" in str(
            exc.value
        )
        assert _SESSION_CACHE.get_cache(_session_cache_key("user-a", "gh-token")) is None


def test_exchange_429_raises_caller_rate_limit():
    with github_http(_exchange_responder(status=429)):
        with pytest.raises(CallerCredentialRateLimitError):
            exchange_github_token("user-a", "gh-token", "copilot-cred")


def test_exchange_other_failure_raises_service_unavailable_without_body():
    with github_http(_exchange_responder(status=500, payload={"secret": "body"})):
        with pytest.raises(ServiceUnavailableError) as exc:
            exchange_github_token("user-a", "gh-token", "copilot-cred")
        assert "body" not in str(exc.value)


def test_exchange_expiry_margin_refetches_near_expiry():
    with github_http(_exchange_responder(_exchange_payload(expires_at=int(time.time()) + 30))) as requests:
        exchange_github_token("user-a", "gh-token", "copilot-cred")
        # ttl under the 60s safety margin -> not cached, second call refetches
        exchange_github_token("user-a", "gh-token", "copilot-cred")
        assert len(requests) == 2


@pytest.mark.asyncio
async def test_aexchange_single_flight_one_http_call():
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_exchange_payload(), request=request)

    with github_http(respond) as requests:
        s1, s2 = await asyncio.gather(
            aexchange_github_token("user-a", "gh-token", "copilot-cred"),
            aexchange_github_token("user-a", "gh-token", "copilot-cred"),
        )
        assert len(requests) == 1
        assert s1.token == s2.token == "copilot-token"


@pytest.mark.asyncio
async def test_aexchange_401_raises_and_evicts():
    with github_http(_exchange_responder(status=401)):
        with pytest.raises(CallerCredentialAuthenticationError):
            await aexchange_github_token("user-a", "gh-token", "copilot-cred")
        assert _SESSION_CACHE.get_cache(_session_cache_key("user-a", "gh-token")) is None


def _kwargs_with_connection(token="gho_user_token"):
    return {
        "litellm_credential_name": "copilot-cred",
        "secret_fields": {
            "user_provider_credentials_user_id": "user-a",
            "user_provider_credentials": {"copilot-cred": token},
        },
    }


@pytest.mark.asyncio
async def test_aattach_injects_session_into_kwargs(per_user_credential):
    with github_http(_exchange_responder()):
        kwargs = _kwargs_with_connection()
        await aattach_github_copilot_user_session(kwargs)
        session = github_copilot_user_session_from(kwargs)
        assert session is not None
        assert session.token == "copilot-token"


def test_attach_sync_injects_session(per_user_credential):
    with github_http(_exchange_responder()):
        kwargs = _kwargs_with_connection()
        attach_github_copilot_user_session(kwargs)
        assert isinstance(kwargs[GITHUB_COPILOT_USER_SESSION_KWARG_KEY], GithubCopilotUserSession)


def test_attach_per_user_without_connection_raises_connect_401(per_user_credential):
    kwargs = {"litellm_credential_name": "copilot-cred", "secret_fields": {}}
    with pytest.raises(CallerCredentialAuthenticationError) as exc:
        attach_github_copilot_user_session(kwargs)
    assert "Connect GitHub Copilot for credential 'copilot-cred' in the LiteLLM UI (LLM Credentials)" in str(exc.value)


def test_attach_shared_mode_does_nothing_without_connection():
    with patch.object(litellm, "credential_list", [_credential(auth_type=None)]):
        kwargs = {"litellm_credential_name": "copilot-cred"}
        attach_github_copilot_user_session(kwargs)
        assert GITHUB_COPILOT_USER_SESSION_KWARG_KEY not in kwargs


def test_evict_drops_cached_session():
    with github_http(_exchange_responder()) as requests:
        exchange_github_token("user-a", "gh-token", "copilot-cred")
        evict_copilot_user_session("user-a", "gh-token")
        exchange_github_token("user-a", "gh-token", "copilot-cred")
        assert len(requests) == 2


@pytest.mark.asyncio
async def test_astart_device_flow_shape():
    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == DEVICE_CODE_URL
        return httpx.Response(
            200,
            json={
                "device_code": "dc",
                "user_code": "UC-123",
                "verification_uri": "https://github.com/login/device",
                "expires_in": 900,
                "interval": 5,
            },
            request=request,
        )

    with github_http(respond):
        start = await astart_device_flow()
        assert (start.device_code, start.user_code, start.verification_uri, start.expires_in, start.interval) == (
            "dc",
            "UC-123",
            "https://github.com/login/device",
            900,
            5,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "github_error,expected_status",
    [
        ("authorization_pending", "pending"),
        ("slow_down", "slow_down"),
        ("expired_token", "expired"),
        ("access_denied", "denied"),
    ],
)
async def test_apoll_device_flow_error_mapping(github_error, expected_status):
    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == ACCESS_TOKEN_URL
        return httpx.Response(200, json={"error": github_error}, request=request)

    with github_http(respond):
        poll = await apoll_device_flow("dc")
        assert poll.status == expected_status


@pytest.mark.asyncio
async def test_apoll_device_flow_connected():
    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == ACCESS_TOKEN_URL
        return httpx.Response(200, json={"access_token": "gho_x"}, request=request)

    with github_http(respond):
        poll = await apoll_device_flow("dc")
        assert poll.status == "connected" and poll.access_token == "gho_x"


def test_is_github_copilot_per_user_request():
    from litellm.llms.github_copilot.per_user_auth import is_github_copilot_per_user_request

    session = GithubCopilotUserSession(token="t", api_base="https://api.githubcopilot.com")
    assert is_github_copilot_per_user_request({GITHUB_COPILOT_USER_SESSION_KWARG_KEY: session}) is True
    assert is_github_copilot_per_user_request({}) is False
    assert is_github_copilot_per_user_request({GITHUB_COPILOT_USER_SESSION_KWARG_KEY: "not-a-session"}) is False
    assert is_github_copilot_per_user_request({"litellm_credential_name": "copilot-cred"}) is False


def _chat_completion_response():
    from litellm.utils import convert_to_model_response_object

    return convert_to_model_response_object(
        response_object={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
        model_response_object=ModelResponse(),
    )


def _secret_fields(user_id: str, github_token: str):
    return {
        "user_provider_credentials_user_id": user_id,
        "user_provider_credentials": {"copilot-cred": github_token},
    }


def test_two_per_user_calls_with_identical_bodies_both_reach_upstream(  # test-quality-ok: upstream dispatch count is the observable signal of the cache exclusion
    per_user_credential, monkeypatch
):
    """Response-cache keys ignore caller identity: without the exclusion the second user's call
    would be served from the first's cache entry and neither run on their own seat."""
    from litellm.caching import Cache

    monkeypatch.setattr(litellm, "cache", Cache())
    with (
        github_http(_exchange_responder()),
        patch(
            "litellm.main.openai_chat_completions.completion",
            return_value=_chat_completion_response(),
        ) as mock_completion,
    ):
        base = {
            "model": "github_copilot/gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "litellm_credential_name": "copilot-cred",
        }
        litellm.completion(**base, secret_fields=_secret_fields("user-a", "gho_a"))
        litellm.completion(**base, secret_fields=_secret_fields("user-b", "gho_b"))
        assert mock_completion.call_count == 2


@pytest.mark.asyncio
async def test_two_per_user_async_calls_with_identical_bodies_both_reach_upstream(  # test-quality-ok: upstream dispatch count is the observable signal of the cache exclusion
    per_user_credential, monkeypatch
):
    """Async path: per-user calls must bypass async_get_cache read and write the same way."""
    from litellm.caching import Cache

    monkeypatch.setattr(litellm, "cache", Cache())
    with (
        github_http(_exchange_responder()),
        patch(
            "litellm.main.openai_chat_completions.acompletion",
            new=AsyncMock(return_value=_chat_completion_response()),
        ) as mock_completion,
    ):
        base = {
            "model": "github_copilot/gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "litellm_credential_name": "copilot-cred",
        }
        await litellm.acompletion(**base, secret_fields=_secret_fields("user-a", "gho_a"))
        await litellm.acompletion(**base, secret_fields=_secret_fields("user-b", "gho_b"))
        assert mock_completion.call_count == 2


def test_shared_mode_second_identical_call_hits_response_cache(  # test-quality-ok: the cache hit is only observable as the upstream not being dispatched
    tmp_path, monkeypatch
):
    """Control: shared device-login github_copilot keeps caching exactly as before, so a second
    identical call does not touch the upstream."""
    import json

    from litellm.caching import Cache

    # a mounted shared login: api-key.json with a far-future token, no device flow needed
    (tmp_path / "api-key.json").write_text(
        json.dumps(
            {
                "token": "shared-copilot-token",
                "expires_at": 4102444800,
                "endpoints": {"api": "https://api.githubcopilot.com"},
            }
        )
    )
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(tmp_path))

    monkeypatch.setattr(litellm, "cache", Cache())
    with patch(
        "litellm.main.openai_chat_completions.completion",
        return_value=_chat_completion_response(),
    ) as mock_completion:
        call = {
            "model": "github_copilot/gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        }
        litellm.completion(**call)
        litellm.completion(**call)
        assert mock_completion.call_count == 1, "shared mode must still serve the second call from cache"


def _embedding_response_payload():
    return {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }


def _responses_payload():
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-5.3-codex",
        "output": [
            {
                "type": "message",
                "id": "m1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
    }


def _anthropic_messages_payload():
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hi"}],
        "model": "claude-haiku-4.5",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 4, "output_tokens": 2},
    }


def _copilot_surface_responder(model_payload):
    def respond(request: httpx.Request) -> httpx.Response:
        if str(request.url) == EXCHANGE_URL:
            return httpx.Response(200, json=_exchange_payload(), request=request)
        if request.url.host.endswith("githubcopilot.com"):
            return httpx.Response(200, json=model_payload, request=request)
        pytest.fail(f"unexpected request to {request.url}")

    return respond


def _no_authenticator():
    from litellm.llms.github_copilot.authenticator import Authenticator

    return patch.object(  # test-quality-ok: patching the shared-login edge asserts it is never consulted
        Authenticator, "get_api_key", side_effect=AssertionError("shared Authenticator must not run")
    )


def _assert_exchange_then_model_call(requests, github_token, model_host="https://api.githubcopilot.com"):
    exchange = [r for r in requests if str(r.url) == EXCHANGE_URL]
    upstream = [r for r in requests if r.url.host.endswith("githubcopilot.com")]
    assert exchange, "expected the GitHub token exchange request"
    assert upstream, "expected the model request to the Copilot host"
    assert exchange[0].headers["Authorization"] == f"token {github_token}"
    assert upstream[0].headers["Authorization"].startswith("Bearer ")
    assert upstream[0].url.host == httpx.URL(model_host).host


def test_e2e_completion_per_user_credential(per_user_credential):
    with (
        github_http(_exchange_responder()) as requests,
        _no_authenticator(),
        patch(
            "litellm.main.openai_chat_completions.completion",
            return_value=_chat_completion_response(),
        ) as mock_completion,
    ):
        response = litellm.completion(
            model="github_copilot/gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            litellm_credential_name="copilot-cred",
            secret_fields=_secret_fields("user-a", "gho_a"),
        )
    assert len(requests) == 1 and str(requests[0].url) == EXCHANGE_URL
    assert requests[0].headers["Authorization"] == "token gho_a"
    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["api_key"] == "copilot-token"
    assert call_kwargs["api_base"] == "https://api.githubcopilot.com"
    extra_headers = call_kwargs["optional_params"]["extra_headers"]
    assert extra_headers["Authorization"] == "Bearer copilot-token"
    assert response.usage.total_tokens > 0


def test_e2e_completion_per_user_session_token_wins_over_caller_authorization(per_user_credential):
    """The session token reaches the wire even when the caller passes their own
    Authorization in extra_headers (chat dispatches through _complete_custom_openai)."""
    with (
        github_http(_exchange_responder()),
        _no_authenticator(),
        patch(
            "litellm.main.openai_chat_completions.completion",
            return_value=_chat_completion_response(),
        ) as mock_completion,
    ):
        litellm.completion(
            model="github_copilot/gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            litellm_credential_name="copilot-cred",
            secret_fields=_secret_fields("user-a", "gho_a"),
            extra_headers={"Authorization": "Bearer caller-token", "x-custom": "keep"},
        )
    extra_headers = mock_completion.call_args.kwargs["optional_params"]["extra_headers"]
    assert extra_headers["Authorization"] == "Bearer copilot-token"
    assert extra_headers["x-custom"] == "keep"


_EVIL_AUTH_HEADERS = {"authorization": "Bearer caller-evil", "x-keep": "1"}


def _assert_session_token_on_the_wire(requests):
    upstream = [r for r in requests if r.url.host.endswith("githubcopilot.com")]
    assert upstream, "expected the model request to the Copilot host"
    assert upstream[0].headers["authorization"] == "Bearer copilot-token"


def test_e2e_completion_per_user_caller_authorization_cannot_override_session(per_user_credential):
    """Caller-supplied Authorization must be stripped at the wrapper: chat's SDK
    client owns its own httpx transport, so the capture point is the headers
    handed to openai_chat_completions.completion."""
    caller_headers: dict[str, str] = dict(_EVIL_AUTH_HEADERS)
    with (
        github_http(_exchange_responder()),
        _no_authenticator(),
        patch(
            "litellm.main.openai_chat_completions.completion",
            return_value=_chat_completion_response(),
        ) as mock_completion,
    ):
        litellm.completion(
            model="github_copilot/gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            litellm_credential_name="copilot-cred",
            secret_fields=_secret_fields("user-a", "gho_a"),
            extra_headers=caller_headers,
        )
    extra_headers = mock_completion.call_args.kwargs["optional_params"]["extra_headers"]
    assert extra_headers["Authorization"] == "Bearer copilot-token"
    assert "authorization" not in extra_headers
    assert extra_headers["x-keep"] == "1"
    assert caller_headers == dict(_EVIL_AUTH_HEADERS), "the caller's dict must not be mutated"


def test_e2e_embedding_per_user_caller_authorization_cannot_override_session(per_user_credential):
    with (
        github_http(_copilot_surface_responder(_embedding_response_payload())) as requests,
        _no_authenticator(),
    ):
        litellm.embedding(
            model="github_copilot/text-embedding-3-small",
            input=["hi"],
            litellm_credential_name="copilot-cred",
            secret_fields=_secret_fields("user-a", "gho_a"),
            extra_headers=dict(_EVIL_AUTH_HEADERS),
        )
    _assert_session_token_on_the_wire(requests)


def test_e2e_responses_per_user_caller_authorization_cannot_override_session(per_user_credential):
    with (
        github_http(_copilot_surface_responder(_responses_payload())) as requests,
        _no_authenticator(),
    ):
        litellm.responses(
            model="github_copilot/gpt-5.3-codex",
            input=[{"role": "user", "content": "hi"}],
            litellm_credential_name="copilot-cred",
            secret_fields=_secret_fields("user-a", "gho_a"),
            extra_headers=dict(_EVIL_AUTH_HEADERS),
        )
    _assert_session_token_on_the_wire(requests)


@pytest.mark.asyncio
async def test_e2e_anthropic_messages_per_user_caller_authorization_cannot_override_session(
    per_user_credential,
):
    with (
        github_http(_copilot_surface_responder(_anthropic_messages_payload())) as requests,
        _no_authenticator(),
    ):
        await litellm.anthropic.messages.acreate(
            model="github_copilot/claude-haiku-4.5",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            litellm_credential_name="copilot-cred",
            secret_fields=_secret_fields("user-a", "gho_a"),
            extra_headers=dict(_EVIL_AUTH_HEADERS),
        )
    _assert_session_token_on_the_wire(requests)


def test_e2e_embedding_per_user_credential(per_user_credential):
    with (
        github_http(_copilot_surface_responder(_embedding_response_payload())) as requests,
        _no_authenticator(),
    ):
        response = litellm.embedding(
            model="github_copilot/text-embedding-3-small",
            input=["hi"],
            litellm_credential_name="copilot-cred",
            secret_fields=_secret_fields("user-a", "gho_a"),
        )
    _assert_exchange_then_model_call(requests, "gho_a")
    assert requests[-1].url.path == "/embeddings"
    assert response.usage.prompt_tokens == 4


def test_e2e_responses_per_user_credential(per_user_credential):
    with (
        github_http(_copilot_surface_responder(_responses_payload())) as requests,
        _no_authenticator(),
    ):
        response = litellm.responses(
            model="github_copilot/gpt-5.3-codex",
            input=[{"role": "user", "content": "hi"}],
            litellm_credential_name="copilot-cred",
            secret_fields=_secret_fields("user-a", "gho_a"),
        )
    _assert_exchange_then_model_call(requests, "gho_a")
    assert requests[-1].url.path == "/responses"
    assert response.usage.total_tokens == 6


@pytest.mark.asyncio
async def test_e2e_anthropic_messages_per_user_credential(per_user_credential):
    with (
        github_http(_copilot_surface_responder(_anthropic_messages_payload())) as requests,
        _no_authenticator(),
    ):
        response = await litellm.anthropic.messages.acreate(
            model="github_copilot/claude-haiku-4.5",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            litellm_credential_name="copilot-cred",
            secret_fields=_secret_fields("user-a", "gho_a"),
        )
    _assert_exchange_then_model_call(requests, "gho_a")
    assert requests[-1].url.path == "/v1/messages"
    assert response["usage"]["input_tokens"] == 4


_CONNECT_MESSAGE = "Connect GitHub Copilot for credential 'copilot-cred' in the LiteLLM UI (LLM Credentials)"


def test_e2e_completion_not_connected_raises_connect_401(per_user_credential):
    with pytest.raises(AuthenticationError, match="Connect GitHub Copilot"):
        litellm.completion(
            model="github_copilot/gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            litellm_credential_name="copilot-cred",
            secret_fields={"user_provider_credentials_user_id": "user-a", "user_provider_credentials": {}},
        )


def test_e2e_embedding_not_connected_raises_connect_401(per_user_credential):
    with pytest.raises(AuthenticationError, match="Connect GitHub Copilot"):
        litellm.embedding(
            model="github_copilot/text-embedding-3-small",
            input=["hi"],
            litellm_credential_name="copilot-cred",
            secret_fields={"user_provider_credentials_user_id": "user-a", "user_provider_credentials": {}},
        )


def test_e2e_responses_not_connected_raises_connect_401(per_user_credential):
    with pytest.raises(AuthenticationError, match="Connect GitHub Copilot"):
        litellm.responses(
            model="github_copilot/gpt-5.3-codex",
            input=[{"role": "user", "content": "hi"}],
            litellm_credential_name="copilot-cred",
            secret_fields={"user_provider_credentials_user_id": "user-a", "user_provider_credentials": {}},
        )


@pytest.mark.asyncio
async def test_e2e_anthropic_messages_not_connected_raises_connect_401(per_user_credential):
    with pytest.raises(AuthenticationError, match="Connect GitHub Copilot"):
        await litellm.anthropic.messages.acreate(
            model="github_copilot/claude-haiku-4.5",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            litellm_credential_name="copilot-cred",
            secret_fields={"user_provider_credentials_user_id": "user-a", "user_provider_credentials": {}},
        )


def _shared_login(tmp_path, monkeypatch):
    import json

    (tmp_path / "api-key.json").write_text(
        json.dumps(
            {
                "token": "shared-copilot-token",
                "expires_at": 4102444800,
                "endpoints": {"api": "https://api.githubcopilot.com"},
            }
        )
    )
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(tmp_path))


def test_e2e_completion_shared_mode_still_uses_authenticator(tmp_path, monkeypatch):
    _shared_login(tmp_path, monkeypatch)
    with patch(
        "litellm.main.openai_chat_completions.completion",
        return_value=_chat_completion_response(),
    ) as mock_completion:
        response = litellm.completion(
            model="github_copilot/gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert mock_completion.call_args.kwargs["api_key"] == "shared-copilot-token"
    assert response.usage.total_tokens > 0


def test_e2e_embedding_shared_mode_still_uses_authenticator(tmp_path, monkeypatch):
    _shared_login(tmp_path, monkeypatch)
    with github_http(_copilot_surface_responder(_embedding_response_payload())) as requests:
        response = litellm.embedding(
            model="github_copilot/text-embedding-3-small",
            input=["hi"],
        )
    upstream = [r for r in requests if r.url.host.endswith("githubcopilot.com")]
    assert upstream and upstream[0].headers["Authorization"] == "Bearer shared-copilot-token"
    assert response.usage.prompt_tokens == 4


def test_e2e_responses_shared_mode_still_uses_authenticator(tmp_path, monkeypatch):
    _shared_login(tmp_path, monkeypatch)
    with github_http(_copilot_surface_responder(_responses_payload())) as requests:
        response = litellm.responses(
            model="github_copilot/gpt-5.3-codex",
            input=[{"role": "user", "content": "hi"}],
        )
    upstream = [r for r in requests if r.url.host.endswith("githubcopilot.com")]
    assert upstream and upstream[0].headers["Authorization"] == "Bearer shared-copilot-token"
    assert response.usage.total_tokens == 6


@pytest.mark.asyncio
async def test_e2e_anthropic_messages_shared_mode_still_uses_authenticator(tmp_path, monkeypatch):
    # The shared Authenticator's access-token refresh refuses to run inside an event
    # loop, so on this async surface the mounted-login path can't reach it; patch the
    # shared-login edge to emulate a mounted api-key.json.
    from litellm.llms.github_copilot.authenticator import Authenticator

    with (
        github_http(_copilot_surface_responder(_anthropic_messages_payload())) as requests,
        patch.object(  # test-quality-ok: patching the shared-login edge is the fixture under test
            Authenticator, "get_api_key", return_value="shared-copilot-token"
        ) as mock_api_key,
        patch.object(  # test-quality-ok: same edge; api_key read is mocked so base must be too
            Authenticator, "get_api_base", return_value="https://api.githubcopilot.com"
        ),
    ):
        response = await litellm.anthropic.messages.acreate(
            model="github_copilot/claude-haiku-4.5",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
        )
    mock_api_key.assert_called()
    upstream = [r for r in requests if r.url.host.endswith("githubcopilot.com")]
    assert upstream and upstream[0].headers["Authorization"] == "Bearer shared-copilot-token"
    assert response["usage"]["input_tokens"] == 4


def test_sync_exchange_single_flight_one_http_call():
    import threading

    def slow_respond(request: httpx.Request) -> httpx.Response:
        time.sleep(0.05)
        return httpx.Response(200, json=_exchange_payload(), request=request)

    sessions: list[GithubCopilotUserSession] = []
    with github_http(slow_respond) as requests:
        threads = [
            threading.Thread(
                target=lambda: sessions.append(exchange_github_token("user-a", "gh-token", "copilot-cred"))
            )
            for _ in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert len(requests) == 1
    assert len(sessions) == 6 and all(s.token == "copilot-token" for s in sessions)


def test_two_users_same_github_token_do_not_share_session():
    with github_http(_exchange_responder()):
        s1 = exchange_github_token("user-a", "gh-same", "copilot-cred")
        s2 = exchange_github_token("user-b", "gh-same", "copilot-cred")
        assert _session_cache_key("user-a", "gh-same") != _session_cache_key("user-b", "gh-same")
        assert s1.token == s2.token == "copilot-token"


def test_router_constructs_with_per_user_credential_deployment(per_user_credential):
    from litellm.llms.github_copilot.authenticator import Authenticator

    with patch.object(Authenticator, "get_api_key", side_effect=AssertionError("shared login")) as get_key:
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "copilot",
                    "litellm_params": {
                        "model": "github_copilot/gpt-4.1",
                        "litellm_credential_name": "copilot-cred",
                    },
                }
            ]
        )
    get_key.assert_not_called()
    deployments: Final = router.get_model_list()
    assert deployments is not None
    assert any(d["model_name"] == "copilot" for d in deployments)


def test_provider_info_per_user_without_session_returns_default_base_and_no_key(per_user_credential):
    from litellm.llms.github_copilot.authenticator import Authenticator
    from litellm.types.router import LiteLLM_Params

    params: Final = LiteLLM_Params(
        model="github_copilot/gpt-4.1",
        litellm_credential_name="copilot-cred",
    )
    with patch.object(Authenticator, "get_api_key", side_effect=AssertionError("shared login")) as get_key:
        _, _, dynamic_api_key, api_base = litellm.get_llm_provider(
            model="github_copilot/gpt-4.1", litellm_params=params
        )
    get_key.assert_not_called()
    assert api_base == "https://api.githubcopilot.com"
    assert dynamic_api_key is None


def test_provider_info_shared_github_copilot_still_uses_authenticator():
    from litellm.llms.github_copilot.authenticator import Authenticator
    from litellm.types.router import LiteLLM_Params

    params: Final = LiteLLM_Params(model="github_copilot/gpt-4.1")
    with (
        patch.object(Authenticator, "get_api_key", return_value="shared-token") as get_key,
        patch.object(Authenticator, "get_api_base", return_value="https://api.githubcopilot.com"),
    ):
        _, _, dynamic_api_key, _ = litellm.get_llm_provider(model="github_copilot/gpt-4.1", litellm_params=params)
    get_key.assert_called_once()
    assert dynamic_api_key == "shared-token"


@pytest.mark.asyncio
async def test_aexchange_cancelled_waiter_does_not_cancel_the_shared_exchange(per_user_credential):
    """Two waiters share one in-flight exchange; cancelling one must not cancel the
    underlying future the other is waiting on."""
    gate = asyncio.Event()

    class _GatedClient:
        async def get(self, url, headers=None):
            await gate.wait()
            return httpx.Response(200, json=_exchange_payload(), request=httpx.Request("GET", url))

    with patch(  # test-quality-ok: the gate holds the HTTP edge open so two waiters overlap
        "litellm.llms.custom_httpx.http_handler.get_async_httpx_client", return_value=_GatedClient()
    ):
        first = asyncio.ensure_future(aexchange_github_token("user-a", "gh-shared", "copilot-cred"))
        await asyncio.sleep(0)
        second = asyncio.ensure_future(aexchange_github_token("user-a", "gh-shared", "copilot-cred"))
        await asyncio.sleep(0)
        second.cancel()
        gate.set()
        session = await first
        assert session.token == "copilot-token" and session.api_base == "https://api.githubcopilot.com"
        with pytest.raises(asyncio.CancelledError):
            await second


@pytest.mark.asyncio
async def test_aexchange_failure_with_a_single_caller_emits_no_unretrieved_warning(per_user_credential):
    """The leader removes the failed future before anyone awaits it; the stored
    exception must be consumed so the loop does not log 'never retrieved'."""
    import litellm.llms.github_copilot.per_user_auth as pua

    captured: list[tuple[object, object]] = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: captured.append((_loop, context)))
    try:
        with (
            github_http(_exchange_responder(status=401)),
            pytest.raises(CallerCredentialAuthenticationError),
        ):
            await aexchange_github_token("user-a", "gh-fail", "copilot-cred")
        _IN_FLIGHT_KEYS = list(pua._IN_FLIGHT)
        import gc

        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)
    assert _IN_FLIGHT_KEYS == []
    assert not any(
        "never retrieved" in str(context.get("message", "")) for _, context in captured if isinstance(context, dict)
    )


@pytest.mark.asyncio
async def test_per_user_streaming_call_writes_nothing_to_the_response_cache(per_user_credential, monkeypatch):
    """Per-user streamed completions must never be written to the shared response
    cache: the handler built before the session attach would otherwise use a stale
    request_kwargs copy without the session marker."""
    from litellm.caching import Cache
    from litellm.caching.caching_handler import LLMCachingHandler

    monkeypatch.setattr(litellm, "cache", Cache())
    handler = LLMCachingHandler(
        original_function=litellm.acompletion,
        request_kwargs={"model": "github_copilot/gpt-4o"},
        start_time=datetime.datetime.now(),
    )
    kwargs = {
        "model": "github_copilot/gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        GITHUB_COPILOT_USER_SESSION_KWARG_KEY: GithubCopilotUserSession(
            token="copilot-token", api_base="https://api.githubcopilot.com"
        ),
    }
    await handler.async_get_cache(
        model="github_copilot/gpt-4o",
        original_function=litellm.acompletion,
        logging_obj=MagicMock(),
        start_time=datetime.datetime.now(),
        call_type="acompletion",
        kwargs=kwargs,
    )
    assert (
        handler.should_store_result_in_cache(original_function=litellm.acompletion, kwargs=handler.request_kwargs)
        is False
    )


@pytest.mark.asyncio
async def test_shared_streaming_call_still_writes_response_cache(tmp_path, monkeypatch):
    """Control: shared device-login streaming keeps caching normally."""
    _shared_login(tmp_path, monkeypatch)
    from litellm.caching import Cache
    from litellm.caching.caching_handler import LLMCachingHandler

    monkeypatch.setattr(litellm, "cache", Cache())
    handler = LLMCachingHandler(
        original_function=litellm.acompletion,
        request_kwargs={"model": "github_copilot/gpt-4o"},
        start_time=datetime.datetime.now(),
    )
    kwargs = {
        "model": "github_copilot/gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
    }
    await handler.async_get_cache(
        model="github_copilot/gpt-4o",
        original_function=litellm.acompletion,
        logging_obj=MagicMock(),
        start_time=datetime.datetime.now(),
        call_type="acompletion",
        kwargs=kwargs,
    )
    assert (
        handler.should_store_result_in_cache(original_function=litellm.acompletion, kwargs=handler.request_kwargs)
        is True
    )


def _per_user_router():
    return litellm.Router(
        model_list=[
            {
                "model_name": "copilot",
                "litellm_params": {
                    "model": "github_copilot/gpt-4.1",
                    "litellm_credential_name": "copilot-cred",
                },
            }
        ]
    )


def test_per_user_deployment_rejects_a_caller_credential_override(per_user_credential):
    """Caller kwargs win the litellm_params merge downstream, so the deployment
    layer is the last place that still sees both names: a different name on a
    per-user deployment must be refused there even if discovery missed the hop."""
    router: Final = _per_user_router()
    deployment: Final = router.get_model_list()[0]
    kwargs: Final = {"litellm_credential_name": "shared-cred", "metadata": {}}
    with pytest.raises(litellm.BadRequestError):
        router._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)


def test_per_user_deployment_allows_the_configured_credential_name(per_user_credential):
    router: Final = _per_user_router()
    deployment: Final = router.get_model_list()[0]
    kwargs: Final = {"litellm_credential_name": "copilot-cred", "metadata": {}}
    router._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)
    assert kwargs["litellm_credential_name"] == "copilot-cred"
    assert "model_info" in kwargs


def test_shared_deployment_keeps_caller_credential_override():
    from litellm.llms.github_copilot.authenticator import Authenticator

    with (
        patch.object(litellm, "credential_list", [_credential(auth_type="shared")]),
        patch.object(Authenticator, "get_api_key", return_value="shared-token"),
    ):
        router: Final = _per_user_router()
        deployment: Final = router.get_model_list()[0]
        kwargs: Final = {"litellm_credential_name": "other-cred", "metadata": {}}
        router._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)
        assert kwargs["litellm_credential_name"] == "other-cred"
        assert "model_info" in kwargs
