import asyncio
import time
from types import TracebackType
from typing import Final
from unittest.mock import MagicMock, patch

import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

import litellm
from litellm.models.credentials import CredentialItem
from litellm.realtime_api import main as realtime_main
from litellm.realtime_api.main import _with_resolved_session_model


class FakeLogging:
    def update_from_kwargs(self, **kwargs):
        pass


def test_resolves_top_level_session_model():
    resolved = _with_resolved_session_model({"model": "alias/gpt-realtime"}, "gpt-realtime")
    assert resolved == {"model": "gpt-realtime"}


def test_session_without_model_is_returned_unchanged():
    session = {"type": "realtime", "audio": {"input": {}}}
    assert _with_resolved_session_model(session, "gpt-realtime") == session


def test_does_not_clobber_flat_transcription_model():
    """The nested transcription model is a different model than the realtime
    conversation model and must not be overwritten with the routing model."""
    resolved = _with_resolved_session_model(
        {"model": "gpt-4o-realtime-preview", "input_audio_transcription": {"model": "whisper-1"}},
        "gpt-4o-realtime-preview",
    )
    assert resolved["input_audio_transcription"]["model"] == "whisper-1"


def test_does_not_clobber_nested_audio_transcription_model():
    resolved = _with_resolved_session_model(
        {
            "model": "gpt-4o-realtime-preview",
            "audio": {"input": {"transcription": {"model": "whisper-1"}}},
        },
        "gpt-4o-realtime-preview",
    )
    assert resolved["audio"]["input"]["transcription"]["model"] == "whisper-1"


def test_original_session_is_not_mutated():
    session = {"model": "alias/gpt-realtime"}
    _with_resolved_session_model(session, "gpt-realtime")
    assert session == {"model": "alias/gpt-realtime"}


def _run_client_secret(session, model, monkeypatch):
    captured = {}

    async def mock_handler(**kwargs):
        captured.update(kwargs)
        return object()

    def mock_get_llm_provider(model, api_base, api_key):
        return model, "openai", None, api_base

    monkeypatch.setattr(realtime_main, "get_llm_provider", mock_get_llm_provider)
    monkeypatch.setattr(
        realtime_main.base_llm_http_handler,
        "async_realtime_client_secret_handler",
        mock_handler,
    )

    asyncio.run(
        realtime_main.acreate_realtime_client_secret.__wrapped__(
            model=model,
            session=session,
            litellm_logging_obj=FakeLogging(),
        )
    )
    return captured


def test_client_secret_session_model_takes_priority_over_top_level(monkeypatch):
    """Backwards-compatible ordering: an explicit session.model wins over the
    top-level model, matching the proxy's own resolution order."""
    captured = _run_client_secret(
        session={"model": "gpt-realtime-session"},
        model="gpt-realtime-top-level",
        monkeypatch=monkeypatch,
    )
    assert captured["model"] == "gpt-realtime-session"
    assert captured["request_data"]["session"]["model"] == "gpt-realtime-session"


async def _hanging_resolver(credentials, project_id, custom_llm_provider) -> tuple[str, str]:
    await asyncio.sleep(30)
    return "", ""


async def _thread_offloaded_hanging_resolver(credentials, project_id, custom_llm_provider) -> tuple[str, str]:
    from litellm.litellm_core_utils.asyncify import asyncify

    await asyncify(time.sleep)(30)
    return "", ""


async def _instant_resolver(credentials, project_id, custom_llm_provider) -> tuple[str, str]:
    return "token-abc", "resolved-project"


@pytest.mark.asyncio
async def test_vertex_credential_resolution_returns_the_resolved_token_and_project():
    assert await realtime_main._resolve_vertex_access_token_bounded(
        credentials="fake-credentials",
        project_id="fake-project",
        resolver=_instant_resolver,
        timeout_seconds=5,
    ) == ("token-abc", "resolved-project")


@pytest.mark.asyncio
async def test_vertex_credential_resolution_times_out_instead_of_hanging():
    """Regression for the realtime accept-then-silence hang: a stalled Google
    OAuth token refresh used to block the vertex branch unbounded (minutes of
    zero frames for the client). It must raise promptly and name the timeout."""
    start = time.monotonic()
    with pytest.raises(ValueError, match="timed out fetching Google OAuth access token"):
        await realtime_main._resolve_vertex_access_token_bounded(
            credentials="fake-credentials",
            project_id="fake-project",
            resolver=_hanging_resolver,
            timeout_seconds=0.05,
        )
    assert time.monotonic() - start < 5


@pytest.mark.asyncio
async def test_vertex_credential_resolution_bounds_a_thread_offloaded_refresh():
    """The real stall is a blocking google-auth refresh that runs in a worker
    thread via asyncify, not a plain awaitable sleep. A timeout that only bounds
    cancellable awaits would leave that shape hanging, so bound the shape the
    proxy actually runs."""
    start = time.monotonic()
    with pytest.raises(ValueError, match="timed out fetching Google OAuth access token"):
        await realtime_main._resolve_vertex_access_token_bounded(
            credentials="fake-credentials",
            project_id="fake-project",
            resolver=_thread_offloaded_hanging_resolver,
            timeout_seconds=0.05,
        )
    assert time.monotonic() - start < 5


@pytest.mark.asyncio
async def test_meta_realtime_dispatches_to_base_handler_with_meta_config(monkeypatch: pytest.MonkeyPatch):
    from litellm.llms.meta.realtime.transformation import MetaRealtimeConfig

    captured: dict[str, object] = {}

    def mock_get_llm_provider(model, api_base, api_key):
        return model.removeprefix("meta/"), "meta", None, api_base

    async def mock_async_realtime(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(realtime_main, "get_llm_provider", mock_get_llm_provider)
    monkeypatch.setattr(realtime_main.base_llm_http_handler, "async_realtime", mock_async_realtime)

    await realtime_main._arealtime.__wrapped__(
        model="meta/muse-voice-transcribe-1.0",
        websocket=MagicMock(),
        litellm_logging_obj=FakeLogging(),
        query_params={"model": "meta/muse-voice-transcribe-1.0", "intent": "transcription"},
    )

    assert isinstance(captured["provider_config"], MetaRealtimeConfig)
    assert captured["model"] == "muse-voice-transcribe-1.0"
    assert captured["query_params"] == {"model": "muse-voice-transcribe-1.0", "intent": "transcription"}


@pytest.mark.asyncio
async def test_arealtime_vertex_branch_resolves_credentials_under_a_bound(monkeypatch):
    """The wiring half of the regression: the vertex branch of _arealtime must
    go through the bounded resolver, so a hung token refresh surfaces as a
    prompt error there rather than as an accepted-then-silent websocket."""

    async def hanging_token_refresh(**kwargs):
        await asyncio.sleep(30)

    def mock_get_llm_provider(model, api_base, api_key):
        return model, "vertex_ai", None, api_base

    monkeypatch.setattr(realtime_main, "get_llm_provider", mock_get_llm_provider)
    monkeypatch.setattr(realtime_main, "vertex_access_token_resolver", hanging_token_refresh)
    monkeypatch.setattr(realtime_main, "REALTIME_CREDENTIAL_RESOLUTION_TIMEOUT_SECONDS", 0.05)

    start = time.monotonic()
    with pytest.raises(ValueError, match="timed out fetching Google OAuth access token"):
        await realtime_main._arealtime.__wrapped__(
            model="gemini-live-2.5-flash",
            websocket=MagicMock(),
            litellm_logging_obj=FakeLogging(),
            vertex_credentials="fake-credentials",
            vertex_project="fake-project",
            vertex_location="us-central1",
        )
    assert time.monotonic() - start < 5


def test_client_secret_forwards_nested_transcription_model_untouched(monkeypatch):
    captured = _run_client_secret(
        session={
            "model": "gpt-4o-realtime-preview",
            "input_audio_transcription": {"model": "whisper-1"},
        },
        model=None,
        monkeypatch=monkeypatch,
    )
    session = captured["request_data"]["session"]
    assert session["model"] == "gpt-4o-realtime-preview"
    assert session["input_audio_transcription"]["model"] == "whisper-1"


class _CapturingConnect:
    def __init__(self, connection: object | None = None) -> None:
        self.url: str | None = None
        self.kwargs: dict[str, object] = {}
        self._connection: Final = connection if connection is not None else MagicMock()

    def __call__(self, url: str, **kwargs: object) -> "_CapturingConnect":
        self.url = url
        self.kwargs = kwargs
        return self

    async def __aenter__(self) -> object:
        return self._connection

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


@pytest.mark.asyncio
async def test_azure_health_check_resolves_stored_credentials(monkeypatch):
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="azure-rt",
                credential_values={
                    "api_key": "sk-from-credential",
                    "api_base": "https://example.openai.azure.com",
                    "api_version": "2025-04-01-preview",
                },
                credential_info={},
            )
        ],
    )
    connect = _CapturingConnect()
    with patch("websockets.connect", connect):
        assert await realtime_main._realtime_health_check(
            model="gpt-realtime",
            custom_llm_provider="azure",
            api_key=None,
            realtime_protocol="beta",
            model_params={"model": "azure/gpt-realtime", "litellm_credential_name": "azure-rt"},
        )
    assert connect.kwargs["additional_headers"] == {"api-key": "sk-from-credential"}
    assert connect.url is not None
    assert connect.url.startswith("wss://example.openai.azure.com")
    assert "api-version=2025-04-01-preview" in connect.url


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("custom_llm_provider", "model", "expected_url"),
    [
        ("xai", "grok-voice-latest", "wss://api.x.ai/v1/realtime?model=grok-voice-latest"),
        ("openai", "gpt-realtime", "wss://api.openai.com/v1/realtime?model=gpt-realtime"),
    ],
)
async def test_bearer_health_check_sends_stored_credential_as_bearer_token(
    monkeypatch, custom_llm_provider: str, model: str, expected_url: str
):
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="voice-key",
                credential_values={"api_key": "sk-from-credential"},
                credential_info={},
            )
        ],
    )
    connect = _CapturingConnect(_ScriptedConnection(_OPENAI_SESSION_CREATED_EVENT))
    with patch("websockets.connect", connect):
        assert await realtime_main._realtime_health_check(
            model=model,
            custom_llm_provider=custom_llm_provider,
            api_key=None,
            model_params={"model": f"{custom_llm_provider}/{model}", "litellm_credential_name": "voice-key"},
        )
    assert connect.kwargs["additional_headers"] == {"Authorization": "Bearer sk-from-credential"}
    assert connect.url == expected_url


@pytest.mark.asyncio
async def test_azure_health_check_probes_ga_transcription_url_for_transcription_model(local_model_cost_map):
    """Regression for LIT-6240: transcription-only models (mode audio_transcription
    in the cost map) are GA-only and 400 on the beta path, so the health probe
    must hit /openai/v1/realtime?intent=transcription like real calls do."""
    connect = _CapturingConnect()
    with patch("websockets.connect", connect):
        assert await realtime_main._realtime_health_check(
            model="gpt-realtime-whisper",
            custom_llm_provider="azure",
            api_key="fake-key",
            api_base="https://my-endpoint.openai.azure.com",
            api_version="2025-04-01-preview",
        )
    assert connect.url == "wss://my-endpoint.openai.azure.com/openai/v1/realtime?intent=transcription"


@pytest.mark.asyncio
async def test_azure_health_check_stays_on_ga_when_deployment_registration_overwrites_mode(
    local_model_cost_map, monkeypatch
):
    """In a live proxy, Router._register_deployment_in_model_cost writes the
    operator's deployment model_info (mode: realtime) over the catalog entry for
    azure/gpt-realtime-whisper, so mode alone misreads the model as speech-capable
    and the probe regresses to the beta path. supported_endpoints survives that
    registration and must keep the probe on the GA transcription path."""
    polluted = {**litellm.model_cost["azure/gpt-realtime-whisper"], "mode": "realtime"}
    monkeypatch.setitem(litellm.model_cost, "azure/gpt-realtime-whisper", polluted)
    connect = _CapturingConnect()
    with patch("websockets.connect", connect):
        assert await realtime_main._realtime_health_check(
            model="gpt-realtime-whisper",
            custom_llm_provider="azure",
            api_key="fake-key",
            api_base="https://my-endpoint.openai.azure.com",
            api_version="2025-04-01-preview",
        )
    assert connect.url == "wss://my-endpoint.openai.azure.com/openai/v1/realtime?intent=transcription"


def test_transcription_only_detection_falls_back_to_mode(local_model_cost_map):
    """azure/whisper-1 declares mode audio_transcription but no supported_endpoints,
    so only the mode signal can classify it as transcription-only."""
    assert realtime_main._is_transcription_only_realtime_model("whisper-1", "azure") is True


def test_transcription_only_detection_rejects_speech_model(local_model_cost_map):
    assert realtime_main._is_transcription_only_realtime_model("gpt-realtime-mini", "azure") is False


@pytest.mark.asyncio
async def test_azure_health_check_probes_the_ga_upstream_for_an_unconfigured_speech_model(monkeypatch):
    monkeypatch.delenv("LITELLM_AZURE_REALTIME_PROTOCOL", raising=False)
    connect = _CapturingConnect()
    with patch("websockets.connect", connect):
        assert await realtime_main._realtime_health_check(
            model="gpt-4o-realtime-preview",
            custom_llm_provider="azure",
            api_key="fake-key",
            api_base="https://my-endpoint.openai.azure.com",
            api_version="2024-10-01-preview",
        )
    assert connect.url == "wss://my-endpoint.openai.azure.com/openai/v1/realtime?model=gpt-4o-realtime-preview"


_AZURE_BETA_HEALTH_URL: Final = (
    "wss://my-endpoint.openai.azure.com/openai/realtime"
    "?api-version=2024-10-01-preview&deployment=gpt-4o-realtime-preview"
)


@pytest.mark.asyncio
async def test_azure_health_check_honors_deployment_realtime_protocol(monkeypatch):
    monkeypatch.delenv("LITELLM_AZURE_REALTIME_PROTOCOL", raising=False)
    connect = _CapturingConnect()
    with patch("websockets.connect", connect):
        assert await realtime_main._realtime_health_check(
            model="gpt-4o-realtime-preview",
            custom_llm_provider="azure",
            api_key="fake-key",
            api_base="https://my-endpoint.openai.azure.com",
            api_version="2024-10-01-preview",
            model_params={"realtime_protocol": "beta"},
        )
    assert connect.url == _AZURE_BETA_HEALTH_URL


@pytest.mark.asyncio
async def test_azure_health_check_honors_env_realtime_protocol(monkeypatch):
    monkeypatch.setenv("LITELLM_AZURE_REALTIME_PROTOCOL", "beta")
    connect = _CapturingConnect()
    with patch("websockets.connect", connect):
        assert await realtime_main._realtime_health_check(
            model="gpt-4o-realtime-preview",
            custom_llm_provider="azure",
            api_key="fake-key",
            api_base="https://my-endpoint.openai.azure.com",
            api_version="2024-10-01-preview",
        )
    assert connect.url == _AZURE_BETA_HEALTH_URL


class _ScriptedConnection:
    def __init__(self, *frames: str) -> None:
        self._frames: Final = iter(frames)

    async def recv(self) -> str:
        return next(self._frames)


class _SilentConnection:
    async def recv(self) -> str:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _ConnectionClosedBeforeAnyEvent:
    async def recv(self) -> str:
        raise ConnectionClosedError(Close(3000, "invalid_api_key"), None)


_OPENAI_INVALID_API_KEY_EVENT: Final = (
    '{"type": "error", "event_id": "event_1", "error": {"type": "invalid_request_error", "code": "invalid_api_key", '
    '"message": "Incorrect API key provided: sk-proj-****0000. You can find your API key at '
    'https://platform.openai.com/account/api-keys.", "param": null, "event_id": null}}'
)
_OPENAI_MISSING_AUTH_EVENT: Final = (
    '{"type": "error", "event_id": "event_2", "error": {"type": "invalid_request_error", "code": null, '
    '"message": "Missing bearer or basic authentication in header", "param": null, "event_id": null}}'
)
_OPENAI_SERVER_ERROR_EVENT: Final = (
    '{"type": "error", "event_id": "event_3", '
    '"error": {"type": "server_error", "code": null, "message": "The server had an error", "param": null}}'
)
_OPENAI_SESSION_CREATED_EVENT: Final = (
    '{"type": "session.created", "event_id": "event_4", "session": {"type": "realtime", "model": "gpt-realtime"}}'
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_event", "expected_exception", "expected_status", "expected_message"),
    [
        (_OPENAI_INVALID_API_KEY_EVENT, litellm.AuthenticationError, 401, "Incorrect API key provided"),
        (_OPENAI_MISSING_AUTH_EVENT, litellm.BadRequestError, 400, "Missing bearer or basic authentication"),
        (_OPENAI_SERVER_ERROR_EVENT, litellm.InternalServerError, 500, "The server had an error"),
    ],
)
async def test_openai_health_check_reports_the_first_error_event_as_unhealthy(
    first_event: str, expected_exception: type[Exception], expected_status: int, expected_message: str
):
    connect: Final = _CapturingConnect(_ScriptedConnection(first_event))
    with patch("websockets.connect", connect), pytest.raises(expected_exception) as raised:
        await realtime_main._realtime_health_check(
            model="gpt-realtime", custom_llm_provider="openai", api_key="sk-wrong"
        )
    assert raised.value.status_code == expected_status
    assert expected_message in str(raised.value)
    assert connect.url == "wss://api.openai.com/v1/realtime?model=gpt-realtime"


@pytest.mark.asyncio
async def test_openai_health_check_without_an_api_key_sends_no_auth_header_and_is_unhealthy():
    connect: Final = _CapturingConnect(_ScriptedConnection(_OPENAI_MISSING_AUTH_EVENT))
    with patch("websockets.connect", connect), pytest.raises(litellm.BadRequestError) as raised:
        await realtime_main._realtime_health_check(model="gpt-realtime", custom_llm_provider="openai", api_key=None)
    assert connect.kwargs["additional_headers"] == {}
    assert "Missing bearer or basic authentication" in str(raised.value)


@pytest.mark.asyncio
async def test_openai_health_check_is_healthy_once_session_created_arrives():
    connect: Final = _CapturingConnect(_ScriptedConnection(_OPENAI_SESSION_CREATED_EVENT))
    with patch("websockets.connect", connect):
        assert await realtime_main._realtime_health_check(
            model="gpt-realtime", custom_llm_provider="openai", api_key="sk-real"
        )


@pytest.mark.asyncio
async def test_openai_health_check_is_unhealthy_when_no_first_event_arrives_in_time():
    connect: Final = _CapturingConnect(_SilentConnection())
    with patch("websockets.connect", connect), pytest.raises(litellm.Timeout) as raised:
        await realtime_main._realtime_health_check(
            model="gpt-realtime", custom_llm_provider="openai", api_key="sk-real", first_event_timeout_seconds=0.01
        )
    assert raised.value.status_code == 408
    assert "no server event within 0.01 seconds" in str(raised.value)


@pytest.mark.asyncio
async def test_openai_health_check_is_unhealthy_when_the_socket_closes_before_any_event():
    connect: Final = _CapturingConnect(_ConnectionClosedBeforeAnyEvent())
    with patch("websockets.connect", connect), pytest.raises(ConnectionClosedError):
        await realtime_main._realtime_health_check(
            model="gpt-realtime", custom_llm_provider="openai", api_key="sk-wrong"
        )


@pytest.mark.asyncio
async def test_xai_health_check_trusts_the_handshake_without_reading_a_first_event():
    connect: Final = _CapturingConnect(_ScriptedConnection())
    with patch("websockets.connect", connect):
        assert await realtime_main._realtime_health_check(
            model="grok-voice-latest", custom_llm_provider="xai", api_key="sk-wrong"
        )
    assert connect.url == "wss://api.x.ai/v1/realtime?model=grok-voice-latest"


class _ConnectThatStopsAfterCapturingTheUrl:
    url: str | None = None

    def __call__(self, url: str, **kwargs: object) -> "_ConnectThatStopsAfterCapturingTheUrl":
        self.url = url
        return self

    async def __aenter__(self) -> None:
        raise RuntimeError("backend url captured, nothing to bridge")

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


@pytest.mark.asyncio
async def test_arealtime_azure_ai_on_a_foundry_host_connects_to_the_azure_openai_realtime_route():
    connect: Final = _ConnectThatStopsAfterCapturingTheUrl()
    with patch("websockets.connect", connect):
        await realtime_main._arealtime.__wrapped__(
            model="azure_ai/gpt-realtime-mini",
            websocket=MagicMock(),
            api_base="https://my-project.services.ai.azure.com",
            api_key="fake-key",
            litellm_logging_obj=FakeLogging(),
        )
    assert connect.url == "wss://my-project.services.ai.azure.com/openai/v1/realtime?model=gpt-realtime-mini"


class _ClientWebSocketWithHeaders:
    def __init__(self, headers: tuple[tuple[bytes, bytes], ...]) -> None:
        self.scope: Final = {"headers": headers}


_GA_CLIENT: Final = _ClientWebSocketWithHeaders(headers=())
_BETA_CLIENT: Final = _ClientWebSocketWithHeaders(headers=((b"openai-beta", b"realtime=v1"),))


async def _azure_backend_url_dialed_for(websocket: _ClientWebSocketWithHeaders, **kwargs: object) -> str | None:
    connect: Final = _ConnectThatStopsAfterCapturingTheUrl()
    with patch("websockets.connect", connect):
        await realtime_main._arealtime.__wrapped__(
            model="azure/gpt-realtime",
            websocket=websocket,
            api_base="https://my-endpoint.openai.azure.com",
            api_key="fake-key",
            litellm_logging_obj=FakeLogging(),
            **kwargs,
        )
    return connect.url


@pytest.mark.asyncio
async def test_arealtime_azure_ga_client_without_beta_header_dials_the_ga_upstream(monkeypatch):
    monkeypatch.delenv("LITELLM_AZURE_REALTIME_PROTOCOL", raising=False)
    assert (
        await _azure_backend_url_dialed_for(_GA_CLIENT)
        == "wss://my-endpoint.openai.azure.com/openai/v1/realtime?model=gpt-realtime"
    )


@pytest.mark.asyncio
async def test_arealtime_azure_beta_header_client_keeps_the_beta_upstream(monkeypatch):
    monkeypatch.delenv("LITELLM_AZURE_REALTIME_PROTOCOL", raising=False)
    assert await _azure_backend_url_dialed_for(_BETA_CLIENT) == (
        "wss://my-endpoint.openai.azure.com/openai/realtime?api-version=2024-10-01-preview&deployment=gpt-realtime"
    )


@pytest.mark.asyncio
async def test_arealtime_azure_explicit_beta_protocol_wins_over_a_ga_client(monkeypatch):
    monkeypatch.delenv("LITELLM_AZURE_REALTIME_PROTOCOL", raising=False)
    assert await _azure_backend_url_dialed_for(_GA_CLIENT, realtime_protocol="beta") == (
        "wss://my-endpoint.openai.azure.com/openai/realtime?api-version=2024-10-01-preview&deployment=gpt-realtime"
    )


@pytest.mark.asyncio
async def test_arealtime_azure_env_beta_protocol_wins_over_a_ga_client(monkeypatch):
    monkeypatch.setenv("LITELLM_AZURE_REALTIME_PROTOCOL", "beta")
    assert await _azure_backend_url_dialed_for(_GA_CLIENT) == (
        "wss://my-endpoint.openai.azure.com/openai/realtime?api-version=2024-10-01-preview&deployment=gpt-realtime"
    )
