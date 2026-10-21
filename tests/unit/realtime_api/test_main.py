import asyncio
import functools
import ipaddress
import json
import ssl
import threading
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import TracebackType
from typing import ClassVar, Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from openai import AsyncOpenAI
from websockets.exceptions import ConnectionClosed

import litellm
from litellm.constants import REALTIME_SESSION_FAILURE_LOGGED_KEY
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.realtime_streaming import TRANSLATION_GUARDRAILS_UNSUPPORTED_MESSAGE, RealTimeStreaming
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openai.workload_identity import _workload_identity_auth
from litellm.models.credentials import CredentialItem
from litellm.realtime_api import main as realtime_main
from litellm.realtime_api.main import with_resolved_session_model
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.realtime import RealtimeQueryParams


@pytest.fixture
def local_model_cost_map(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


class FakeLogging:
    def update_from_kwargs(self, **kwargs):
        pass

    def pre_call(self, **kwargs):
        pass


class _CallCapture:
    def __init__(self) -> None:
        self.values: Mapping[str, object] | None = None


def test_resolves_top_level_session_model():
    resolved = with_resolved_session_model({"model": "alias/gpt-realtime"}, "gpt-realtime")
    assert resolved == {"model": "gpt-realtime"}


def test_session_without_model_is_returned_unchanged():
    session = {"type": "realtime", "audio": {"input": {}}}
    assert with_resolved_session_model(session, "gpt-realtime") == session


def test_does_not_clobber_flat_transcription_model():
    """The nested transcription model is a different model than the realtime
    conversation model and must not be overwritten with the routing model."""
    resolved = with_resolved_session_model(
        {"model": "gpt-4o-realtime-preview", "input_audio_transcription": {"model": "whisper-1"}},
        "gpt-4o-realtime-preview",
    )
    assert resolved["input_audio_transcription"]["model"] == "whisper-1"


def test_does_not_clobber_nested_audio_transcription_model():
    resolved = with_resolved_session_model(
        {
            "model": "gpt-4o-realtime-preview",
            "audio": {"input": {"transcription": {"model": "whisper-1"}}},
        },
        "gpt-4o-realtime-preview",
    )
    assert resolved["audio"]["input"]["transcription"]["model"] == "whisper-1"


def test_original_session_is_not_mutated():
    session = {"model": "alias/gpt-realtime"}
    with_resolved_session_model(session, "gpt-realtime")
    assert session == {"model": "alias/gpt-realtime"}


def _run_client_secret(session, model, monkeypatch):
    captured = {}

    async def mock_handler(**kwargs):
        captured.update(kwargs)
        return object()

    def mock_get_llm_provider(model, api_base, api_key, custom_llm_provider: str | None = None):
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

    def mock_get_llm_provider(model, api_base, api_key, custom_llm_provider: str | None = None):
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

    def mock_get_llm_provider(model, api_base, api_key, custom_llm_provider: str | None = None):
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


@pytest.mark.asyncio
async def test_arealtime_live_session_dispatches_to_openai_live_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Final = _CallCapture()

    def mock_get_llm_provider(
        model: str,
        api_base: str | None,
        api_key: str | None,
        custom_llm_provider: str | None = None,
    ) -> tuple[str, str, str | None, str | None]:
        return "gpt-live-1", "openai", "resolved-key", "https://live-gateway.example/v1"

    class FakeLiveHandler:
        async def async_live_session(self, **kwargs: object) -> None:
            captured.values = kwargs

    monkeypatch.setattr(realtime_main, "get_llm_provider", mock_get_llm_provider)
    monkeypatch.setattr(realtime_main, "openai_live_sessions", FakeLiveHandler())
    start_frame: Final[dict[str, object]] = {
        "type": "session.start",
        "session": {"model": "live-alias", "instructions": "Stay brief"},
    }
    websocket: Final = MagicMock()
    logging_obj: Final = FakeLogging()

    await realtime_main._arealtime.__wrapped__(
        model="live-alias",
        websocket=websocket,
        litellm_logging_obj=logging_obj,
        live_session_start=start_frame,
    )

    assert captured.values is not None
    assert captured.values["model"] == "gpt-live-1"
    assert captured.values["websocket"] is websocket
    assert captured.values["logging_obj"] is logging_obj
    assert captured.values["session_start"] == start_frame
    assert captured.values["api_base"] == "https://live-gateway.example/v1"
    assert captured.values["api_key"] == "resolved-key"


@pytest.mark.asyncio
async def test_arealtime_live_session_rejects_matching_guardrail_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingGuardrail(CustomGuardrail):
        def __init__(self) -> None:
            super().__init__(
                guardrail_name="live-blocker",
                event_hook=GuardrailEventHooks.realtime_input_transcription,
                default_on=True,
            )
            self.request_data: dict[str, object] | None = None

        def should_run_guardrail(self, data: dict[str, object], event_type: GuardrailEventHooks) -> bool:
            self.request_data = data
            return super().should_run_guardrail(data, event_type)

    class FakeLiveHandler:
        def __init__(self) -> None:
            self.called = False

        async def async_live_session(self, **kwargs: object) -> None:
            self.called = True

    guardrail: Final = RecordingGuardrail()
    live_handler: Final = FakeLiveHandler()

    def mock_get_llm_provider(
        model: str,
        api_base: str | None,
        api_key: str | None,
        custom_llm_provider: str | None = None,
    ) -> tuple[str, str, str | None, str | None]:
        return "gpt-live-1", "openai", "resolved-key", "https://live-gateway.example/v1"

    monkeypatch.setattr(realtime_main, "get_llm_provider", mock_get_llm_provider)
    monkeypatch.setattr(realtime_main, "openai_live_sessions", live_handler)
    monkeypatch.setattr(litellm, "callbacks", [guardrail])

    with pytest.raises(ValueError, match="Guardrails are not supported on OpenAI Live sessions"):
        await realtime_main._arealtime.__wrapped__(
            model="live-alias",
            websocket=MagicMock(),
            litellm_logging_obj=FakeLogging(),
            litellm_metadata={"session_id": "live-session"},
            live_session_start={"type": "session.start", "session": {"model": "live-alias"}},
        )

    assert guardrail.request_data == {"litellm_metadata": {"session_id": "live-session"}, "stream": True}
    assert live_handler.called is False


@pytest.mark.asyncio
async def test_arealtime_live_session_rejects_streaming_only_pre_call_guardrail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLiveHandler:
        def __init__(self) -> None:
            self.called = False

        async def async_live_session(self, **kwargs: object) -> None:
            self.called = True

    guardrail: Final = CustomGuardrail(
        guardrail_name="streaming-only-blocker",
        event_hook=GuardrailEventHooks.pre_call,
        default_on=True,
        stream_scope="streaming",
    )
    live_handler: Final = FakeLiveHandler()

    def mock_get_llm_provider(
        model: str,
        api_base: str | None,
        api_key: str | None,
        custom_llm_provider: str | None = None,
    ) -> tuple[str, str, str | None, str | None]:
        return "gpt-live-1", "openai", "resolved-key", "https://live-gateway.example/v1"

    monkeypatch.setattr(realtime_main, "get_llm_provider", mock_get_llm_provider)
    monkeypatch.setattr(realtime_main, "openai_live_sessions", live_handler)
    monkeypatch.setattr(litellm, "callbacks", [guardrail])

    with pytest.raises(ValueError, match="Guardrails are not supported on OpenAI Live sessions"):
        await realtime_main._arealtime.__wrapped__(
            model="live-alias",
            websocket=MagicMock(),
            litellm_logging_obj=FakeLogging(),
            litellm_metadata={"session_id": "live-session"},
            live_session_start={"type": "session.start", "session": {"model": "live-alias"}},
        )

    assert live_handler.called is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key_metadata",
    (
        {"user_api_key_metadata": {"disable_global_guardrails": True}},
        {"user_api_key_team_metadata": {"opted_out_global_guardrails": ["default-on-blocker"]}},
    ),
)
async def test_arealtime_key_metadata_cannot_opt_out_of_a_default_on_realtime_guardrail(
    monkeypatch: pytest.MonkeyPatch, key_metadata: dict[str, object]
) -> None:
    class FakeLiveHandler:
        def __init__(self) -> None:
            self.called = False

        async def async_live_session(self, **kwargs: object) -> None:
            self.called = True

    live_handler: Final = FakeLiveHandler()
    monkeypatch.setattr(realtime_main, "openai_live_sessions", live_handler)
    monkeypatch.setattr(
        litellm,
        "callbacks",
        [
            CustomGuardrail(
                guardrail_name="default-on-blocker",
                event_hook=GuardrailEventHooks.realtime_input_transcription,
                default_on=True,
            )
        ],
    )

    with pytest.raises(ValueError, match="Guardrails are not supported on OpenAI Live sessions"):
        await realtime_main._arealtime.__wrapped__(
            model="openai/gpt-live-1",
            websocket=MagicMock(),
            api_key="fake-key",
            api_base="https://live-gateway.example/v1",
            litellm_logging_obj=FakeLogging(),
            metadata=key_metadata,
            live_session_start={"type": "session.start", "session": {"model": "gpt-live-1"}},
        )

    assert live_handler.called is False


@pytest.mark.asyncio
async def test_arealtime_live_session_rejects_non_openai_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_get_llm_provider(
        model: str,
        api_base: str | None,
        api_key: str | None,
        custom_llm_provider: str | None = None,
    ) -> tuple[str, str, str | None, str | None]:
        return "claude-live", "anthropic", None, None

    monkeypatch.setattr(realtime_main, "get_llm_provider", mock_get_llm_provider)

    with pytest.raises(ValueError, match="anthropic"):
        await realtime_main._arealtime.__wrapped__(
            model="anthropic/claude-live",
            websocket=MagicMock(),
            litellm_logging_obj=FakeLogging(),
            live_session_start={"type": "session.start", "session": {"model": "claude-live"}},
        )


@pytest.mark.asyncio
async def test_arealtime_without_live_start_preserves_openai_realtime_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: Final = _CallCapture()

    def mock_get_llm_provider(
        model: str,
        api_base: str | None,
        api_key: str | None,
        custom_llm_provider: str | None = None,
    ) -> tuple[str, str, str | None, str | None]:
        return "gpt-realtime", "openai", "resolved-key", "https://realtime-gateway.example/v1"

    class FakeRealtimeHandler:
        async def async_realtime(self, **kwargs: object) -> None:
            captured.values = kwargs

    monkeypatch.setattr(realtime_main, "get_llm_provider", mock_get_llm_provider)
    monkeypatch.setattr(realtime_main, "openai_realtime", FakeRealtimeHandler())

    await realtime_main._arealtime.__wrapped__(
        model="gpt-realtime",
        websocket=MagicMock(),
        litellm_logging_obj=FakeLogging(),
    )

    assert captured.values is not None
    assert captured.values["model"] == "gpt-realtime"
    assert captured.values["api_key"] == "resolved-key"
    assert captured.values["api_base"] == "https://realtime-gateway.example/v1"


class _CapturingConnect:
    def __init__(self) -> None:
        self.url: str | None = None
        self.kwargs: dict[str, object] = {}

    def __call__(self, url: str, **kwargs: object) -> "_CapturingConnect":
        self.url = url
        self.kwargs = kwargs
        return self

    async def __aenter__(self) -> MagicMock:
        return MagicMock()

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
    connect = _CapturingConnect()
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
async def test_bearer_health_check_sends_the_secret_behind_an_oidc_credential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS", str(tmp_path))
    token: Final = tmp_path / "token"
    token.write_text("sk-from-token-file")
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [CredentialItem(credential_name="voice-sa", credential_values={"api_key": f"oidc/file/{token}"}, credential_info={})],
    )
    connect: Final = _CapturingConnect()
    with patch("websockets.connect", connect):
        assert await realtime_main._realtime_health_check(
            model="gpt-realtime",
            custom_llm_provider="openai",
            api_key=None,
            model_params={"model": "openai/gpt-realtime", "litellm_credential_name": "voice-sa"},
        )
    assert connect.kwargs["additional_headers"] == {"Authorization": "Bearer sk-from-token-file"}


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


@pytest.mark.asyncio
async def test_arealtime_routes_a_bare_model_name_to_its_explicit_provider():
    connect: Final = _ConnectThatStopsAfterCapturingTheUrl()
    with patch("websockets.connect", connect):
        await realtime_main._arealtime.__wrapped__(
            model="my-realtime",
            websocket=MagicMock(close=AsyncMock()),
            custom_llm_provider="openai",
            api_base="https://realtime-gateway.example/v1",
            api_key="fake-key",
            query_params={"model": "my-realtime"},
            litellm_logging_obj=FakeLogging(),
        )
    assert connect.url == "wss://realtime-gateway.example/v1/realtime?model=my-realtime"


class _GatewayThatRecordsTheUrl:
    url: str | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.url = str(request.url)
        return httpx.Response(200, json={"value": "ek_test"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "route_kwargs", "path"),
    [
        (realtime_main.acreate_realtime_client_secret, {}, "client_secrets"),
        (realtime_main.acreate_realtime_transcription_session, {}, "transcription_sessions"),
        (realtime_main.arealtime_calls, {"openai_ephemeral_key": "ek_test", "sdp_body": b"v=0"}, "calls"),
    ],
)
async def test_realtime_http_routes_send_a_bare_model_name_to_its_explicit_provider(
    route: Callable[..., Awaitable[object]], route_kwargs: Mapping[str, object], path: str
) -> None:
    gateway: Final = _GatewayThatRecordsTheUrl()
    await route.__wrapped__(
        model="my-realtime",
        custom_llm_provider="openai",
        api_base="https://realtime-gateway.example/v1",
        api_key="fake-key",
        litellm_logging_obj=FakeLogging(),
        client=AsyncHTTPHandler(transport=httpx.MockTransport(gateway)),
        **route_kwargs,
    )
    assert gateway.url == f"https://realtime-gateway.example/v1/realtime/{path}"


class _GatewayThatRecordsTheBody:
    body: object = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.body = json.loads(request.content)
        return httpx.Response(200, json={"client_secret": {"value": "ek_test"}})


@pytest.mark.asyncio
async def test_transcription_session_sends_the_deployment_model_upstream_in_every_transcription_shape() -> None:
    gateway: Final = _GatewayThatRecordsTheBody()
    await realtime_main.acreate_realtime_transcription_session.__wrapped__(
        model="openai/gpt-transcribe",
        transcription_session={
            "audio": {"input": {"transcription": {"model": "team-transcriber", "language": "en"}}},
            "input_audio_transcription": {"model": "team-transcriber"},
            "input_audio_format": "pcm16",
        },
        api_base="https://realtime-gateway.example/v1",
        api_key="fake-key",
        litellm_logging_obj=FakeLogging(),
        client=AsyncHTTPHandler(transport=httpx.MockTransport(gateway)),
    )
    assert gateway.body == {
        "audio": {"input": {"transcription": {"model": "gpt-transcribe", "language": "en"}}},
        "input_audio_transcription": {"model": "gpt-transcribe"},
        "input_audio_format": "pcm16",
    }


def _create_realtime_session_with_an_explicit_null(route: str, client: AsyncHTTPHandler) -> Awaitable[httpx.Response]:
    if route == "client_secrets":
        return realtime_main.acreate_realtime_client_secret.__wrapped__(
            session={"type": "realtime", "model": "gpt-realtime", "audio": {"input": {"turn_detection": None}}},
            custom_llm_provider="openai",
            api_base="https://realtime-gateway.example/v1",
            api_key="fake-key",
            litellm_logging_obj=FakeLogging(),
            client=client,
        )
    if route == "translation_client_secrets":
        return realtime_main.acreate_realtime_translation_client_secret.__wrapped__(
            model="gpt-realtime-translate",
            session={"audio": {"input": {"noise_reduction": None}}},
            custom_llm_provider="openai",
            api_base="https://realtime-gateway.example/v1",
            api_key="fake-key",
            litellm_logging_obj=_realtime_http_logging(route),
            client=client,
        )
    return realtime_main.acreate_realtime_transcription_session.__wrapped__(
        model="openai/gpt-transcribe",
        transcription_session={"audio": {"input": {"turn_detection": None}}},
        api_base="https://realtime-gateway.example/v1",
        api_key="fake-key",
        litellm_logging_obj=FakeLogging(),
        client=client,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "body_path"),
    (
        ("client_secrets", ("session", "audio", "input", "turn_detection")),
        ("translation_client_secrets", ("session", "audio", "input", "noise_reduction")),
        ("transcription_sessions", ("audio", "input", "turn_detection")),
    ),
)
async def test_realtime_session_requests_forward_an_explicit_null_upstream(
    route: str, body_path: tuple[str, ...]
) -> None:
    gateway: Final = _GatewayThatRecordsTheBody()

    await _create_realtime_session_with_an_explicit_null(
        route, AsyncHTTPHandler(transport=httpx.MockTransport(gateway))
    )

    parent: Final = functools.reduce(lambda node, key: node[key], body_path[:-1], gateway.body)
    assert body_path[-1] in parent
    assert parent[body_path[-1]] is None


class _ClientWebSocketWithHeaders:
    def __init__(self, headers: tuple[tuple[bytes, bytes], ...]) -> None:
        self.scope: Final = {"headers": headers}


_GA_CLIENT: Final = _ClientWebSocketWithHeaders(headers=())
_BETA_CLIENT: Final = _ClientWebSocketWithHeaders(headers=((b"openai-beta", b"realtime=v1"),))


_GA_DEFAULT_SESSIONS: Final = (
    pytest.param("gpt-realtime-2.1", "realtime", None, id="ga-only-model"),
    pytest.param("gpt-realtime-2", "realtime", None, id="ga-only-alias"),
    pytest.param("gpt-realtime-2-2026-05-06", "realtime", None, id="ga-only-snapshot"),
    pytest.param("my-transcribe-deployment", "realtime", {"intent": "transcription"}, id="transcription-intent"),
    pytest.param("my-translate-deployment", "translation", None, id="translation-mode"),
)


@pytest.mark.parametrize(("model", "realtime_mode", "query_params"), _GA_DEFAULT_SESSIONS)
def test_azure_realtime_protocol_defaults_to_ga_when_none_is_configured(
    local_model_cost_map, model: str, realtime_mode: str, query_params: RealtimeQueryParams | None
) -> None:
    from litellm.llms.azure.realtime.handler import azure_realtime_protocol_for_client

    protocol: Final = azure_realtime_protocol_for_client(
        None, model=model, realtime_mode=realtime_mode, query_params=query_params, websocket=_BETA_CLIENT
    )

    assert protocol.name == "GA"


@pytest.mark.parametrize(("model", "realtime_mode", "query_params"), _GA_DEFAULT_SESSIONS)
def test_azure_realtime_protocol_configured_by_the_operator_wins(
    local_model_cost_map, model: str, realtime_mode: str, query_params: RealtimeQueryParams | None
) -> None:
    from litellm.llms.azure.realtime.handler import azure_realtime_protocol_for_client

    protocol: Final = azure_realtime_protocol_for_client(
        "beta", model=model, realtime_mode=realtime_mode, query_params=query_params, websocket=_GA_CLIENT
    )

    assert protocol.name == "beta"


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


class _BetaClientThatStaysConnected:
    def __init__(self) -> None:
        self.scope: Final = {"headers": ((b"openai-beta", b"realtime=v1"),)}
        self.sent: tuple[str, ...] = ()

    async def receive_text(self) -> str:
        return await asyncio.get_running_loop().create_future()

    async def send_text(self, data: str) -> None:
        self.sent = (*self.sent, data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        return None


class _ConnectToABackendThatSendsOneFrame:
    def __init__(self, frame: Mapping[str, object]) -> None:
        self.url: str | None = None
        self._backend: Final = MagicMock(recv=AsyncMock(side_effect=[json.dumps(frame), ConnectionClosed(None, None)]))

    def __call__(
        self, url: str, *, additional_headers: Mapping[str, str], max_size: int, ssl: object
    ) -> "_ConnectToABackendThatSendsOneFrame":
        self.url = url
        return self

    async def __aenter__(self) -> MagicMock:
        return self._backend

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


def _realtime_websocket_logging(*failure_callbacks: CustomLogger) -> Logging:
    return Logging(
        model="gpt-realtime",
        messages=[],
        stream=False,
        call_type="_arealtime",
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        litellm_call_id="test-realtime-websocket",
        function_id="test-realtime-websocket",
        dynamic_async_failure_callbacks=list(failure_callbacks),
    )


_GA_SESSION_CREATED: Final = {
    "type": "session.created",
    "event_id": "event_1",
    "session": {
        "type": "realtime",
        "id": "sess_1",
        "model": "my-realtime-deployment",
        "output_modalities": ["text"],
        "audio": {"output": {"voice": "marin"}},
    },
}


async def _azure_session_created_seen_by_a_beta_client(
    session_kwargs: Mapping[str, object],
) -> tuple[str | None, object]:
    client: Final = _BetaClientThatStaysConnected()
    connect: Final = _ConnectToABackendThatSendsOneFrame(_GA_SESSION_CREATED)
    with patch("websockets.connect", connect):
        await realtime_main._arealtime.__wrapped__(
            model="azure/my-realtime-deployment",
            websocket=client,
            api_base="https://my-endpoint.openai.azure.com",
            api_key="fake-key",
            litellm_logging_obj=_realtime_websocket_logging(),
            **session_kwargs,
        )
    return connect.url, json.loads(client.sent[0])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("session_kwargs", "expected_url"),
    (
        pytest.param(
            {"realtime_protocol": "GA"},
            "wss://my-endpoint.openai.azure.com/openai/v1/realtime?model=my-realtime-deployment",
            id="operator-set-ga",
        ),
        pytest.param(
            {"query_params": {"intent": "transcription"}},
            "wss://my-endpoint.openai.azure.com/openai/v1/realtime?intent=transcription",
            id="transcription-intent",
        ),
    ),
)
async def test_arealtime_azure_beta_client_on_a_ga_session_it_did_not_need_bridging_for_keeps_ga_session_events(
    monkeypatch: pytest.MonkeyPatch, session_kwargs: Mapping[str, object], expected_url: str
) -> None:
    monkeypatch.delenv("LITELLM_AZURE_REALTIME_PROTOCOL", raising=False)

    assert await _azure_session_created_seen_by_a_beta_client(session_kwargs) == (expected_url, _GA_SESSION_CREATED)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_model_kwargs",
    (
        pytest.param({"base_model": "azure/gpt-realtime-2.1"}, id="base-model-param"),
        pytest.param({"metadata": {"model_info": {"base_model": "azure/gpt-realtime-2.1"}}}, id="model-info"),
    ),
)
async def test_arealtime_azure_bridges_a_beta_client_to_ga_when_the_deployment_base_model_is_ga_only(
    local_model_cost_map, monkeypatch: pytest.MonkeyPatch, base_model_kwargs: Mapping[str, object]
) -> None:
    monkeypatch.delenv("LITELLM_AZURE_REALTIME_PROTOCOL", raising=False)

    assert await _azure_session_created_seen_by_a_beta_client(base_model_kwargs) == (
        "wss://my-endpoint.openai.azure.com/openai/v1/realtime?model=my-realtime-deployment",
        _GA_SESSION_CREATED_BRIDGED_TO_BETA,
    )


_GA_SESSION_CREATED_BRIDGED_TO_BETA: Final = {
    "type": "session.created",
    "event_id": "event_1",
    "session": {"id": "sess_1", "model": "my-realtime-deployment", "modalities": ["text"], "voice": "marin"},
}

_ROUTER_REQUEST_SHAPES: Final = (
    pytest.param({}, id="sdk-router-call"),
    pytest.param({"metadata": {"user_api_key_hash": "hashed-key"}}, id="proxy-request-metadata"),
)


def _router_with_an_azure_deployment_aliasing(deployment_model: str, base_model: str) -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": "team-alias",
                "litellm_params": {
                    "model": deployment_model,
                    "api_base": "https://my-endpoint.openai.azure.com",
                    "api_key": "fake-key",
                },
                "model_info": {"base_model": base_model},
            }
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("request_kwargs", _ROUTER_REQUEST_SHAPES)
async def test_router_realtime_bridges_a_beta_client_to_ga_when_model_info_base_model_is_ga_only(
    local_model_cost_map, monkeypatch: pytest.MonkeyPatch, request_kwargs: Mapping[str, object]
) -> None:
    monkeypatch.delenv("LITELLM_AZURE_REALTIME_PROTOCOL", raising=False)
    router: Final = _router_with_an_azure_deployment_aliasing(
        "azure/my-realtime-deployment", "azure/gpt-realtime-2.1-mini"
    )
    client: Final = _BetaClientThatStaysConnected()
    connect: Final = _ConnectToABackendThatSendsOneFrame(_GA_SESSION_CREATED)

    with patch("websockets.connect", connect):
        await router._arealtime(model="team-alias", websocket=client, **request_kwargs)

    assert (connect.url, json.loads(client.sent[0])) == (
        "wss://my-endpoint.openai.azure.com/openai/v1/realtime?model=my-realtime-deployment",
        _GA_SESSION_CREATED_BRIDGED_TO_BETA,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("request_kwargs", _ROUTER_REQUEST_SHAPES)
async def test_router_transcription_session_uses_the_ga_route_when_model_info_base_model_is_ga_only(
    local_model_cost_map, request_kwargs: Mapping[str, object]
) -> None:
    router: Final = _router_with_an_azure_deployment_aliasing(
        "azure/my-whisper-deployment", "azure/gpt-realtime-whisper"
    )
    gateway: Final = _GatewayThatRecordsTheUrl()

    await router.acreate_realtime_transcription_session(
        model="team-alias",
        transcription_session={"input_audio_format": "pcm16"},
        client=AsyncHTTPHandler(transport=httpx.MockTransport(gateway)),
        **request_kwargs,
    )

    assert gateway.url == "https://my-endpoint.openai.azure.com/openai/v1/realtime/transcription_sessions"


async def _vertex_provider_config_for(monkeypatch, model: str, vertex_location: str | None):
    from litellm.llms.vertex_ai.audio_transcription.realtime_transformation import VertexChirpRealtimeConfig
    from litellm.llms.vertex_ai.realtime.transformation import VertexAIRealtimeConfig

    captured: dict[str, object] = {}

    def mock_get_llm_provider(model, api_base, api_key, custom_llm_provider: str | None = None):
        return model.removeprefix("vertex_ai/"), "vertex_ai", None, api_base

    async def mock_token_resolver(**kwargs):
        return "access-token", kwargs["project_id"]

    async def mock_async_realtime(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(realtime_main, "get_llm_provider", mock_get_llm_provider)
    monkeypatch.setattr(realtime_main, "vertex_access_token_resolver", mock_token_resolver)
    monkeypatch.setattr(realtime_main.base_llm_http_handler, "async_realtime", mock_async_realtime)
    monkeypatch.setattr(litellm, "vertex_location", None)
    monkeypatch.delenv("VERTEXAI_LOCATION", raising=False)
    await realtime_main._arealtime.__wrapped__(
        model=model,
        websocket=MagicMock(),
        litellm_logging_obj=FakeLogging(),
        query_params={"model": model, "intent": "transcription"},
        vertex_credentials="fake-credentials",
        vertex_project="proj-1",
        vertex_location=vertex_location,
    )
    provider_config = captured["provider_config"]
    assert isinstance(provider_config, (VertexAIRealtimeConfig, VertexChirpRealtimeConfig))
    return provider_config, captured["model"]


@pytest.mark.asyncio
async def test_arealtime_routes_chirp_models_to_the_speech_to_text_backend(monkeypatch):
    from litellm.llms.vertex_ai.audio_transcription.realtime_transformation import VertexChirpRealtimeConfig

    provider_config, model = await _vertex_provider_config_for(monkeypatch, "vertex_ai/chirp_3", None)
    assert isinstance(provider_config, VertexChirpRealtimeConfig)
    assert model == "chirp_3"
    assert provider_config.get_complete_url(None, model) == "us-speech.googleapis.com"
    assert provider_config.validate_environment({}, model, "https://us-speech.googleapis.com") == {}


@pytest.mark.asyncio
async def test_arealtime_routes_chirp_models_to_the_configured_speech_region(monkeypatch):
    provider_config, model = await _vertex_provider_config_for(monkeypatch, "vertex_ai/chirp_3", "europe-west4")
    assert provider_config.get_complete_url(None, model) == "europe-west4-speech.googleapis.com"


@pytest.mark.asyncio
async def test_arealtime_keeps_gemini_live_on_the_vertex_realtime_websocket(monkeypatch):
    from litellm.llms.vertex_ai.realtime.transformation import VertexAIRealtimeConfig

    provider_config, model = await _vertex_provider_config_for(monkeypatch, "vertex_ai/gemini-live-2.5-flash", None)
    assert isinstance(provider_config, VertexAIRealtimeConfig)
    assert provider_config.get_complete_url(None, model).startswith("wss://us-central1-aiplatform.googleapis.com/")


@pytest.mark.asyncio
async def test_realtime_health_check_names_the_batch_mode_for_chirp_models():
    with pytest.raises(ValueError, match="mode audio_transcription"):
        await realtime_main._realtime_health_check(model="chirp_3", custom_llm_provider="vertex_ai", api_key=None)


class _ClosableGaClientWebSocket:
    def __init__(self) -> None:
        self.scope: Final = {"headers": ()}

    async def close(self, code: int = 1000, reason: str = "") -> None:
        return None


@pytest.mark.asyncio
async def test_arealtime_openai_forwards_the_intent_query_param_to_the_upstream_url():
    connect: Final = _ConnectThatStopsAfterCapturingTheUrl()
    with patch("websockets.connect", connect):
        await realtime_main._arealtime.__wrapped__(
            model="openai/gpt-realtime",
            websocket=_ClosableGaClientWebSocket(),
            api_key="fake-key",
            query_params={"model": "openai/gpt-realtime", "intent": "chat"},
            litellm_logging_obj=FakeLogging(),
        )
    assert connect.url == "wss://api.openai.com/v1/realtime?model=gpt-realtime&intent=chat"


@pytest.mark.asyncio
@pytest.mark.parametrize("intent", (None, "transcription"))
async def test_arealtime_openai_connects_to_api_base_rather_than_the_callers_client_base_url(
    intent: str | None,
) -> None:
    connect: Final = _ConnectThatStopsAfterCapturingTheUrl()
    query_params: Final[RealtimeQueryParams] = (
        {"model": "openai/gateway-realtime", "intent": intent} if intent else {"model": "openai/gateway-realtime"}
    )
    with patch("websockets.connect", connect), patch("websockets.asyncio.client.connect", connect):
        await realtime_main._arealtime.__wrapped__(
            model="openai/gateway-realtime",
            websocket=_ClosableGaClientWebSocket(),
            api_key="fake-key",
            api_base="https://configured-gateway.example/",
            client=AsyncOpenAI(api_key="caller-key", base_url="https://caller-client.example/v1"),
            query_params=query_params,
            litellm_logging_obj=FakeLogging(),
        )

    expected_query: Final = "model=gateway-realtime&intent=transcription" if intent else "model=gateway-realtime"
    assert connect.url == f"wss://configured-gateway.example/v1/realtime?{expected_query}"


@pytest.mark.parametrize(
    ("model", "api_base", "client_query_params", "expected_backend_url"),
    [
        (
            "openai/gpt-live-transcribe",
            "https://api.openai.com/",
            {"model": "my-transcribe-alias", "intent": "transcription"},
            "wss://api.openai.com/v1/realtime?intent=transcription",
        ),
        (
            "openai/gpt-live-transcribe",
            "https://eu.api.openai.com/",
            {"model": "my-transcribe-alias", "intent": "transcription"},
            "wss://eu.api.openai.com/v1/realtime?intent=transcription",
        ),
        (
            "openai/gpt-live-transcribe",
            "https://api.openai.com/",
            {"model": "my-realtime-alias"},
            "wss://api.openai.com/v1/realtime?model=gpt-live-transcribe",
        ),
        (
            "openai/gateway-transcribe-alias",
            "https://gateway.example/",
            {"model": "my-transcribe-alias", "intent": "transcription"},
            "wss://gateway.example/v1/realtime?model=gateway-transcribe-alias&intent=transcription",
        ),
        (
            "xai/grok-voice",
            "https://api.x.ai/v1",
            {"model": "my-voice-alias", "intent": "transcription"},
            "wss://api.x.ai/v1/realtime?model=grok-voice&intent=transcription",
        ),
    ],
)
@pytest.mark.asyncio
async def test_arealtime_drops_model_from_the_upstream_url_only_for_transcription_sessions_on_openai_hosts(
    model, api_base, client_query_params, expected_backend_url
):
    connect: Final = _ConnectThatStopsAfterCapturingTheUrl()
    with patch("websockets.connect", connect):
        await realtime_main._arealtime.__wrapped__(
            model=model,
            websocket=_ClosableGaClientWebSocket(),
            api_base=api_base,
            api_key="fake-key",
            litellm_logging_obj=FakeLogging(),
            query_params=client_query_params,
        )
    assert connect.url == expected_backend_url


async def _vertex_health_check_connect_for(monkeypatch: pytest.MonkeyPatch, api_base: str | None) -> _CapturingConnect:
    async def fake_token_resolver(
        credentials: object, project_id: str | None, custom_llm_provider: str
    ) -> tuple[str, str]:
        return "access-token", project_id or ""

    monkeypatch.setattr(realtime_main, "vertex_access_token_resolver", fake_token_resolver)
    connect: Final = _CapturingConnect()
    with patch("websockets.connect", connect):
        assert await realtime_main._realtime_health_check(
            model="gemini-3.8-live",
            custom_llm_provider="vertex_ai",
            api_key=None,
            api_base=api_base,
            model_params={
                "vertex_project": "proj-1",
                "vertex_credentials": "fake-credentials",
                "vertex_location": "us",
            },
        )
    return connect


@pytest.mark.asyncio
async def test_vertex_health_check_sends_no_tls_argument_to_a_plain_ws_api_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect: Final = await _vertex_health_check_connect_for(monkeypatch, "http://127.0.0.1:8080")
    assert connect.url == "ws://127.0.0.1:8080/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"
    assert connect.kwargs["ssl"] is None


@pytest.mark.asyncio
async def test_vertex_health_check_keeps_tls_for_the_multi_region_host(monkeypatch: pytest.MonkeyPatch) -> None:
    connect: Final = await _vertex_health_check_connect_for(monkeypatch, None)
    assert connect.url == (
        "wss://aiplatform.us.rep.googleapis.com/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"
    )
    assert connect.kwargs["ssl"] is not None


BLOCKED_PHRASE = "XSECRETBLOCKTESTPHRASEX"


class PhraseBlockingGuardrail(CustomGuardrail):
    async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
        for text in inputs.get("texts", []):
            if BLOCKED_PHRASE in text:
                raise ValueError("Content blocked: contains forbidden test phrase.")
        return inputs


def _make_guardrail(event_hook=GuardrailEventHooks.pre_call):
    return PhraseBlockingGuardrail(
        guardrail_name="integration-test-guard",
        event_hook=event_hook,
        default_on=True,
    )


async def _build_streaming(client_events, backend_ws):
    client_ws = MagicMock()
    input_queue: asyncio.Queue = asyncio.Queue()

    async def send_text(data: str):
        client_events.append(json.loads(data))

    client_ws.send_text = send_text
    client_ws.receive_text = input_queue.get

    logging_obj: Final = MagicMock(
        pre_call=MagicMock(),
        async_success_handler=AsyncMock(),
        success_handler=MagicMock(),
        dispatch_success_handlers=AsyncMock(),
        dispatch_failure_handlers=AsyncMock(),
        model_call_details={},
    )

    streaming = RealTimeStreaming(
        websocket=client_ws,
        backend_ws=backend_ws,
        logging_obj=logging_obj,
        request_data={"guardrails": ["integration-test-guard"]},
    )
    return streaming, input_queue


@pytest.mark.asyncio
async def test_voice_transcript_blocked_by_guardrail():
    """
    Simulate a backend-side voice transcription event containing the blocked phrase.
    Guardrail must block it - no response.create sent to OpenAI.
    """
    from websockets.exceptions import ConnectionClosed

    guardrail = _make_guardrail(GuardrailEventHooks.realtime_input_transcription)
    litellm.callbacks = [guardrail]

    client_events: list[dict] = []

    # Build the transcript event that would come from the OpenAI backend
    transcript_event = json.dumps(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": f"This is {BLOCKED_PHRASE} in my voice message",
            "item_id": "item_integ_test",
        }
    ).encode()

    # Mock backend that delivers the transcript then closes
    backend_ws = MagicMock()
    backend_ws.recv = AsyncMock(
        side_effect=[
            transcript_event,
            ConnectionClosed(None, None),
        ]
    )
    backend_ws.send = AsyncMock()

    try:
        streaming, _ = await _build_streaming(client_events, backend_ws)
        await streaming.backend_to_client_send_messages()

        event_types = [e.get("type") for e in client_events]

        # 1. Error event must be sent to client
        error_events = [e for e in client_events if e.get("type") == "error"]
        assert len(error_events) >= 1, f"Expected guardrail error event, got: {event_types}"
        assert error_events[0]["error"]["type"] == "guardrail_violation"

        # 2. Check what was sent to backend.
        #    The guardrail may send response.cancel + conversation.item.create (block msg)
        #    + response.create (to speak the block message). That's acceptable.
        #    What we assert is that a response.cancel was sent (blocking the original).
        sent_to_backend = [
            json.loads(c.args[0]) for c in backend_ws.send.call_args_list if c.args and isinstance(c.args[0], str)
        ]
        response_cancels = [e for e in sent_to_backend if e.get("type") == "response.cancel"]
        assert len(response_cancels) >= 1 or len(sent_to_backend) == 0, (
            f"Guardrail should have sent response.cancel or nothing, got: {sent_to_backend}"
        )

        # Note: The guardrail may or may not send transcript deltas; the error event
        # (assertion #1) is the primary signal that the blocked content was handled.

    finally:
        litellm.callbacks = []


@pytest.fixture
def federated_deployment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, str]:
    token_file: Final = tmp_path / "subject_token.jwt"
    token_file.write_text("subject-token")
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_IDENTITY_PROVIDER_ID",
        "OPENAI_SERVICE_ACCOUNT_ID",
        "OPENAI_IDENTITY_TOKEN_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    for attribute in ("api_key", "openai_key", "api_base"):
        monkeypatch.setattr(litellm, attribute, None)
    _workload_identity_auth.cache_clear()
    return {
        "openai_identity_provider_id": "idp_realtime",
        "openai_service_account_id": "user-realtime",
        "openai_identity_token_file": str(token_file),
    }


def _mock_token_exchange() -> None:
    respx.post("https://auth.openai.com/oauth/token").mock(
        return_value=httpx.Response(200, json={"access_token": "federated-bearer", "expires_in": 3600})
    )


class _ConnectThatStopsAfterCapturingTheAuthorization:
    authorization: str | None = None

    def __call__(
        self, url: str, *, additional_headers: Mapping[str, str], **kwargs: object
    ) -> "_ConnectThatStopsAfterCapturingTheAuthorization":
        self.authorization = additional_headers.get("Authorization")
        return self

    async def __aenter__(self) -> None:
        raise RuntimeError("backend headers captured, nothing to bridge")

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "live_session_start",
    [None, {"type": "session.start", "session": {"model": "gpt-live-1"}}],
    ids=["realtime", "live_session"],
)
@respx.mock
async def test_openai_websocket_sessions_send_the_federated_bearer_for_a_keyless_deployment(
    federated_deployment: dict[str, str], live_session_start: Mapping[str, object] | None
) -> None:
    _mock_token_exchange()
    connect: Final = _ConnectThatStopsAfterCapturingTheAuthorization()
    with patch("websockets.connect", connect):
        await realtime_main._arealtime.__wrapped__(
            model="gpt-realtime",
            custom_llm_provider="openai",
            websocket=MagicMock(close=AsyncMock()),
            litellm_logging_obj=FakeLogging(),
            live_session_start=live_session_start,
            **federated_deployment,
        )
    assert connect.authorization == "Bearer federated-bearer"


class _GatewayThatRecordsTheAuthorization:
    authorization: str | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.authorization = request.headers.get("Authorization")
        return httpx.Response(200, json={"value": "ek_test"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "model"),
    [
        (realtime_main.acreate_realtime_client_secret, "gpt-realtime"),
        (realtime_main.acreate_realtime_transcription_session, "gpt-realtime"),
        (realtime_main.acreate_realtime_translation_client_secret, "gpt-realtime-translate"),
    ],
    ids=["client_secrets", "transcription_sessions", "translation_client_secrets"],
)
@respx.mock
async def test_openai_realtime_http_routes_send_the_federated_bearer_for_a_keyless_deployment(
    federated_deployment: dict[str, str], route: Callable[..., Awaitable[object]], model: str
) -> None:
    _mock_token_exchange()
    gateway: Final = _GatewayThatRecordsTheAuthorization()
    await route.__wrapped__(
        model=model,
        custom_llm_provider="openai",
        litellm_logging_obj=Logging(
            model=model,
            messages=[],
            stream=False,
            call_type="acreate_realtime_client_secret",
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            litellm_call_id="test-federated-realtime",
            function_id="test-federated-realtime",
        ),
        client=AsyncHTTPHandler(transport=httpx.MockTransport(gateway)),
        **federated_deployment,
    )
    assert gateway.authorization == "Bearer federated-bearer"


@pytest.mark.asyncio
@pytest.mark.parametrize("flow", ("client_secret", "calls"))
async def test_translation_http_routes_preserve_explicit_azure_provider(flow: str) -> None:
    expected_path: Final = "/openai/v1/realtime/translations/" + {
        "client_secret": "client_secrets",
        "calls": "calls",
    }[flow]

    def send_response(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "azure-deployment.example"
        assert request.url.path == expected_path
        return httpx.Response(200, json={"value": "ek-test", "routed_path": request.url.path})

    params: Final[Mapping[str, object]] = {
        "model": "translation-deployment",
        "custom_llm_provider": "azure",
        "api_base": "https://azure-deployment.example/openai/v1/",
        "api_key": "test-key",
        "client": AsyncHTTPHandler(transport=httpx.MockTransport(send_response)),
        "litellm_logging_obj": Logging(
            model="translation-deployment",
            messages=[],
            stream=False,
            call_type="acreate_realtime_translation_client_secret",
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            litellm_call_id="test-translation-routing",
            function_id="test-translation-routing",
        ),
    }
    response: Final = await (
        realtime_main.acreate_realtime_translation_client_secret.__wrapped__(**params)
        if flow == "client_secret"
        else realtime_main.arealtime_translation_calls.__wrapped__(
            openai_ephemeral_key="ek-test", sdp_body=b"v=0", **params
        )
    )

    assert response.status_code == 200
    assert response.json() == {"value": "ek-test", "routed_path": expected_path}


@pytest.mark.asyncio
@respx.mock
async def test_openai_realtime_health_check_sends_the_federated_bearer_for_a_keyless_deployment(
    federated_deployment: dict[str, str],
) -> None:
    _mock_token_exchange()
    connect: Final = _CapturingConnect()
    with patch("websockets.connect", connect):
        assert await realtime_main.realtime_health_check(
            model="gpt-realtime",
            custom_llm_provider="openai",
            api_key=None,
            model_params={"model": "openai/gpt-realtime", **federated_deployment},
        )
    assert connect.kwargs["additional_headers"] == {"Authorization": "Bearer federated-bearer"}


def _realtime_http_route_call(
    route: str,
    logging_obj: Logging,
    custom_llm_provider: str,
    api_base: str,
    client: AsyncHTTPHandler | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Awaitable[httpx.Response]:
    if route == "client_secrets":
        return realtime_main.acreate_realtime_client_secret.__wrapped__(
            model="gpt-realtime",
            custom_llm_provider=custom_llm_provider,
            api_key="sk-test",
            api_base=api_base,
            client=client,
            litellm_logging_obj=logging_obj,
        )
    if route == "translation_client_secrets":
        return realtime_main.acreate_realtime_translation_client_secret.__wrapped__(
            model="gpt-realtime-translate",
            custom_llm_provider=custom_llm_provider,
            api_key="sk-test",
            api_base=api_base,
            client=client,
            litellm_logging_obj=logging_obj,
            metadata=metadata,
        )
    if route == "calls":
        return realtime_main.arealtime_calls.__wrapped__(
            openai_ephemeral_key="ek_test",
            sdp_body=b"v=0",
            model="gpt-realtime",
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            client=client,
            litellm_logging_obj=logging_obj,
        )
    return realtime_main.arealtime_translation_calls.__wrapped__(
        openai_ephemeral_key="ek_test",
        sdp_body=b"v=0",
        model="gpt-realtime-translate",
        custom_llm_provider=custom_llm_provider,
        api_base=api_base,
        client=client,
        litellm_logging_obj=logging_obj,
        metadata=metadata,
    )


def _realtime_http_logging(route: str) -> Logging:
    return Logging(
        model="gpt-realtime",
        messages=[],
        stream=False,
        call_type="acreate_realtime_client_secret",
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        litellm_call_id=f"test-realtime-{route}",
        function_id=f"test-realtime-{route}",
    )


_OPENAI_REALTIME_HTTP_ROUTE_PATHS: Final = {
    "client_secrets": "/v1/realtime/client_secrets",
    "translation_client_secrets": "/v1/realtime/translations/client_secrets",
    "calls": "/v1/realtime/calls",
    "translation_calls": "/v1/realtime/translations/calls",
}


def _validity(builder: x509.CertificateBuilder) -> x509.CertificateBuilder:
    now: Final = datetime.now(timezone.utc)
    return (
        builder.serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(hours=1))
    )


def _write_private_ca_and_loopback_certificate(directory: Path) -> tuple[Path, Path, Path]:
    ca_key: Final = ec.generate_private_key(ec.SECP256R1())
    server_key: Final = ec.generate_private_key(ec.SECP256R1())
    ca_name: Final = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "litellm realtime test CA")])
    ca_certificate: Final = (
        _validity(x509.CertificateBuilder())
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    server_certificate: Final = (
        _validity(x509.CertificateBuilder())
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")]))
        .issuer_name(ca_name)
        .public_key(server_key.public_key())
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    ca_path: Final = directory / "ca.pem"
    certificate_path: Final = directory / "server.pem"
    key_path: Final = directory / "server.key"
    ca_path.write_bytes(ca_certificate.public_bytes(serialization.Encoding.PEM))
    certificate_path.write_bytes(server_certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        server_key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
    )
    return ca_path, certificate_path, key_path


class _EchoPathHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        body: Final = json.dumps({"value": "ek_test", "path": self.path}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def private_ca_upstream(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    ca_path, certificate_path, key_path = _write_private_ca_and_loopback_certificate(tmp_path)
    tls: Final = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(certificate_path, key_path)
    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), _EchoPathHandler)
    server.socket = tls.wrap_socket(server.socket, server_side=True)
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"https://127.0.0.1:{server.server_address[1]}", ca_path
    server.shutdown()
    server.server_close()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", tuple(_OPENAI_REALTIME_HTTP_ROUTE_PATHS))
async def test_openai_realtime_http_routes_trust_the_ca_bundle_set_in_litellm_ssl_verify(
    route: str, private_ca_upstream: tuple[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    api_base, ca_path = private_ca_upstream
    monkeypatch.setattr(litellm, "ssl_verify", str(ca_path))

    response: Final = await _realtime_http_route_call(
        route, _realtime_http_logging(route), custom_llm_provider="openai", api_base=api_base
    )

    assert response.status_code == 200
    assert response.json() == {"value": "ek_test", "path": _OPENAI_REALTIME_HTTP_ROUTE_PATHS[route]}


@pytest.mark.asyncio
@pytest.mark.parametrize("route", tuple(_OPENAI_REALTIME_HTTP_ROUTE_PATHS))
async def test_openai_realtime_http_routes_surface_the_upstream_error_body_verbatim(route: str) -> None:
    upstream_error: Final = {
        "error": {
            "message": "Invalid model",
            "type": "invalid_request_error",
            "param": "session.model",
            "code": "invalid_value",
        }
    }

    with pytest.raises(BaseLLMException) as raised:
        await _realtime_http_route_call(
            route,
            _realtime_http_logging(route),
            custom_llm_provider="openai",
            api_base="https://upstream.example",
            client=AsyncHTTPHandler(
                transport=httpx.MockTransport(lambda request: httpx.Response(400, json=upstream_error))
            ),
        )

    assert raised.value.status_code == 400
    assert json.loads(raised.value.message) == upstream_error


class _GatewayThatRecordsTheSession:
    session: Mapping[str, object] | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.session = json.loads(request.content)["session"]
        return httpx.Response(200, json={"value": "ek_test"})


@pytest.mark.asyncio
async def test_client_secret_keeps_the_realtime_session_model_when_input_transcription_names_another() -> None:
    gateway: Final = _GatewayThatRecordsTheSession()
    logging_obj: Final = _realtime_http_logging("client_secrets")

    await realtime_main.acreate_realtime_client_secret.__wrapped__(
        model="gpt-realtime",
        session={
            "type": "realtime",
            "model": "gpt-realtime",
            "audio": {"input": {"transcription": {"model": "gpt-realtime-whisper"}}},
        },
        custom_llm_provider="openai",
        api_key="sk-test",
        api_base="https://upstream.example",
        litellm_logging_obj=logging_obj,
        client=AsyncHTTPHandler(transport=httpx.MockTransport(gateway)),
    )

    assert gateway.session is not None
    assert gateway.session["model"] == "gpt-realtime"
    assert gateway.session["audio"] == {"input": {"transcription": {"model": "gpt-realtime-whisper"}}}
    assert logging_obj.model == "gpt-realtime"


@pytest.mark.asyncio
async def test_client_secret_session_model_wins_over_a_provider_qualified_top_level_model(
    local_model_cost_map: None,
) -> None:
    gateway: Final = _GatewayThatRecordsTheSession()
    logging_obj: Final = _realtime_http_logging("client_secrets")

    await realtime_main.acreate_realtime_client_secret.__wrapped__(
        model="openai/gpt-realtime-2.1",
        session={"type": "realtime", "model": "gpt-realtime-2.1-mini"},
        api_key="sk-test",
        api_base="https://upstream.example",
        litellm_logging_obj=logging_obj,
        client=AsyncHTTPHandler(transport=httpx.MockTransport(gateway)),
    )

    assert gateway.session is not None
    assert gateway.session["model"] == "gpt-realtime-2.1-mini"
    assert logging_obj.model == "gpt-realtime-2.1-mini"


class _UpstreamThatMustNotBeCalled:
    called: bool = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.called = True
        return httpx.Response(200, json={"value": "ek_test"})


class _RecordingClientWebSocket:
    def __init__(self) -> None:
        self.scope: Final = {"headers": ()}
        self.sent: tuple[str, ...] = ()
        self.close_code: int | None = None

    async def send_text(self, data: str) -> None:
        self.sent = (*self.sent, data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.close_code = code


def _block_every_request_with_a_guardrail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        litellm,
        "callbacks",
        [
            CustomGuardrail(
                guardrail_name="translation-blocker", event_hook=GuardrailEventHooks.pre_call, default_on=True
            )
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("translation_client_secrets", "translation_calls"))
async def test_translation_http_routes_refuse_a_guarded_request_before_any_upstream_call(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _block_every_request_with_a_guardrail(monkeypatch)
    upstream: Final = _UpstreamThatMustNotBeCalled()

    with pytest.raises(litellm.BadRequestError, match=TRANSLATION_GUARDRAILS_UNSUPPORTED_MESSAGE) as raised:
        await _realtime_http_route_call(
            route,
            _realtime_http_logging(route),
            custom_llm_provider="openai",
            api_base="https://upstream.example",
            client=AsyncHTTPHandler(transport=httpx.MockTransport(upstream)),
        )

    assert raised.value.status_code == 400
    assert upstream.called is False


_DEFAULT_ON_GUARDRAIL_OPT_OUTS: Final = (
    pytest.param({"user_api_key_metadata": {"disable_global_guardrails": True}}, id="key-disables-global"),
    pytest.param(
        {"user_api_key_team_metadata": {"opted_out_global_guardrails": ["translation-blocker"]}}, id="team-opts-out"
    ),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "path"),
    (("translation_client_secrets", "client_secrets"), ("translation_calls", "calls")),
)
@pytest.mark.parametrize("key_metadata", _DEFAULT_ON_GUARDRAIL_OPT_OUTS)
async def test_translation_http_routes_serve_a_key_opted_out_of_the_default_on_guardrail(
    route: str, path: str, key_metadata: Mapping[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    _block_every_request_with_a_guardrail(monkeypatch)
    gateway: Final = _GatewayThatRecordsTheUrl()

    await _realtime_http_route_call(
        route,
        _realtime_http_logging(route),
        custom_llm_provider="openai",
        api_base="https://upstream.example",
        client=AsyncHTTPHandler(transport=httpx.MockTransport(gateway)),
        metadata=key_metadata,
    )

    assert gateway.url == f"https://upstream.example/v1/realtime/translations/{path}"


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ("translation_client_secrets", "translation_calls"))
async def test_translation_http_routes_refuse_a_provider_other_than_openai_and_azure_before_any_upstream_call(
    route: str,
) -> None:
    upstream: Final = _UpstreamThatMustNotBeCalled()

    with pytest.raises(litellm.BadRequestError, match="supports only the azure, openai providers, got xai") as raised:
        await _realtime_http_route_call(
            route,
            _realtime_http_logging(route),
            custom_llm_provider="xai",
            api_base="https://upstream.example",
            client=AsyncHTTPHandler(transport=httpx.MockTransport(upstream)),
        )

    assert raised.value.status_code == 400
    assert upstream.called is False


@pytest.mark.asyncio
async def test_translation_websocket_closes_1008_for_a_guarded_request_before_the_upstream_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_every_request_with_a_guardrail(monkeypatch)
    websocket: Final = _RecordingClientWebSocket()
    connect: Final = _CapturingConnect()

    with patch("websockets.connect", connect):
        await realtime_main._arealtime.__wrapped__(
            model="openai/gpt-realtime-translate",
            websocket=websocket,
            api_key="fake-key",
            realtime_mode="translation",
            litellm_logging_obj=_realtime_websocket_logging(),
        )

    assert connect.url is None
    assert websocket.close_code == 1008
    assert [json.loads(event)["error"]["message"] for event in websocket.sent] == [
        TRANSLATION_GUARDRAILS_UNSUPPORTED_MESSAGE
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("key_metadata", _DEFAULT_ON_GUARDRAIL_OPT_OUTS)
async def test_translation_websocket_opens_for_a_key_opted_out_of_the_default_on_guardrail(
    key_metadata: Mapping[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    _block_every_request_with_a_guardrail(monkeypatch)
    websocket: Final = _RecordingClientWebSocket()
    connect: Final = _CapturingConnect()

    with patch("websockets.connect", connect):
        await realtime_main._arealtime.__wrapped__(
            model="openai/gpt-realtime-translate",
            websocket=websocket,
            api_key="fake-key",
            realtime_mode="translation",
            litellm_logging_obj=FakeLogging(),
            metadata=key_metadata,
        )

    assert connect.url == "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"
    assert all(TRANSLATION_GUARDRAILS_UNSUPPORTED_MESSAGE not in event for event in websocket.sent)


class _FailureRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.failures: tuple[object, ...] = ()

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.failures = (*self.failures, kwargs.get("exception"))


@pytest.mark.asyncio
async def test_translation_websocket_refusal_logs_the_session_as_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_every_request_with_a_guardrail(monkeypatch)
    recorder: Final = _FailureRecorder()
    logging_obj: Final = _realtime_websocket_logging(recorder)

    with patch("websockets.connect", _CapturingConnect()):
        await realtime_main._arealtime.__wrapped__(
            model="openai/gpt-realtime-translate",
            websocket=_RecordingClientWebSocket(),
            api_key="fake-key",
            realtime_mode="translation",
            litellm_logging_obj=logging_obj,
        )

    assert [(type(failure), getattr(failure, "status_code", None)) for failure in recorder.failures] == [
        (litellm.BadRequestError, 400)
    ]
    assert TRANSLATION_GUARDRAILS_UNSUPPORTED_MESSAGE in str(recorder.failures[0])
    assert logging_obj.model_call_details[REALTIME_SESSION_FAILURE_LOGGED_KEY] is True


class _NativeLifecycleGuardrail(CustomGuardrail):
    use_native_lifecycle_hooks: ClassVar[bool] = True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("guardrail_class", "event_hook"),
    ((_NativeLifecycleGuardrail, GuardrailEventHooks.pre_call), (CustomGuardrail, GuardrailEventHooks.post_call)),
)
async def test_translation_websocket_opens_under_a_default_on_guardrail_the_realtime_runner_skips(
    guardrail_class: type[CustomGuardrail], event_hook: GuardrailEventHooks, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        litellm,
        "callbacks",
        [guardrail_class(guardrail_name="runner-skipped", event_hook=event_hook, default_on=True)],
    )
    websocket: Final = _RecordingClientWebSocket()
    connect: Final = _CapturingConnect()

    with patch("websockets.connect", connect):
        await realtime_main._arealtime.__wrapped__(
            model="openai/gpt-realtime-translate",
            websocket=websocket,
            api_key="fake-key",
            realtime_mode="translation",
            litellm_logging_obj=FakeLogging(),
        )

    assert connect.url == "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"
    assert all(TRANSLATION_GUARDRAILS_UNSUPPORTED_MESSAGE not in event for event in websocket.sent)


@pytest.mark.asyncio
async def test_translation_websocket_closes_1008_for_a_provider_other_than_openai_and_azure() -> None:
    websocket: Final = _RecordingClientWebSocket()
    connect: Final = _CapturingConnect()

    with patch("websockets.connect", connect):
        await realtime_main._arealtime.__wrapped__(
            model="xai/grok-voice",
            websocket=websocket,
            api_key="fake-key",
            realtime_mode="translation",
            litellm_logging_obj=_realtime_websocket_logging(),
        )

    assert connect.url is None
    assert websocket.close_code == 1008
    assert [json.loads(event)["error"]["message"] for event in websocket.sent] == [
        "Realtime translation supports only the azure, azure_ai, openai providers, got xai"
    ]
