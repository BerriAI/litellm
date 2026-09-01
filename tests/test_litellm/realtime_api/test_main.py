import asyncio
import time
from types import TracebackType
from unittest.mock import MagicMock, patch


import pytest

import litellm
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
    def __init__(self) -> None:
        self.url: str | None = None

    def __call__(self, url: str, **kwargs: object) -> "_CapturingConnect":
        self.url = url
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
async def test_azure_health_check_keeps_beta_path_for_speech_model():
    connect = _CapturingConnect()
    with patch("websockets.connect", connect):
        assert await realtime_main._realtime_health_check(
            model="gpt-4o-realtime-preview",
            custom_llm_provider="azure",
            api_key="fake-key",
            api_base="https://my-endpoint.openai.azure.com",
            api_version="2024-10-01-preview",
        )
    assert connect.url == (
        "wss://my-endpoint.openai.azure.com/openai/realtime"
        "?api-version=2024-10-01-preview&deployment=gpt-4o-realtime-preview"
    )


@pytest.mark.asyncio
async def test_azure_health_check_honors_deployment_realtime_protocol():
    connect = _CapturingConnect()
    with patch("websockets.connect", connect):
        assert await realtime_main._realtime_health_check(
            model="gpt-4o-realtime-preview",
            custom_llm_provider="azure",
            api_key="fake-key",
            api_base="https://my-endpoint.openai.azure.com",
            api_version="2024-10-01-preview",
            model_params={"realtime_protocol": "GA"},
        )
    assert connect.url == "wss://my-endpoint.openai.azure.com/openai/v1/realtime?model=gpt-4o-realtime-preview"
