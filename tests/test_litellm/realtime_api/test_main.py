import asyncio
import json
import time
from types import TracebackType
from typing import Final
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.realtime_api import main as realtime_main
from litellm.realtime_api.main import _with_resolved_session_model


class FakeLogging:
    def update_from_kwargs(self, **kwargs):
        pass


@pytest.mark.parametrize("provider", [litellm.LlmProviders.XAI, litellm.LlmProviders.OPENAI, litellm.LlmProviders.GEMINI])
def test_realtime_handler_factory_does_not_read_headers_without_a_handler(provider):
    from litellm.types.router import GenericLiteLLMParams

    read_headers = MagicMock(side_effect=AssertionError("Headers must not be read"))
    assert realtime_main.ProviderConfigManager.get_provider_realtime_handler(
        provider, GenericLiteLLMParams(), read_headers
    ) is None
    read_headers.assert_not_called()


def test_realtime_handler_factory_passes_actual_chatgpt_headers(tmp_path, monkeypatch):
    from litellm.llms.chatgpt.realtime import ChatGPTRealtime
    from litellm.types.router import GenericLiteLLMParams

    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    monkeypatch.setenv("CHATGPT_AUTH_FILE", "auth.json")
    (tmp_path / "auth.json").write_text(
        json.dumps({"access_token": "factory-test-token", "account_id": "factory-account", "expires_at": time.time() + 3600})
    )
    params = GenericLiteLLMParams(litellm_session_id="factory-session")
    headers = {"openai-alpha": "quicksilver=v2"}
    extra_headers = {"x-gateway-route": "required"}
    read_headers = MagicMock(return_value=headers)
    result = realtime_main.ProviderConfigManager.get_provider_realtime_handler(
        litellm.LlmProviders.CHATGPT, params, read_headers, extra_headers
    )
    assert isinstance(result, ChatGPTRealtime)
    read_headers.assert_called_once_with()
    outgoing_headers = result._get_additional_headers("unused")
    assert outgoing_headers["openai-alpha"] == headers["openai-alpha"]
    assert outgoing_headers["x-gateway-route"] == extra_headers["x-gateway-route"]
    assert outgoing_headers["session_id"] == "factory-session"
    assert outgoing_headers["Authorization"] == "Bearer factory-test-token"


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


@pytest.mark.parametrize("is_call", [False, True])
@pytest.mark.parametrize("provider", ["chatgpt", "openai", "azure"])
def test_realtime_http_provider_controls_dynamic_base_precedence(provider, is_call, monkeypatch):
    from litellm.types.router import GenericLiteLLMParams

    monkeypatch.delenv("CHATGPT_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_CHATGPT_API_BASE", raising=False)
    config, base, key = realtime_main._get_realtime_http_provider_config(
        custom_llm_provider=provider,
        dynamic_api_base="https://dynamic.example/v1",
        dynamic_api_key="dynamic-key",
        litellm_params=GenericLiteLLMParams(api_base="https://configured.example/v1"),
        is_call=is_call,
    )
    expected_base = "https://configured.example/v1" if provider == "chatgpt" else "https://dynamic.example/v1"
    assert base == expected_base
    assert key == ("chatgpt-oauth" if provider == "chatgpt" else "dynamic-key")
    assert config is not None
    if provider == "chatgpt":
        assert config.get_realtime_calls_url(base, "gpt-realtime-1.5") == expected_base + "/realtime/calls"
    else:
        assert config.get_realtime_calls_extra_headers({"x-gateway-route": "required"}) == {
            "x-gateway-route": "required"
        }
