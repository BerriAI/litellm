import asyncio
import time
from unittest.mock import MagicMock


import pytest

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


def _run_arealtime(monkeypatch, provider, **kwargs):
    captured = {}

    async def mock_async_realtime(**call_kwargs):
        captured.update(call_kwargs)

    def mock_get_llm_provider(model, api_base, api_key):
        return model, provider, None, None

    monkeypatch.setattr(realtime_main, "get_llm_provider", mock_get_llm_provider)
    handler = realtime_main.openai_realtime if provider == "openai" else realtime_main.azure_realtime
    monkeypatch.setattr(handler, "async_realtime", mock_async_realtime)

    asyncio.run(
        realtime_main._arealtime.__wrapped__(
            model="gpt-4o-realtime-preview",
            websocket=object(),
            litellm_logging_obj=FakeLogging(),
            **kwargs,
        )
    )
    return captured


def test_openai_realtime_uses_explicit_api_key(monkeypatch):
    """A key resolved from litellm_credential_name arrives as the explicit
    api_key argument and must not be discarded."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(realtime_main.litellm, "api_key", None)
    monkeypatch.setattr(realtime_main.litellm, "openai_key", None)

    captured = _run_arealtime(monkeypatch, "openai", api_key="sk-from-credential", api_base="http://localhost:8799")
    assert captured["api_key"] == "sk-from-credential"
    assert captured["api_base"] == "http://localhost:8799"


def test_azure_realtime_uses_explicit_api_key(monkeypatch):
    monkeypatch.delenv("AZURE_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_API_BASE", raising=False)
    monkeypatch.setattr(realtime_main.litellm, "api_key", None)
    monkeypatch.setattr(realtime_main.litellm, "openai_key", None)

    captured = _run_arealtime(
        monkeypatch, "azure", api_key="azure-from-credential", api_base="https://my-azure.openai.azure.com"
    )
    assert captured["api_key"] == "azure-from-credential"