import threading
from collections.abc import Generator, Iterator
from typing import Final

import fakeredis
import pytest

import litellm
from tests._support.recording_server import RecordingServer, recording_service


@pytest.fixture
def recording_server() -> Generator[RecordingServer]:
    with recording_service() as server:
        yield server


@pytest.fixture
def local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(
        litellm,
        "model_cost",
        litellm.get_model_cost_map(url=""),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # get_model_cost_map is untyped
    )
    litellm.get_model_info.cache_clear()  # pyright: ignore[reportAny, reportFunctionMemberAccess]  # lru_cache helpers are not visible to the type checker
    yield
    litellm.get_model_info.cache_clear()  # pyright: ignore[reportAny, reportFunctionMemberAccess]  # lru_cache helpers are not visible to the type checker


@pytest.fixture
def isolated_azure_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "AZURE_AI_API_KEY",
        "AZURE_AI_API_BASE",
        "AZURE_AD_TOKEN",
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET",
        "AZURE_USERNAME",
        "AZURE_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "enable_azure_ad_token_refresh", False)


@pytest.fixture
def redis_url() -> Generator[str]:
    server: Final = fakeredis.TcpFakeServer(("127.0.0.1", 0), server_type="redis")
    worker: Final = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"redis://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
