from __future__ import annotations

from collections.abc import Iterator

import pytest
from requests import RequestException

from proxy_client import ProxyClient
from e2e_http import NoBody, Success
from load_client import LoadClient, build_client
from load_constants import LOAD_MODEL
from models import KeyGenerateBody, LiteLLMParamsBody, ModelsListResponse
from lifecycle import ResourceManager

LOAD_MODEL_PARAMS = LiteLLMParamsBody(
    model="openai/load-mock",
    mock_response="This is a mock response for the throughput load test.",
)


@pytest.fixture(scope="session")
def client() -> LoadClient:
    return build_client()


def _model_is_servable(gateway: ProxyClient, model_name: str) -> bool:
    result = gateway.transport.get(
        "/v1/models",
        headers=gateway.transport.master,
        params=NoBody(),
        response_type=ModelsListResponse,
    )
    return isinstance(result, Success) and any(entry.id == model_name for entry in result.data.data)


@pytest.fixture(scope="session", autouse=True)
def _ensure_load_model(  # pyright: ignore[reportUnusedFunction]  # pytest autouse session fixture, wired by name
    client: LoadClient,
) -> Iterator[None]:
    gateway = client.gateway
    if _model_is_servable(gateway, LOAD_MODEL):
        yield
        return

    try:
        model_id = gateway.create_model(LOAD_MODEL, LOAD_MODEL_PARAMS)
    except (AssertionError, RequestException) as exc:
        if _model_is_servable(gateway, LOAD_MODEL):
            yield
            return
        raise AssertionError(
            f"failed to register {LOAD_MODEL!r} for the throughput load test "
            f"(not listed on the data plane and /model/new failed): {exc}"
        ) from exc

    try:
        yield
    finally:
        gateway.delete_model(model_id)


@pytest.fixture
def load_key(resources: ResourceManager, client: LoadClient) -> str:
    key = client.gateway.generate_key(KeyGenerateBody(models=[LOAD_MODEL], user_id="e2e-load"))
    resources.defer(lambda: client.gateway.delete_key(key))
    return key
