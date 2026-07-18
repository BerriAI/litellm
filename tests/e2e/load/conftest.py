from __future__ import annotations

from collections.abc import Iterator

import pytest

from e2e_config import unique_marker
from load_client import LoadClient, build_client
from load_constants import LOAD_MOCK_PARAMS
from lifecycle import ResourceManager
from models import KeyGenerateBody
from proxy_client import ProxyClient


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> LoadClient:
    return build_client(proxy)


@pytest.fixture(scope="session")
def load_model(client: LoadClient) -> Iterator[str]:
    """Register a fresh mock deployment for this session and delete it after.

    A fixed name like ``load-mock`` is unsafe on a shared stage proxy: a prior
    run (or a hand-registered row) can leave a deployment without mock_response,
    so Locust would hit real OpenAI with an invalid model and fail ~all requests
    while the fixture skipped /model/new because the name was already listed.
    """
    model_name = f"load-mock-{unique_marker()}"
    model_id = client.proxy.create_model(model_name, LOAD_MOCK_PARAMS)
    try:
        yield model_name
    finally:
        client.proxy.delete_model(model_id)


@pytest.fixture
def load_key(resources: ResourceManager, client: LoadClient, load_model: str) -> str:
    key = client.proxy.generate_key(
        KeyGenerateBody(models=[load_model], user_id=f"e2e-load-{unique_marker()}")
    )
    resources.defer(lambda: client.proxy.delete_key(key))
    return key
