from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import pytest
from cli_harness.rate_limiter import get_default_limiter
from codex.harness import CodexHarness, codex_harness
from codex.setups import Deployment, ModelSetup
from e2e_config import PROXY_BASE_URL, unique_marker
from lifecycle import ResourceManager
from proxy_client import ProxyClient


@dataclass(frozen=True, slots=True)
class _Client:
    proxy: ProxyClient


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> _Client:
    return _Client(proxy)


@pytest.fixture
def setup(deployment: Deployment, proxy: ProxyClient, resources: ResourceManager) -> ModelSetup:
    model: Final = f"e2e-codex-{unique_marker()}"
    model_id: Final = proxy.create_model(model, deployment.litellm_params, provider_live=True)
    resources.defer(lambda: proxy.delete_model(model_id))
    return ModelSetup(model=model, rate_bucket=deployment.rate_bucket)


@pytest.fixture
def codex(setup: ModelSetup, resources: ResourceManager) -> Iterator[CodexHarness]:
    with codex_harness(
        gateway=PROXY_BASE_URL,
        api_key=resources.key(models=[setup.model], user_id=None),
        rate_bucket=setup.rate_bucket,
        limiter=get_default_limiter(),
    ) as harness:
        yield harness
