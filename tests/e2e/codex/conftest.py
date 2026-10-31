from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest
from cli_harness.rate_limiter import get_default_limiter
from codex.harness import CodexHarness, codex_harness
from codex.setups import Deployment, ModelSetup
from e2e_config import PROXY_BASE_URL, unique_marker
from models import KeyGenerateBody
from proxy_client import ProxyClient


@pytest.fixture
def setup(deployment: Deployment, proxy: ProxyClient) -> Iterator[ModelSetup]:
    model: Final = f"e2e-codex-{unique_marker()}"
    model_id: Final = proxy.create_model(model, deployment.litellm_params, provider_live=True)
    yield ModelSetup(model=model, rate_bucket=deployment.rate_bucket)
    proxy.delete_model(model_id)


@pytest.fixture
def codex(setup: ModelSetup, proxy: ProxyClient) -> Iterator[CodexHarness]:
    key: Final = proxy.generate_key(KeyGenerateBody(models=[setup.model]))
    with codex_harness(
        gateway=PROXY_BASE_URL, api_key=key, rate_bucket=setup.rate_bucket, limiter=get_default_limiter()
    ) as harness:
        yield harness
    proxy.delete_key(key)
