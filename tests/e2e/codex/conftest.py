from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Final

import pytest
from cli_harness.rate_limiter import get_default_limiter
from codex.harness import CodexHarness, codex_harness
from codex.setups import Deployment, ModelSetup
from e2e_config import PROXY_BASE_URL, unique_marker
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient


@dataclass(frozen=True, slots=True)
class _Client:
    proxy: ProxyClient


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> _Client:
    return _Client(proxy)


@pytest.fixture
def register_model(proxy: ProxyClient, resources: ResourceManager) -> Callable[[LiteLLMParamsBody], str]:
    def register(litellm_params: LiteLLMParamsBody) -> str:
        model: Final = f"e2e-codex-{unique_marker()}"
        model_id: Final = proxy.create_model(model, litellm_params, provider_live=True)
        resources.defer(lambda: proxy.delete_model(model_id))
        return model

    return register


@pytest.fixture
def setup(deployment: Deployment, register_model: Callable[[LiteLLMParamsBody], str]) -> ModelSetup:
    return ModelSetup(model=register_model(deployment.litellm_params), rate_bucket=deployment.rate_bucket)


@pytest.fixture
def key_models(setup: ModelSetup) -> tuple[str, ...]:
    return (setup.model,)


@pytest.fixture
def codex(setup: ModelSetup, key_models: tuple[str, ...], resources: ResourceManager) -> Iterator[CodexHarness]:
    with codex_harness(
        gateway=PROXY_BASE_URL,
        api_key=resources.key(models=list(key_models), user_id=None),
        rate_bucket=setup.rate_bucket,
        limiter=get_default_limiter(),
    ) as harness:
        yield harness
