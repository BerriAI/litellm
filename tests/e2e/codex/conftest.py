from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
from cli_harness.rate_limiter import get_default_limiter
from codex.compat_results import pytest_runtest_makereport as pytest_runtest_makereport
from codex.harness import CodexHarness, codex_harness
from codex.setups import Deployment, ModelSetup
from e2e_config import PROXY_BASE_URL, unique_marker
from lifecycle import ResourceManager
from models import KeyGenerateBody, ModelInfoBody, ModelNewBody, TeamNewBody
from proxy_client import ProxyClient
from pydantic import TypeAdapter

CODEX_DIR: Final = Path(__file__).resolve().parent
ASSERTION_FAILURE: Final = "^AssertionError: "
RERUN_PATTERNS: Final = TypeAdapter[list[str] | None](list[str] | None)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    configured: Final = RERUN_PATTERNS.validate_python(config.getoption("only_rerun", default=None))
    rerun_on: Final = (*(configured or ()), ASSERTION_FAILURE)
    for item in items:
        if CODEX_DIR in item.path.parents:
            item.add_marker(pytest.mark.flaky(only_rerun=rerun_on))


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
    return ModelSetup(
        model=model, rate_bucket=deployment.rate_bucket, api_key=resources.key(models=[model], user_id=None)
    )


@pytest.fixture
def real_name_setup(deployment: Deployment, proxy: ProxyClient, resources: ResourceManager) -> ModelSetup:
    model: Final = deployment.real_model_name
    team_id: Final = proxy.create_team(TeamNewBody(team_alias=f"e2e-codex-{unique_marker()}", models=[model]))
    resources.defer(lambda: proxy.delete_team(team_id))
    api_key: Final = proxy.generate_key(KeyGenerateBody(team_id=team_id, models=[model]))
    resources.defer(lambda: proxy.delete_key(api_key))
    model_id: Final = proxy.register_model(
        ModelNewBody(
            model_name=model, litellm_params=deployment.litellm_params, model_info=ModelInfoBody(team_id=team_id)
        ),
        listed_for=api_key,
        provider_live=True,
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return ModelSetup(model=model, rate_bucket=deployment.rate_bucket, api_key=api_key)


@pytest.fixture
def codex(setup: ModelSetup) -> Iterator[CodexHarness]:
    yield from _codex_for(setup)


@pytest.fixture
def real_name_codex(real_name_setup: ModelSetup) -> Iterator[CodexHarness]:
    yield from _codex_for(real_name_setup)


def _codex_for(setup: ModelSetup) -> Iterator[CodexHarness]:
    with codex_harness(
        gateway=PROXY_BASE_URL, api_key=setup.api_key, rate_bucket=setup.rate_bucket, limiter=get_default_limiter()
    ) as harness:
        yield harness
