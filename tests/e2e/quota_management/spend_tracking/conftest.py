"""Spend-tracking suite's `client` fixture and driver-model registration.

The shared lifecycle (resources/scoped_key), proxy liveness skip, and e2e marker
live in the parent tests/e2e/conftest.py. SpendClient exposes the shared Gateway
(GatewayProvider), so the `resources` fixture cleans up keys and customers this
suite creates.

The suite drives real calls through three deployments. On the stage gateway they
are baked into the proxy config; on a local dev proxy they usually are not, so
`driver_models` registers whichever are missing via /model/new and deletes only
the ones it created, never a config-baked deployment. Each registration carries
the provider key from the test runner's env when set (so a local proxy whose
container env lacks the key still works); otherwise it falls back to an
os.environ reference resolved from the proxy's own env, the stage convention.
"""

import os
from typing import Iterator

import pytest

from e2e_config import CHEAP_ANTHROPIC_MODEL
from models import LiteLLMParamsBody
from spend_e2e_client import SpendClient, build_client


def _driver_params(provider_model: str, env_var: str) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=provider_model,
        api_key=os.environ.get(env_var) or f"os.environ/{env_var}",
    )


DRIVER_MODELS: tuple[tuple[str, str, str], ...] = (
    ("gemini-2.5-flash", "gemini/gemini-2.5-flash", "GEMINI_API_KEY"),
    (CHEAP_ANTHROPIC_MODEL, f"anthropic/{CHEAP_ANTHROPIC_MODEL}", "ANTHROPIC_API_KEY"),
    ("openai-text-embedding-3-small", "openai/text-embedding-3-small", "OPENAI_API_KEY"),
)


@pytest.fixture(scope="session")
def client() -> SpendClient:
    return build_client()


@pytest.fixture(scope="session", autouse=True)
def driver_models(client: SpendClient) -> Iterator[None]:
    existing = frozenset(entry.model_name for entry in client.gateway.model_info())
    created = tuple(
        client.gateway.create_model(name, _driver_params(provider_model, env_var))
        for name, provider_model, env_var in DRIVER_MODELS
        if name not in existing
    )
    yield
    for model_id in created:
        client.gateway.delete_model(model_id)
