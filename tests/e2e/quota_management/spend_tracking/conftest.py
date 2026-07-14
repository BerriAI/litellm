"""Spend-tracking suite's `client` fixture and driver-model registration.

The shared lifecycle (resources/scoped_key), proxy liveness skip, and e2e marker
live in the parent tests/e2e/conftest.py. SpendClient exposes the shared Gateway
(GatewayProvider), so the `resources` fixture cleans up keys and customers this
suite creates.

The suite drives real calls through four deployments (gemini, anthropic, bedrock,
openai embeddings). On the stage gateway they are baked into the proxy config; on a
local dev proxy they usually are not, so `driver_models` registers whichever are
missing via /model/new and deletes only the ones it created, never a config-baked
deployment. Each registration carries the provider credential from the test runner's
env when set (so a local proxy whose container env lacks it still works); otherwise
it falls back to an os.environ reference resolved from the proxy's own env, the stage
convention. Bedrock is credentialed with AWS keys rather than a single api_key.
"""

import os
from typing import Iterator

import pytest

from models import LiteLLMParamsBody
from spend_e2e_client import SpendClient, build_client


def _env_ref(name: str) -> str:
    """The runner's value when set (so a local proxy whose container env lacks it
    still works), else an os.environ reference the proxy resolves from its own env."""
    return os.environ.get(name) or f"os.environ/{name}"


def _api_key_params(provider_model: str, env_var: str) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(model=provider_model, api_key=_env_ref(env_var))


def _bedrock_params(provider_model: str) -> LiteLLMParamsBody:
    """Bedrock authenticates with AWS creds, not a single api_key."""
    return LiteLLMParamsBody(
        model=provider_model,
        aws_access_key_id=_env_ref("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=_env_ref("AWS_SECRET_ACCESS_KEY"),
        aws_region_name=os.environ.get("AWS_REGION_NAME") or _env_ref("AWS_REGION"),
    )


DRIVER_MODELS: tuple[tuple[str, LiteLLMParamsBody], ...] = (
    ("gemini-2.5-flash", _api_key_params("gemini/gemini-2.5-flash", "GEMINI_API_KEY")),
    ("claude-haiku-4-5", _api_key_params("anthropic/claude-haiku-4-5", "ANTHROPIC_API_KEY")),
    (
        "bedrock-claude-haiku-4-5",
        _bedrock_params("bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"),
    ),
    (
        "openai-text-embedding-3-small",
        _api_key_params("openai/text-embedding-3-small", "OPENAI_API_KEY"),
    ),
)


@pytest.fixture(scope="session")
def client() -> SpendClient:
    return build_client()


@pytest.fixture(scope="session", autouse=True)
def driver_models(client: SpendClient) -> Iterator[None]:
    existing = frozenset(entry.model_name for entry in client.gateway.model_info())
    created = tuple(
        client.gateway.create_model(name, params)
        for name, params in DRIVER_MODELS
        if name not in existing
    )
    yield
    for model_id in created:
        client.gateway.delete_model(model_id)
