"""`client_apps` suite's `client` fixture and the two deployments it drives.

Lifecycle (resources/scoped_key), proxy liveness gate, and the e2e/covers
markers all live in the parent tests/e2e/conftest.py. ClientAppsClient holds the
shared ProxyClient so the keys these tests mint tear down through it.

The suite registers its own deployments through /model/new, one Anthropic and
one OpenAI, under names unique to the run, and deletes them at session end, so
it never depends on which aliases the proxy's config file serves. Each carries
the provider key from the test runner's env when set; otherwise an os.environ
reference the proxy resolves from its own env. They stay on the real provider
path (`provider_live`) because a coding client's prompts change every run, so a
recorded response could never stand in for them. Each deployment's name starts
with the real model id: Claude Code 2.1.278 keys its request shape on the model
name, and an unrecognized one gets a system-role message Anthropic rejects with a
400 before the CLI falls back, which the proxy would bill as a failure row.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Final

import pytest

from client_apps_client import ClientAppsClient
from e2e_config import unique_marker
from models import LiteLLMParamsBody
from proxy_client import ProxyClient


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> ClientAppsClient:
    return ClientAppsClient(proxy=proxy)


@pytest.fixture(scope="session")
def anthropic_model(proxy: ProxyClient) -> Iterator[str]:
    yield from _suite_model(proxy, "anthropic/claude-haiku-4-5", "ANTHROPIC_API_KEY")


@pytest.fixture(scope="session")
def openai_model(proxy: ProxyClient) -> Iterator[str]:
    yield from _suite_model(proxy, "openai/gpt-5.6", "OPENAI_API_KEY")


def _suite_model(proxy: ProxyClient, provider_model: str, env_var: str) -> Iterator[str]:
    model_name: Final = f"{provider_model.split('/', 1)[1]}-client-apps-{unique_marker()}"
    model_id: Final = proxy.create_model(
        model_name,
        LiteLLMParamsBody(model=provider_model, api_key=os.environ.get(env_var) or f"os.environ/{env_var}"),
        provider_live=True,
    )
    yield model_name
    proxy.delete_model(model_id)
