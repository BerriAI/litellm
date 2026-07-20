"""Router suite's `client` fixture.

The shared lifecycle (resources/scoped_key), proxy liveness gate, and e2e marker
live in the parent tests/e2e/conftest.py. ComplexityRouterClient holds the shared
ProxyClient, so the `resources` fixture cleans up keys this suite creates.

Also registers `complexity-smart-router` via management /model/new when the
proxy does not already list it (compose has it in static config; stage does not).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from requests import RequestException

from complexity_router_client import ComplexityRouterClient, build_client
from proxy_client import ProxyClient
from e2e_http import NoBody, Success
from lifecycle import ResourceManager
from models import (
    ChatBody,
    ChatMessage,
    KeyGenerateBody,
    LiteLLMParamsBody,
    ModelsListResponse,
)

ROUTER_MODEL = "complexity-smart-router"
ROUTER_PARAMS = LiteLLMParamsBody(
    model="auto_router/complexity_router",
    complexity_router_config={
        "classifier_type": "llm",
        "classifier_llm_config": {"model": "gpt-5.5"},
        "tiers": {
            "SIMPLE": "gpt-5.5",
            "MEDIUM": "claude-haiku-4-5",
            "COMPLEX": "claude-haiku-4-5",
            "REASONING": "claude-haiku-4-5",
        },
    },
)
# Key must be allowed to call the virtual router and both tier backends.
ROUTER_KEY_MODELS = [ROUTER_MODEL, "gpt-5.5", "claude-haiku-4-5"]


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> ComplexityRouterClient:
    return build_client(proxy)


def _model_is_servable(proxy: ProxyClient, model_name: str) -> bool:
    result = proxy.transport.get(
        "/v1/models",
        headers=proxy.transport.master,
        params=NoBody(),
        response_type=ModelsListResponse,
    )
    return isinstance(result, Success) and any(entry.id == model_name for entry in result.data.data)


def _router_is_callable(proxy: ProxyClient) -> bool:
    """True only when a short chat against the virtual router succeeds; every error
    (the Invalid-model-name reload race, but also 401, 5xx, and network) counts as
    not-callable so infra/auth blips can't be mistaken for a working router."""
    key = proxy.generate_key(KeyGenerateBody(models=ROUTER_KEY_MODELS, user_id="e2e-complexity-probe"))
    try:
        result = proxy.chat(
            key,
            ChatBody(
                model=ROUTER_MODEL,
                messages=[ChatMessage(role="user", content="hi")],
                max_tokens=1,
            ),
        )
    finally:
        proxy.delete_key(key)
    return isinstance(result, Success)


@pytest.fixture(scope="session", autouse=True)
def _ensure_complexity_smart_router(  # pyright: ignore[reportUnusedFunction]  # pytest autouse session fixture, wired by name
    client: ComplexityRouterClient,
) -> Iterator[None]:
    """Ensure the complexity router virtual model exists for this session.

    Compose already declares it in docker-compose.yml; stage does not. Register
    via ProxyClient.create_model (waits for data-plane /v1/models) when missing, then
    probe a real chat so a list-only false positive cannot pass the fixture.
    """
    proxy = client.proxy
    if _model_is_servable(proxy, ROUTER_MODEL) and _router_is_callable(proxy):
        yield
        return

    try:
        model_id = proxy.create_model(ROUTER_MODEL, ROUTER_PARAMS)
    except (AssertionError, RequestException) as exc:
        if _model_is_servable(proxy, ROUTER_MODEL) and _router_is_callable(proxy):
            yield
            return
        raise AssertionError(
            f"failed to register {ROUTER_MODEL!r} for the complexity router e2e "
            f"(not listed/callable on the data plane and /model/new failed): {exc}"
        ) from exc

    try:
        if not _router_is_callable(proxy):
            raise AssertionError(
                f"{ROUTER_MODEL!r} registered as {model_id!r} and listed on "
                f"/v1/models but chat still returns Invalid model name; "
                f"data-plane router reload incomplete"
            )
        yield
    finally:
        proxy.delete_model(model_id)


@pytest.fixture
def complexity_key(resources: ResourceManager, client: ComplexityRouterClient) -> str:
    """Per-test key allowed to call the complexity router and its tier backends."""
    key = client.proxy.generate_key(
        KeyGenerateBody(models=ROUTER_KEY_MODELS, user_id="e2e-complexity-router")
    )
    resources.defer(lambda: client.proxy.delete_key(key))
    return key
