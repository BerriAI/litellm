"""Router suite's `client` fixture.

The shared lifecycle (resources/scoped_key), proxy liveness skip, and e2e marker
live in the parent tests/e2e/conftest.py. ComplexityRouterClient holds the shared
Gateway, so the `resources` fixture cleans up keys this suite creates.

Also registers `complexity-smart-router` via management /model/new when the
proxy does not already list it (compose has it in static config; stage does not).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from requests import RequestException

from complexity_router_client import ComplexityRouterClient, build_client
from e2e_gateway import Gateway
from e2e_http import NoBody, Success
from models import LiteLLMParamsBody, ModelsListResponse

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


@pytest.fixture(scope="session")
def client() -> ComplexityRouterClient:
    return build_client()


def _model_is_servable(gateway: Gateway, model_name: str) -> bool:
    result = gateway.transport.get(
        "/v1/models",
        headers=gateway.transport.master,
        params=NoBody(),
        response_type=ModelsListResponse,
    )
    return isinstance(result, Success) and any(entry.id == model_name for entry in result.data.data)


@pytest.fixture(scope="session", autouse=True)
def _ensure_complexity_smart_router(  # pyright: ignore[reportUnusedFunction]  # pytest autouse session fixture, wired by name
    client: ComplexityRouterClient,
) -> Iterator[None]:
    """Ensure the complexity router virtual model exists for this session.

    Compose already declares it in docker-compose.yml; stage does not. Register
    via /model/new when missing and tear down only what we created.
    """
    gateway = client.gateway
    if _model_is_servable(gateway, ROUTER_MODEL):
        yield
        return

    try:
        model_id = gateway.create_model(ROUTER_MODEL, ROUTER_PARAMS)
    except (AssertionError, RequestException) as exc:
        if _model_is_servable(gateway, ROUTER_MODEL):
            yield
            return
        raise AssertionError(
            f"failed to register {ROUTER_MODEL!r} for the complexity router e2e "
            f"(not listed on /v1/models and /model/new failed): {exc}"
        ) from exc

    try:
        yield
    finally:
        gateway.delete_model(model_id)
