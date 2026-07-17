"""Router suite's `client` fixture.

The shared lifecycle (resources/scoped_key), proxy liveness skip, and e2e marker
live in the parent tests/e2e/conftest.py. ComplexityRouterClient holds the shared
Gateway, so the `resources` fixture cleans up keys this suite creates.

Also registers `complexity-smart-router` via management /model/new when the
proxy does not already list it (compose has it in static config; stage does not).
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from requests import RequestException

from complexity_router_client import ComplexityRouterClient, build_client
from e2e_gateway import Gateway
from e2e_http import NoBody, Success, unwrap
from models import (
    LiteLLMParamsBody,
    ModelInfoBody,
    ModelNewBody,
    ModelNewResponse,
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


def _register_router_model(gateway: Gateway) -> str:
    """POST /model/new only; returns the proxy model_id before data-plane wait.

    Split from create_model so a slow control→data propagation timeout still
    leaves us a model_id for teardown (avoids orphaning complexity-smart-router).
    """
    return unwrap(
        gateway.transport.post(
            "/model/new",
            headers=gateway.transport.master,
            json=ModelNewBody(
                model_name=ROUTER_MODEL,
                litellm_params=ROUTER_PARAMS,
                model_info=ModelInfoBody(),
            ),
            response_type=ModelNewResponse,
        )
    ).model_id


def _await_router_model_servable(gateway: Gateway) -> None:
    deadline = time.monotonic() + gateway.poll_timeout
    while time.monotonic() < deadline:
        if _model_is_servable(gateway, ROUTER_MODEL):
            return
        time.sleep(gateway.poll_interval)
    raise AssertionError(
        f"model {ROUTER_MODEL!r} was created but never became servable on the data "
        f"plane within {gateway.poll_timeout}s of /model/new"
    )


@pytest.fixture(scope="session", autouse=True)
def _ensure_complexity_smart_router(  # pyright: ignore[reportUnusedFunction]  # pytest autouse session fixture, wired by name
    client: ComplexityRouterClient,
) -> Iterator[None]:
    """Ensure the complexity router virtual model exists for this session.

    Compose already declares it in docker-compose.yml; stage does not. Register
    via /model/new when missing and tear down only what we created.

    Always re-check servability after registration: a bare /model/new 200 is not
    enough if control→data propagation lags (stage) or the data plane filters
    the virtual name. Without this the chat cell fails with a vague
    ``Invalid model name`` 400 instead of a registration error.
    """
    gateway = client.gateway
    if _model_is_servable(gateway, ROUTER_MODEL):
        yield
        return

    model_id: str | None = None
    try:
        model_id = _register_router_model(gateway)
    except (AssertionError, RequestException) as exc:
        if _model_is_servable(gateway, ROUTER_MODEL):
            yield
            return
        raise AssertionError(
            f"failed to register {ROUTER_MODEL!r} for the complexity router e2e "
            f"(not listed on /v1/models and /model/new failed): {exc}"
        ) from exc

    try:
        _await_router_model_servable(gateway)
        if not _model_is_servable(gateway, ROUTER_MODEL):
            raise AssertionError(
                f"{ROUTER_MODEL!r} registered as {model_id!r} but still missing "
                f"from /v1/models after wait; data-plane sync failed"
            )
        yield
    finally:
        if model_id is not None:
            gateway.delete_model(model_id)
