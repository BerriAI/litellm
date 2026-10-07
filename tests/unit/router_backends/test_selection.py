from __future__ import annotations

from collections.abc import Generator
from typing import Final

import pytest

from litellm.router_backends.python_router import PythonRouter
from litellm.router_backends.selection import RustRouterUnsupportedError, select_backend
from litellm.rust_bridge import configuration
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout

MODEL_LIST: Final = ({"model_name": "gpt", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "k"}},)


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    configuration.reset_rust_configuration()
    yield
    configuration.reset_rust_configuration()


def test_python_decision_builds_the_python_router_with_the_callers_arguments() -> None:
    backend: Final = select_backend((), {"model_list": list(MODEL_LIST), "num_retries": 7}, rules=())

    assert isinstance(backend, PythonRouter)
    assert backend.num_retries == 7
    assert backend.get_model_names() == ["gpt"]


def test_opt_in_declines_an_unsupported_config_to_the_python_router() -> None:
    rules: Final = (RouteRule(Route.ROUTER, Rollout.RUST_OPT_OUT),)

    backend: Final = select_backend((), {"model_list": list(MODEL_LIST), "routing_strategy": "least-busy"}, rules)

    assert isinstance(backend, PythonRouter)
    assert backend.routing_strategy == "least-busy"


def test_required_raises_naming_what_the_rust_router_cannot_serve() -> None:
    rules: Final = (RouteRule(Route.ROUTER, Rollout.RUST_REQUIRED),)

    with pytest.raises(RustRouterUnsupportedError, match="'routing_strategy'"):
        select_backend((), {"model_list": list(MODEL_LIST), "routing_strategy": "least-busy"}, rules)
