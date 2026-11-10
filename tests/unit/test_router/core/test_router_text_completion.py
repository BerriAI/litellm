"""The sync ``Router.text_completion`` walk: retries, fallbacks and the text attempt's model."""

from collections.abc import Callable
from typing import Final

import pytest

import litellm
from litellm.router_core import RouterCore
from tests.unit.test_router.core.scripted_host import (
    ScriptedHost,
    deployment,
    rate_limit,
)

MakeCore = Callable[..., RouterCore]

PROMPT: Final = "Say hello."
GROUP_A: Final = [deployment("a", "a1")]


def build(make_core: MakeCore, **arguments: object) -> RouterCore:  # kwargs-ok: the router's constructor arguments
    return make_core(model_list=GROUP_A, **arguments)


def text_completion(core: RouterCore, model: str = "a", **request: object) -> object:  # kwargs-ok: request kwargs
    return core.text_completion(  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method
        model=model,
        prompt=PROMPT,
        **request,
    )


def test_text_completion_sends_the_deployment_model_and_prompt(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, num_retries=0)

    response = text_completion(core)

    assert getattr(response, "model", None) == "openai/a1"
    assert [call.model for call in host.calls_to("text_completion")] == ["openai/a1"]
    assert host.calls_to("text_completion")[0].kwargs["prompt"] == PROMPT
    assert host.calls_to("text_completion")[0].model_group == "a"


def test_text_completion_retries_after_a_deployment_failure(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, num_retries=1)
    host.answer("openai/a1", rate_limit())

    response = text_completion(core)

    assert getattr(response, "model", None) == "openai/a1"
    assert host.deployments_called("text_completion") == ["a1", "a1"]


def test_text_completion_stops_after_the_retry_budget(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, num_retries=1)
    host.answer("openai/a1", rate_limit())

    with pytest.raises(litellm.RateLimitError):
        text_completion(core)

    assert host.deployments_called("text_completion") == ["a1", "a1"]


def test_text_completion_falls_back_to_the_next_group(make_core: MakeCore, host: ScriptedHost) -> None:
    core = make_core(
        model_list=[deployment("a", "a1"), deployment("b", "b1")],
        fallbacks=[{"a": ["b"]}],
        num_retries=0,
    )
    host.answer("openai/a1", rate_limit())

    response = text_completion(core)

    assert getattr(response, "model", None) == "openai/b1"
    assert host.deployments_called("text_completion") == ["a1", "b1"]


def test_text_completion_uses_the_request_fallback_list(make_core: MakeCore, host: ScriptedHost) -> None:
    core = make_core(
        model_list=[deployment("a", "a1"), deployment("b", "b1")],
        num_retries=0,
    )
    host.answer("openai/a1", rate_limit())

    response = text_completion(core, fallbacks=["b"])

    assert getattr(response, "model", None) == "openai/b1"
    assert host.deployments_called("text_completion") == ["a1", "b1"]
