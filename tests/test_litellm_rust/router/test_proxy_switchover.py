"""What the proxy needs from the Rust router backend: its constructor arguments, its fallback
checks, its reads of the router, and a hand-over to Python for whatever Rust does not serve."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final

import pytest

from litellm.router import Router
from litellm.router_backends.python_router import PythonRouter
from litellm.router_backends.rust_router import RustRouter, RustRouterUnsupportedError
from litellm.router_backends.selection import pinned_backend
from litellm.rust_bridge.configuration import Decision
from litellm.types.router import Deployment, RouterGeneralSettings
from tests.test_litellm_rust.support.router_backends import BACKENDS, Backend, router_headers

pytestmark = pytest.mark.requires_rust_extension

MESSAGES: Final = ({"role": "user", "content": "hi"},)


def _deployment(
    group: str, name: str, response: object = None
) -> dict[str, object]:  # mutable-ok: PythonRouter consumes model_list dicts
    return {
        "model_name": group,
        "litellm_params": {"model": f"openai/{name}", "api_key": "k", "mock_response": response or f"from {name}"},
        "model_info": {"id": name},
    }


def _content(response: object) -> str:
    return str(response.choices[0].message.content)  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownArgumentType]  # ModelResponse


class _Check:
    """A fallback access or budget check that records what it was asked and rejects `denied`."""

    def __init__(self, denied: frozenset[str]) -> None:
        self.denied: Final = denied
        self.asked: Final[list[tuple[str, object, object]]] = []  # mutable-ok: one entry per question

    async def __call__(self, *, model: str, request_kwargs: Mapping[str, object], llm_router: object) -> bool:
        metadata: Final = request_kwargs.get("metadata")
        self.asked.append(
            (
                model,
                request_kwargs.get("model"),
                metadata.get("user_api_key_hash") if isinstance(metadata, Mapping) else None,
            )
        )
        return model not in self.denied


def _arguments(access: _Check, budget: _Check, h_answers: bool = False) -> Mapping[str, object]:
    return {
        "model_list": (
            _deployment("g", "a", Exception("primary broke")),
            _deployment("h", "b", None if h_answers else Exception("h broke")),
            _deployment("i", "c"),
            _deployment("j", "d"),
        ),
        "fallbacks": [{"g": ["h", "i", "j"]}],
        "num_retries": 0,
        "fallback_access_check": access,
        "fallback_budget_check": budget,
    }


Observed = tuple[
    str, Mapping[str, object], tuple[tuple[str, object, object], ...], tuple[tuple[str, object, object], ...]
]


async def _observe(
    build: Callable[[Mapping[str, object], int], Backend], denied: frozenset[str], over: frozenset[str]
) -> Observed:
    access: Final = _Check(denied)
    budget: Final = _Check(over)
    response: Final = await build(_arguments(access, budget), 1).acompletion(
        "g", MESSAGES, metadata={"user_api_key_hash": "caller"}
    )
    content: Final = str(response.choices[0].message.content)  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownArgumentType]  # ModelResponse
    return content, router_headers(response), tuple(access.asked), tuple(budget.asked)


@pytest.mark.parametrize(
    ("denied", "over", "expected"),
    (
        pytest.param(frozenset(), frozenset(), "from c", id="nothing-rejected"),
        pytest.param(frozenset({"h"}), frozenset(), "from c", id="an-unauthorized-target-is-skipped"),
        pytest.param(frozenset(), frozenset({"h", "i"}), "from d", id="over-budget-targets-are-skipped"),
    ),
)
async def test_both_backends_put_each_fallback_target_to_the_proxys_checks(
    denied: frozenset[str], over: frozenset[str], expected: str
) -> None:
    observed: Final = [await _observe(build, denied, over) for build in BACKENDS]

    assert observed[1] == observed[0]
    assert observed[1][0] == expected


async def test_the_checks_read_the_request_as_the_failed_hop_left_it() -> None:
    _, _, access, _ = await _observe(BACKENDS[1], frozenset(), frozenset())

    assert access == (("h", "g", "caller"), ("i", "h", "caller"))


@pytest.mark.parametrize("build", BACKENDS)
def test_the_sync_path_runs_the_checks_too(build: Callable[[Mapping[str, object], int], Backend]) -> None:
    access: Final = _Check(frozenset({"h"}))
    budget: Final = _Check(frozenset({"h"}))

    response: Final = build(_arguments(access, budget, h_answers=True), 1).completion("g", MESSAGES)

    assert response.choices[0].message.content == "from c"  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]  # ModelResponse
    assert [model for model, _, _ in access.asked] == ["h", "i"]
    assert [model for model, _, _ in budget.asked] == ["i"]


def _facade(arguments: Mapping[str, object], decision: Decision = Decision.RUST_WITH_FALLBACK) -> Router:
    with pinned_backend(decision):
        return Router(**arguments)


def _plain(arguments: Mapping[str, object]) -> Mapping[str, object]:
    return {key: list(value) if isinstance(value, tuple) else value for key, value in arguments.items()}


_TWO_GROUPS: Final = _plain({"model_list": (_deployment("g", "a"), _deployment("g", "b"), _deployment("h", "c"))})
_READS: Final = (
    ("get_model_list", lambda router: router.get_model_list()),
    ("get_model_names", lambda router: router.get_model_names()),
    ("get_model_ids", lambda router: router.get_model_ids()),
    ("get_deployment", lambda router: router.get_deployment(model_id="b")),
    ("get_model_group_info", lambda router: router.get_model_group_info(model_group="g")),
    ("get_model_access_groups", lambda router: router.get_model_access_groups()),
    ("deployment_names", lambda router: router.deployment_names),
    ("is_recognized_model", lambda router: router.is_recognized_model("h")),
    ("get_fully_blocked_model_names", lambda router: router.get_fully_blocked_model_names()),
    ("_zero_cost_cache", lambda router: getattr(router, "_zero_cost_cache", None)),
)


@pytest.mark.parametrize(("name", "read"), _READS, ids=[name for name, _ in _READS])
async def test_the_proxys_reads_answer_as_the_python_router_does_and_stay_on_rust(
    name: str, read: Callable[[Router], object]
) -> None:
    del name
    rust: Final = _facade(_TWO_GROUPS)
    python: Final = _facade(_TWO_GROUPS, Decision.PYTHON)
    for router in (rust, python):
        router.upsert_deployment(Deployment(**_deployment("h", "d")))

    assert read(rust) == read(python)
    assert isinstance(rust.backend, RustRouter)


async def test_an_operation_rust_does_not_serve_hands_the_instance_over_for_good() -> None:
    router: Final = _facade(_TWO_GROUPS)
    router.search_tools = [{"search_tool_name": "web", "litellm_params": {"search_provider": "tavily"}}]
    router.update_settings(num_retries=4)
    rust: Final = router.backend

    response: Final = await router.aembedding(model="g", input=["hi"], mock_response=[0.5])

    assert response.data[0]["embedding"] == [0.5]  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]  # EmbeddingResponse
    assert isinstance(router.backend, PythonRouter)
    assert (router.search_tools, router.num_retries) == (rust.search_tools, 4)  # pyright: ignore[reportAttributeAccessIssue]  # Router views
    assert _content(await router.acompletion("h", MESSAGES)) == "from c"
    assert getattr(rust, "num_retries") == 4


@pytest.mark.parametrize(
    "option",
    (
        pytest.param({"specific_deployment": True, "model": "openai/c"}, id="specific-deployment"),
        pytest.param({"api_key": "client-key", "api_base": "https://example.invalid"}, id="client-side-credentials"),
        pytest.param({"routing_strategy": "least-busy"}, id="per-request-routing-strategy"),
    ),
)
async def test_a_request_option_rust_does_not_serve_is_answered_by_the_python_router(
    option: Mapping[str, object],
) -> None:
    router: Final = _facade(_TWO_GROUPS)
    model: Final = str(option.get("model", "h"))

    response: Final = await router.acompletion(
        model, MESSAGES, **{key: value for key, value in option.items() if key != "model"}
    )

    assert _content(response) == "from c"
    assert isinstance(router.backend, PythonRouter)


def test_required_mode_raises_instead_of_handing_over() -> None:
    router: Final = _facade(_TWO_GROUPS, Decision.RUST_REQUIRED)

    with pytest.raises(RustRouterUnsupportedError, match=r"Router\.aembedding"):
        _ = router.aembedding

    assert isinstance(router.backend, RustRouter)


def test_an_assignable_setting_stays_on_rust() -> None:
    router: Final = _facade(_TWO_GROUPS)

    router.cache_responses = True

    assert isinstance(router.backend, RustRouter)
    assert router.backend._normalizer.cache_responses is True  # pyright: ignore[reportPrivateUsage]  # where attempts read it


def test_the_proxys_constructor_arguments_stay_on_rust_and_read_like_python() -> None:
    access: Final = _Check(frozenset())
    arguments: Final = {
        "model_list": [
            *_TWO_GROUPS["model_list"],  # pyright: ignore[reportGeneralTypeIssues]  # a list
            {"model_name": "broken", "litellm_params": {"model": "no-such-provider-model"}},
        ],
        "router_general_settings": RouterGeneralSettings(async_only_mode=True),
        "search_tools": [],
        "ignore_invalid_deployments": True,
        "fallback_access_check": access,
        "fallback_budget_check": access,
        "auto_router_capability_limit": lambda: None,
    }
    rust: Final = _facade(arguments)
    python: Final = _facade(arguments, Decision.PYTHON)

    assert isinstance(rust.backend, RustRouter)
    assert (rust.model_names, rust.get_model_list()) == (python.model_names, python.get_model_list())
    assert rust.router_general_settings.async_only_mode is True


async def test_the_proxys_anthropic_messages_entry_point_is_served_by_rust() -> None:
    router: Final = _facade(_TWO_GROUPS)

    response: Final = await router.anthropic_messages(model="h", messages=list(MESSAGES), max_tokens=10)

    assert isinstance(response, Mapping)
    assert response["content"][0]["text"] == "from c"
    assert isinstance(router.backend, RustRouter)
