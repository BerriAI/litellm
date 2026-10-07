"""The Rust router backend against the Python one, on the same config and the same seed."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Final, Protocol

import pytest

import litellm
from litellm.router_backends.python_router import PythonRouter
from litellm.router_backends.rust_router import NATIVE_ROUTER, RustRouter
from litellm.router_backends.selection import select_backend
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from litellm.types.router import RouterRateLimitError
from litellm.types.utils import ModelResponse

pytestmark = pytest.mark.requires_rust_extension

MESSAGES: Final = ({"role": "user", "content": "hi"},)


class Backend(Protocol):
    async def acompletion(
        self, model: str, messages: Sequence[Mapping[str, object]], **kwargs: object
    ) -> object: ...  # kwargs-ok: Router.acompletion's surface

    def completion(
        self, model: str, messages: Sequence[Mapping[str, object]], **kwargs: object
    ) -> object: ...  # kwargs-ok: Router.completion's surface


def _deployment(
    group: str, name: str, response: object = None
) -> dict[str, object]:  # mutable-ok: PythonRouter consumes model_list dicts
    return {
        "model_name": group,
        "litellm_params": {"model": f"openai/{name}", "api_key": "k", "mock_response": response or f"from {name}"},
        "model_info": {"id": name},
    }


def _python(arguments: Mapping[str, object], seed: int) -> Backend:
    random.seed(seed)
    router: Final = PythonRouter(
        **{key: list(value) if isinstance(value, tuple) else value for key, value in arguments.items()}
    )
    return router  # pyright: ignore[reportReturnType]  # PythonRouter.acompletion's untyped signature


def _rust(arguments: Mapping[str, object], seed: int) -> Backend:
    native: Final = NATIVE_ROUTER.load()
    assert native is not None
    return RustRouter(
        {key: list(value) if isinstance(value, tuple) else value for key, value in arguments.items()}, native, seed
    )


BACKENDS: Final[tuple[Callable[[Mapping[str, object], int], Backend], ...]] = (_python, _rust)


def _content(response: object) -> str:
    assert isinstance(response, ModelResponse)
    return str(response.choices[0].message.content)  # pyright: ignore[reportAttributeAccessIssue]  # non-streaming choice


def _headers(response: object) -> Mapping[str, object]:
    hidden: Final = getattr(response, "_hidden_params", {})
    headers: Final = hidden.get("additional_headers", {}) if isinstance(hidden, Mapping) else {}
    return {key: value for key, value in headers.items() if key.startswith("x-litellm-")}


async def _picks(backend: Backend, count: int) -> list[str]:
    return [_content(await backend.acompletion("g", MESSAGES)) for _ in range(count)]


@pytest.mark.parametrize("weights", ((None, None, None), (3, 1, 1)))
async def test_seeded_backends_pick_the_same_deployments(weights: tuple[int | None, ...]) -> None:
    model_list: Final = tuple(
        {**_deployment("g", name), "litellm_params": {**_deployment("g", name)["litellm_params"], "weight": weight}}  # pyright: ignore[reportGeneralTypeIssues]  # nested dict merge
        if weight is not None
        else _deployment("g", name)
        for name, weight in zip(("a", "b", "c"), weights)
    )

    python_picks: Final = await _picks(_python({"model_list": model_list}, 11), 12)
    rust_picks: Final = await _picks(_rust({"model_list": model_list}, 11), 12)

    assert rust_picks == python_picks
    assert len(set(python_picks)) > 1


@pytest.mark.parametrize("build", BACKENDS)
async def test_a_mock_fallback_answers_from_the_fallback_group(
    build: Callable[[Mapping[str, object], int], Backend],
) -> None:
    backend: Final = build(
        {"model_list": (_deployment("g", "a"), _deployment("h", "b")), "fallbacks": [{"g": ["h"]}]}, 1
    )

    response: Final = await backend.acompletion("g", MESSAGES, mock_testing_fallbacks=True)

    assert _content(response) == "from b"
    assert _headers(response) == {
        "x-litellm-model-group": "h",
        "x-litellm-attempted-retries": 0,
        "x-litellm-attempted-fallbacks": 1,
    }


async def test_failed_fallbacks_raise_the_same_error_from_both_backends() -> None:
    arguments: Final = {
        "model_list": (
            _deployment("g", "a", Exception("primary broke")),
            _deployment("h", "b", Exception("fallback broke")),
        ),
        "fallbacks": [{"g": ["h"]}],
        "num_retries": 1,
    }
    raised: Final[list[BaseException]] = []  # mutable-ok: collects one error per backend
    for build in BACKENDS:
        with pytest.raises(litellm.InternalServerError) as caught:
            await build(arguments, 3).acompletion("g", MESSAGES)
        raised.append(caught.value)

    python_error, rust_error = raised
    assert str(rust_error) == str(python_error)
    assert "Fallback to h also failed" in str(rust_error)
    assert (getattr(rust_error, "num_retries", None), getattr(rust_error, "max_retries", None)) == (
        getattr(python_error, "num_retries", None),
        getattr(python_error, "max_retries", None),
    )


async def test_rate_limited_deployments_cool_down_until_the_router_rejects() -> None:
    arguments: Final = {
        "model_list": (
            _deployment("g", "a", "litellm.RateLimitError"),
            _deployment("g", "b", "litellm.RateLimitError"),
        ),
        "num_retries": 0,
    }
    outcomes: Final[list[tuple[str, ...]]] = []  # mutable-ok: one sequence of outcomes per backend
    for build in BACKENDS:
        backend = build(arguments, 5)
        seen: list[str] = []  # mutable-ok: outcomes in order
        for _ in range(3):
            try:
                await backend.acompletion("g", MESSAGES)
            except RouterRateLimitError as error:
                seen.append(f"rejected:{error.type}")
            except litellm.RateLimitError:
                seen.append("rate_limited")
        outcomes.append(tuple(seen))

    assert outcomes[0] == outcomes[1]
    assert outcomes[1][-1] == "rejected:all_deployments_in_cooldown"


@pytest.mark.parametrize("build", BACKENDS)
def test_sync_completion_retries_and_answers(build: Callable[[Mapping[str, object], int], Backend]) -> None:
    backend: Final = build({"model_list": (_deployment("g", "a"),), "num_retries": 2}, 1)

    response: Final = backend.completion("g", MESSAGES)

    assert _content(response) == "from a"


async def test_the_facade_serves_a_supported_config_from_the_rust_backend() -> None:
    backend: Final = select_backend(
        (), {"model_list": [_deployment("g", "a")]}, rules=(RouteRule(Route.ROUTER, Rollout.RUST_REQUIRED),)
    )

    assert isinstance(backend, RustRouter)
    assert _content(await backend.acompletion("g", MESSAGES)) == "from a"
    assert backend.get_model_names() == ["g"]
    with pytest.raises(NotImplementedError, match=r"Router\.upsert_deployment"):
        _ = backend.upsert_deployment
