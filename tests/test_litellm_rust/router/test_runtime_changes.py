"""Runtime model list and settings changes on the Rust router backend, against the Python one."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final

import pytest

import litellm
from litellm.router import Router
from litellm.router_backends.python_router import PythonRouter
from litellm.router_backends.rust_router import RustRouter, RustRouterUnsupportedError
from litellm.router_backends.selection import pinned_backend
from litellm.rust_bridge.configuration import Decision
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo
from litellm.types.utils import ModelResponse
from tests.test_litellm_rust.support.router_backends import BACKENDS, Backend

pytestmark = pytest.mark.requires_rust_extension

MESSAGES: Final = ({"role": "user", "content": "hi"},)


def _deployment(group: str, name: str, **params: object) -> Deployment:  # kwargs-ok: extra litellm_params
    return Deployment(
        model_name=group,
        litellm_params=LiteLLM_Params(model=f"openai/{name}", api_key="k", mock_response=f"from {name}", **params),
        model_info=ModelInfo(id=name),
    )


def _arguments() -> Mapping[str, object]:
    return {"model_list": (_deployment("g", "a").model_dump(exclude_none=True),), "num_retries": 0}


async def _answers(
    backend: Backend, model: str, count: int, **kwargs: object
) -> tuple[str, ...]:  # kwargs-ok: request options
    responses: Final = [await backend.acompletion(model, MESSAGES, **kwargs) for _ in range(count)]
    return tuple(_text(response) for response in responses)


def _text(response: object) -> str:
    assert isinstance(response, ModelResponse)
    return str(response.choices[0].message.content)  # non-streaming choice


Change = Callable[[Backend], object]


@pytest.mark.parametrize(
    ("change", "model", "expected"),
    (
        pytest.param(
            lambda backend: backend.upsert_deployment(_deployment("g", "b")),
            "g",
            {"from a", "from b"},
            id="an-added-deployment-takes-traffic",
        ),
        pytest.param(
            lambda backend: backend.upsert_deployment(
                Deployment(
                    model_name="g",
                    litellm_params=LiteLLM_Params(model="openai/a", api_key="k", mock_response="edited a"),
                    model_info=ModelInfo(id="a"),
                )
            ),
            "g",
            {"edited a"},
            id="an-edited-deployment-serves-its-new-params",
        ),
        pytest.param(
            lambda backend: (backend.upsert_deployment(_deployment("g", "b")), backend.delete_deployment("a")),
            "g",
            {"from b"},
            id="a-deleted-deployment-takes-no-traffic",
        ),
        pytest.param(
            lambda backend: (
                backend.upsert_deployment(_deployment("h", "c")),
                backend.update_settings(fallbacks=[{"g": ["h"]}]),
            ),
            "g",
            {"from c"},
            id="fallbacks-set-at-runtime-apply",
        ),
    ),
)
async def test_a_runtime_change_routes_the_same_way_on_both_backends(
    change: Change, model: str, expected: set[str]
) -> None:
    kwargs: Final = {"mock_testing_fallbacks": True} if expected == {"from c"} else {}
    observed: Final = []  # mutable-ok: one answer sequence per backend
    for build in BACKENDS:
        backend = build(_arguments(), 3)
        change(backend)
        observed.append(await _answers(backend, model, 12, **kwargs))

    assert observed[1] == observed[0]
    assert set(observed[1]) == expected


async def test_deleting_the_last_deployment_fails_the_same_way_on_both_backends() -> None:
    errors: Final[list[str]] = []  # mutable-ok: one error per backend
    for build in BACKENDS:
        backend = build(_arguments(), 3)
        removed = backend.delete_deployment("a")
        assert removed is not None
        with pytest.raises(litellm.BadRequestError) as caught:
            await backend.acompletion("g", MESSAGES)
        errors.append(str(caught.value))

    assert errors[1] == errors[0]


def test_an_unchanged_upsert_reports_nothing_on_both_backends() -> None:
    assert [build(_arguments(), 3).upsert_deployment(_deployment("g", "a")) for build in BACKENDS] == [None, None]


async def test_a_change_rust_cannot_serve_hands_the_router_to_python_for_good() -> None:
    with pinned_backend(Decision.RUST_WITH_FALLBACK):
        router: Final = Router(model_list=[_deployment("g", "a").model_dump(exclude_none=True)], num_retries=0)
    assert isinstance(router.backend, RustRouter)

    router.upsert_deployment(_deployment("g", "b", tags=["batch"]))

    assert isinstance(router.backend, PythonRouter)
    assert {deployment["model_info"]["id"] for deployment in router.backend.model_list} == {"a", "b"}
    assert set(await _answers(router, "g", 12)) == {"from a", "from b"}  # pyright: ignore[reportArgumentType]  # Router serves Backend


def test_a_change_rust_cannot_serve_raises_when_rust_is_required() -> None:
    with pinned_backend(Decision.RUST_REQUIRED):
        router: Final = Router(model_list=[_deployment("g", "a").model_dump(exclude_none=True)], num_retries=0)

    with pytest.raises(RustRouterUnsupportedError, match=r"litellm_params\.tags"):
        router.upsert_deployment(_deployment("g", "b", tags=["batch"]))
    assert isinstance(router.backend, RustRouter)
