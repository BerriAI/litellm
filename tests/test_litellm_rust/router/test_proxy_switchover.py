"""What the proxy needs from the Rust router backend: its constructor arguments, its fallback
checks, its reads of the router, and a hand-over to Python for whatever Rust does not serve."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final

import pytest

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
