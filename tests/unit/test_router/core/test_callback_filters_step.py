"""
The ``callback_filters`` host step (``RustCallbackChecks._answer_callback_filters``,
litellm/router_rust/steps/callbacks.py): a pick reads the candidates' deployments back only when a
routing callback filters them, and the kept ids come back in order, once each.
"""

from collections.abc import Mapping
from typing import Final

import pytest

from litellm.integrations.custom_logger import CustomLogger
from litellm.router_rust.steps.callbacks import filtered_ids, filters_deployments
from tests.unit.test_router.core.registry_helpers import MakeCore
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment

GROUP: Final = [deployment("g", "a"), deployment("g", "b"), deployment("g", "c")]


class Narrow(CustomLogger):
    """Keeps the deployments other than ``dropped``."""

    def __init__(self, dropped: str) -> None:
        super().__init__()
        self.dropped = dropped

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list,
        messages: object,
        request_kwargs: object = None,
        parent_otel_span: object = None,
    ) -> list:
        return [d for d in healthy_deployments if d["model_info"]["id"] != self.dropped]


class Observer(CustomLogger):
    """A callback that does not override ``async_filter_deployments``."""


def _entry(deployment_id: object) -> Mapping[str, object]:
    return {"model_info": {"id": deployment_id}}


def test_filtered_ids_keep_the_filter_order_once_each_among_the_candidates() -> None:
    remaining: Final = [_entry("c"), _entry("x"), _entry("a"), _entry("c"), _entry(7), {"model_info": None}, "junk"]

    assert filtered_ids(remaining, ["a", "b", "c"]) == ["c", "a"]


def test_only_an_overriding_filter_counts_as_a_filter() -> None:
    assert filters_deployments(Narrow(dropped="a"))
    assert not filters_deployments(Observer())
    assert not filters_deployments(object())


@pytest.mark.asyncio
async def test_a_pick_without_filters_reads_no_deployment_back(
    make_core: MakeCore, host: ScriptedHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    host.lists["callbacks"].append(Observer())
    core: Final = make_core(model_list=list(GROUP))

    def refuse(_self: object, _ids: object) -> list[dict[str, object]]:  # mutable-ok: the patched signature
        raise AssertionError("no filter runs, so no deployment is read back")

    monkeypatch.setattr(type(core), "_records_of", refuse)

    picked: Final = {(await core.select("g"))["model_info"]["id"] for _ in range(30)}  # pyright: ignore[reportIndexIssue]

    assert picked == {"a", "b", "c"}


@pytest.mark.asyncio
async def test_a_pick_with_a_filter_reads_the_candidates_and_honours_it(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.lists["callbacks"].append(Narrow(dropped="b"))
    core: Final = make_core(model_list=list(GROUP))

    picked: Final = {(await core.select("g"))["model_info"]["id"] for _ in range(30)}  # pyright: ignore[reportIndexIssue]

    assert picked == {"a", "c"}
