"""
The ``health()`` priority helpers and the ``registry()`` members that read router state beyond the
deployment list: ``client(client_type="max_parallel_requests")``, ``plugins("routing")`` and
``zero_cost()``. Each test pins what the Python core does; every core must answer the same.
"""

from typing import Final

import pytest

import litellm
from litellm.router_api import ChangeSet
from litellm.router_utils.client_initalization_utils import MaxParallelRequestsLimit
from litellm.types.router import Deployment, LiteLLM_Params, RoutingContext
from tests.unit.test_router.core.scripted_host import deployment
from tests.unit.test_router.core.selection_helpers import MESSAGES, MakeCore

# --- health().split_priority ----------------------------------------------------------------------


def test_split_priority_gives_the_request_priority_and_the_kwargs_without_it(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    assert core.health().split_priority("g", {"priority": 5, "user": "u"}) == (5, {"user": "u"})


def test_split_priority_falls_back_to_the_default_priority_when_the_request_sets_none(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], default_priority=2)
    assert core.health().split_priority("g", {"user": "u"}) == (2, {"user": "u"})


def test_split_priority_without_a_default_leaves_the_request_unqueued(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    assert core.health().split_priority("g", {"priority": None}) == (None, {})


def test_split_priority_refuses_a_priority_that_is_no_integer(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], default_priority=2)
    with pytest.raises(litellm.BadRequestError, match="priority must be an integer, got str 'high'"):
        core.health().split_priority("g", {"priority": "high"})


def test_split_priority_drops_a_bad_priority_for_a_group_that_drops_params(make_core: MakeCore) -> None:
    """A deployment of the group with ``drop_params=True`` turns the bad priority into the default."""
    core: Final = make_core(model_list=[deployment("g", "a", drop_params=True)], default_priority=2)
    assert core.health().split_priority("g", {"priority": "high"}) == (2, {})


def test_split_priority_reads_a_pinned_deployment_s_params_for_its_id(make_core: MakeCore) -> None:
    """A deployment id as the model reads that deployment's ``drop_params``, not a group's."""
    core: Final = make_core(
        model_list=[deployment("g", "a", drop_params=True), deployment("g", "b")], default_priority=1
    )
    assert core.health().split_priority("a", {"priority": "high"}) == (1, {})
    with pytest.raises(litellm.BadRequestError):
        core.health().split_priority("b", {"priority": "high"})


# --- health().priority_header_fields --------------------------------------------------------------


def test_priority_header_fields_read_an_integer_header(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    assert core.health().priority_header_fields("g", {}, "3") == {"priority": 3}


@pytest.mark.parametrize("header", [None, ""])
def test_priority_header_fields_are_empty_without_a_header(make_core: MakeCore, header: str | None) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    assert core.health().priority_header_fields("g", {}, header) == {}


def test_priority_header_fields_pass_a_bad_header_on_for_the_call_to_refuse(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    assert core.health().priority_header_fields("g", {}, "urgent") == {"priority": "urgent"}


def test_priority_header_fields_drop_a_bad_header_when_the_request_drops_params(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    assert core.health().priority_header_fields("g", {"drop_params": True}, "urgent") == {}


# --- registry().client(client_type="max_parallel_requests") ---------------------------------------


def test_the_max_parallel_requests_client_is_the_deployment_s_cached_limit(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", max_parallel_requests=2)])
    entry: Final = deployment("g", "a", max_parallel_requests=2)
    limit: Final = core.registry().client(entry, client_type="max_parallel_requests")
    assert isinstance(limit, MaxParallelRequestsLimit)
    assert (limit.max_parallel_requests, limit.model_id, limit.model_group, limit.in_flight) == (2, "a", "g", 0)
    assert core.registry().client(entry, client_type="max_parallel_requests") is limit


def test_the_max_parallel_requests_limit_comes_from_rpm_or_the_router_default(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a", rpm=7), deployment("g", "b")], default_max_parallel_requests=4
    )
    by_rpm: Final = core.registry().client(deployment("g", "a", rpm=7), client_type="max_parallel_requests")
    by_default: Final = core.registry().client(deployment("g", "b"), client_type="max_parallel_requests")
    assert isinstance(by_rpm, MaxParallelRequestsLimit)
    assert isinstance(by_default, MaxParallelRequestsLimit)
    assert (by_rpm.max_parallel_requests, by_default.max_parallel_requests) == (7, 4)


def test_a_deployment_without_a_limit_has_no_max_parallel_requests_client(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    assert core.registry().client(deployment("g", "a"), client_type="max_parallel_requests") is None


def test_the_client_lookup_reads_the_deployment_id_first(make_core: MakeCore) -> None:
    """``deployment["model_info"]["id"]``: a deployment without model_info raises, whatever the client type."""
    core: Final = make_core(model_list=[deployment("g", "a")])
    entry: Final = {"model_name": "g", "litellm_params": {"model": "openai/a"}}
    for client_type in ("max_parallel_requests", "async", None):
        with pytest.raises(KeyError, match="model_info"):
            core.registry().client(entry, client_type=client_type)


@pytest.mark.asyncio
async def test_a_full_max_parallel_requests_client_turns_the_next_call_away(make_core: MakeCore) -> None:
    """The limit ``client`` returns is the one calls take their slot from."""
    core: Final = make_core(model_list=[deployment("g", "a", max_parallel_requests=1)], num_retries=0)
    entry: Final = deployment("g", "a", max_parallel_requests=1)
    limit: Final = core.registry().client(entry, client_type="max_parallel_requests")
    assert isinstance(limit, MaxParallelRequestsLimit)
    with limit, pytest.raises(litellm.RateLimitError, match="already has max_parallel_requests=1"):
        await core.acompletion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # every core serves acompletion
    assert limit.in_flight == 0


# --- registry().plugins("routing") and zero_cost() ------------------------------------------------


class _Keep:
    async def run(self, context: RoutingContext) -> RoutingContext:
        return context


def test_the_routing_plugins_are_the_ones_the_router_was_built_with(make_core: MakeCore) -> None:
    plugin: Final = _Keep()
    core: Final = make_core(model_list=[deployment("g", "a")], plugins=[plugin])
    assert core.registry().plugins("routing") == [plugin]


def test_a_router_without_plugins_lists_no_routing_plugins(make_core: MakeCore) -> None:
    assert make_core(model_list=[deployment("g", "a")]).registry().plugins("routing") == []


def test_zero_cost_is_an_empty_lookup_the_auth_checks_fill(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    lookup: Final = core.registry().zero_cost()
    assert lookup == {}
    assert isinstance(lookup, dict)
    lookup["g"] = True
    assert core.registry().zero_cost() == {"g": True}


def test_a_deployment_change_empties_the_zero_cost_lookup(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    lookup: Final = core.registry().zero_cost()
    assert isinstance(lookup, dict)
    lookup["g"] = True
    core.apply(
        ChangeSet(
            upsert=[
                Deployment(
                    model_name="g",
                    litellm_params=LiteLLM_Params(model="openai/b", api_key="sk-test"),
                    model_info={"id": "b"},
                )
            ]
        )
    )
    assert core.registry().zero_cost() == {}
