"""
Gap tests for the deployment registry of ``router_python_core.py`` (about lines 11550-14060):
provider validation in ``_add_deployment``, the os.environ references of Azure ``dataSources``,
vertex pass-through registration, config-sourced rows, index repair after a delete, cost-map
registration (``responses/`` keys, strategy-marker pricing, replay), deployment budgets kept in
step with upserts and deletes, credentials with provider, router / deployment model info, and how
model group info merges its deployments.
"""

import os
import uuid
from collections.abc import Iterator
from typing import Final

import pytest

import litellm
from litellm.router_api import ChangeSet
from litellm.types.router import Deployment
from tests.unit.test_router.core.gaps_l11100_helpers import MakeCore, ModelsEndpoint, complexity_marker, model_ids
from tests.unit.test_router.core.registry_helpers import team_deployment
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment


def _unique(prefix: str = "l11100") -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


@pytest.fixture
def data_source_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("L11100_ENDPOINT", "https://search.example")
    monkeypatch.delenv("L11100_MISSING", raising=False)
    yield


# --- provider validation ------------------------------------------------------------------------------


def test_a_provider_missing_from_the_hosts_provider_list_is_unsupported(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.set_dynamic(provider_list=["anthropic"])

    with pytest.raises(Exception, match=r"^Unsupported provider - openai$"):
        make_core(model_list=[deployment("gpt", "a")])


def test_ignore_invalid_deployments_drops_a_deployment_of_an_unsupported_provider(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.set_dynamic(provider_list=["anthropic"])

    core: Final = make_core(
        model_list=[deployment("gpt", "a"), deployment("claude", "c", model="anthropic/claude-opus-4-6")],
        ignore_invalid_deployments=True,
    )

    assert model_ids(core) == ["c"]


def test_a_prompt_management_model_skips_provider_validation(make_core: MakeCore, host: ScriptedHost) -> None:
    """A ``<custom logger>/...`` model (here langfuse) is resolved from the prompt at call time, so it is
    accepted even when no provider is supported at all."""
    host.set_dynamic(provider_list=[])

    core: Final = make_core(model_list=[deployment("prompted", "p", model="langfuse/my-prompt")])

    assert model_ids(core) == ["p"]
    assert core.registry().names(kind="deployment_names") == ["langfuse/my-prompt"]


# --- Azure dataSources and vertex pass-through ---------------------------------------------------------


@pytest.mark.usefixtures("data_source_env")
def test_data_source_endpoint_and_key_env_references_are_resolved(make_core: MakeCore) -> None:
    """Only ``endpoint`` and ``key`` are read from the environment; a missing variable becomes ``""``."""
    core: Final = make_core(
        model_list=[
            deployment(
                "gpt",
                "a",
                dataSources=[
                    {
                        "type": "AzureCognitiveSearch",
                        "parameters": {
                            "endpoint": "os.environ/L11100_ENDPOINT",
                            "key": "os.environ/L11100_MISSING",
                            "indexName": "os.environ/L11100_ENDPOINT",
                        },
                    }
                ],
            )
        ]
    )

    stored: Final = core.registry().all()[0]["litellm_params"]["dataSources"]
    assert stored == [
        {
            "type": "AzureCognitiveSearch",
            "parameters": {
                "endpoint": "https://search.example",
                "key": "",
                "indexName": "os.environ/L11100_ENDPOINT",
            },
        }
    ]


def test_a_vertex_pass_through_deployment_without_project_and_location_is_rejected(make_core: MakeCore) -> None:
    with pytest.raises(ValueError, match="vertex_project") as raised:
        make_core(model_list=[deployment("gem", "g", model="vertex_ai/gemini-pro", use_in_pass_through=True)])
    assert str(raised.value) == (
        "vertex_project, and vertex_location must be set in litellm_params for pass-through endpoints."
    )


def test_pass_through_needs_no_registration_for_other_providers(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("gpt", "a", use_in_pass_through=True)])

    assert model_ids(core) == ["a"]


# --- config rows and index repair -----------------------------------------------------------------------


def test_config_deployments_leave_out_rows_that_came_from_the_db(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            deployment("gpt", "a"),
            deployment("gpt", "b", model_info={"db_model": True}),
            deployment("other", "c", model_info={"db_model": False}),
        ]
    )

    assert [entry["model_info"]["id"] for entry in core.registry().deployments(source="config")] == ["a", "c"]


def test_deleting_a_later_deployment_keeps_earlier_team_indices_and_shifts_later_ones(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            team_deployment("internal-1", "t1", "TA", "pub"),
            deployment("gpt", "b"),
            team_deployment("internal-2", "t2", "TA", "other"),
        ]
    )

    core.apply(ChangeSet(delete=["b"]))

    assert dict(core.registry().index("team_model")) == {("TA", "pub"): [0], ("TA", "other"): [1]}


def test_deleting_a_teams_only_wildcard_deployment_drops_its_team_pattern_router(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[team_deployment("internal", "tw", "TA", "openai/*"), deployment("gpt", "b")],
    )
    assert list(core.registry().patterns(team=True)) == ["TA"]

    core.apply(ChangeSet(delete=["tw"]))

    assert dict(core.registry().patterns(team=True)) == {}


# --- cost-map registration ---------------------------------------------------------------------------------


def test_a_responses_model_registers_its_backend_key_with_and_without_the_responses_segment(
    make_core: MakeCore,
) -> None:
    """The shared backend keys carry no custom pricing: that stays under the deployment id."""
    name: Final = _unique()
    deployment_id: Final = _unique("l11100-id")
    make_core(model_list=[deployment("gpt", deployment_id, model=f"openai/responses/{name}", input_cost_per_token=0.1)])

    assert litellm.model_cost[f"openai/responses/{name}"].get("input_cost_per_token") is None
    assert litellm.model_cost[f"openai/{name}"].get("input_cost_per_token") is None
    assert litellm.model_cost[deployment_id]["input_cost_per_token"] == 0.1


def test_custom_pricing_on_a_strategy_marker_is_not_registered_as_a_price(make_core: MakeCore) -> None:
    """A marker is never the deployment that is billed, so even an explicit zero price stays out of its
    cost-map entry."""
    marker_id: Final = _unique()
    make_core(
        model_list=[
            deployment("small", "s"),
            deployment("big", "b"),
            complexity_marker("smart", marker_id, input_cost_per_token=0.0, output_cost_per_token=0.0),
        ]
    )

    entry: Final = litellm.model_cost[marker_id]
    assert "input_cost_per_token" not in entry
    assert "output_cost_per_token" not in entry


def test_replaying_cost_registrations_restores_the_entries_of_deployments_still_served(make_core: MakeCore) -> None:
    kept: Final = _unique()
    deleted: Final = _unique()
    core: Final = make_core(
        model_list=[
            deployment("gpt", kept, input_cost_per_token=0.25),
            deployment("gpt", deleted, input_cost_per_token=0.5),
        ]
    )
    core.apply(ChangeSet(delete=[deleted]))
    litellm.model_cost.pop(kept, None)
    litellm.model_cost.pop(deleted, None)

    core.replay_model_cost_registrations()  # pyright: ignore[reportAttributeAccessIssue]  # public on the core

    assert litellm.model_cost[kept]["input_cost_per_token"] == 0.25
    assert deleted not in litellm.model_cost


# --- deployment budgets --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deployment_budgets_from_the_model_list_are_registered_at_construction(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            deployment("gpt", "a", max_budget=1.0, budget_duration="1d"),
            deployment("gpt", "b", max_budget=2.0, budget_duration="1d"),
            deployment("gpt", "c"),
        ]
    )

    limiter: Final = core.registry().plugins("budget_limiter")
    assert limiter is not None
    assert limiter.deployment_budget_config is not None
    assert {key: config.max_budget for key, config in limiter.deployment_budget_config.items()} == {"a": 1.0, "b": 2.0}


@pytest.mark.asyncio
async def test_deleting_and_upserting_deployments_keep_the_budget_limiter_in_step(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            deployment("gpt", "a", max_budget=1.0, budget_duration="1d"),
            deployment("gpt", "b", max_budget=2.0, budget_duration="1d"),
        ]
    )
    limiter: Final = core.registry().plugins("budget_limiter")
    assert limiter is not None

    core.apply(ChangeSet(delete=["a"]))
    assert list(limiter.deployment_budget_config or {}) == ["b"]

    core.apply(ChangeSet(upsert=[Deployment(**deployment("gpt", "b"))]))  # pyright: ignore[reportArgumentType]
    assert limiter.deployment_budget_config is None

    core.apply(ChangeSet(upsert=[Deployment(**deployment("gpt", "b", max_budget=3.0, budget_duration="1d"))]))  # pyright: ignore[reportArgumentType]
    assert {key: config.max_budget for key, config in (limiter.deployment_budget_config or {}).items()} == {"b": 3.0}


@pytest.mark.asyncio
@pytest.mark.parametrize("params", [{"max_budget": 1.0}, {"budget_duration": "1d"}], ids=["no-duration", "no-budget"])
async def test_adding_a_deployment_with_half_a_budget_installs_no_budget_limiter(
    make_core: MakeCore, params: dict[str, object]
) -> None:
    core: Final = make_core(model_list=[deployment("gpt", "a")])

    core.add_deployment(Deployment(**deployment("gpt", "half", **params)))  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

    assert core.registry().plugins("budget_limiter") is None


# --- credentials with provider -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entry", "expected_provider"),
    [
        (deployment("gpt", "a", model="gpt-4o"), "openai"),
        (deployment("gpt", "a", model="gpt-4o", custom_llm_provider="azure", api_base="https://x"), "azure"),
        (deployment("gpt", "a", model="anthropic/claude-opus-4-6"), "anthropic"),
    ],
    ids=["bare-model-defaults-to-openai", "explicit-provider-wins", "provider-prefix"],
)
def test_credentials_with_provider_name_the_provider(
    make_core: MakeCore, entry: dict[str, object], expected_provider: str
) -> None:
    found: Final = make_core(model_list=[entry]).registry().credentials("a", form="with_provider")

    assert found is not None
    assert found["custom_llm_provider"] == expected_provider


def test_credentials_with_provider_drop_an_unknown_credential_name(make_core: MakeCore) -> None:
    found: Final = (
        make_core(model_list=[deployment("gpt", "a", litellm_credential_name="l11100-no-such-credential")])
        .registry()
        .credentials("a", form="with_provider")
    )

    assert found == {"api_key": "sk-test", "model": "openai/a", "custom_llm_provider": "openai"}


# --- router and deployment model info ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "model_info", "expected_key"),
    [
        ("azure/gpt-4o", {}, "azure/gpt-4o"),
        ("azure/my-deployment", {"base_model": "azure/gpt-4o"}, "azure/gpt-4o"),
    ],
    ids=["mapped-deployment-name", "base-model"],
)
def test_router_model_info_of_an_azure_deployment(
    make_core: MakeCore, model: str, model_info: dict[str, object], expected_key: str
) -> None:
    core: Final = make_core(
        model_list=[
            deployment("az", "az", model=model, api_base="https://x", api_version="2024", model_info=model_info)
        ]
    )

    info: Final = core.registry().router_model_info(None, "az", id="az")
    builtin: Final = litellm.get_model_info("azure/gpt-4o")
    assert info["key"] == expected_key
    assert info["max_input_tokens"] == builtin["max_input_tokens"]
    assert info["input_cost_per_token"] == builtin["input_cost_per_token"]


def test_router_model_info_of_an_unmapped_azure_deployment_is_its_zeroed_stub(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("az", "az", model="azure/l11100-unmapped", api_base="https://x", api_version="2024")]
    )

    info: Final = core.registry().router_model_info(None, "az", id="az")
    assert info["key"] == "azure/l11100-unmapped"
    assert info["max_input_tokens"] is None
    assert info["input_cost_per_token"] == 0


def test_deployment_model_info_takes_defaults_from_a_custom_base_model(make_core: MakeCore) -> None:
    """The deployment's own pricing wins; the rest comes from its ``base_model``'s built-in entry."""
    name: Final = _unique()
    core: Final = make_core(
        model_list=[
            deployment(
                "gpt", name, model=f"openai/{name}", input_cost_per_token=0.5, model_info={"base_model": "gpt-4o"}
            )
        ]
    )

    info: Final = core.registry().deployment_model_info(name, f"openai/{name}")
    builtin: Final = litellm.get_model_info("gpt-4o")
    assert info is not None
    assert info["input_cost_per_token"] == 0.5
    assert info["output_cost_per_token"] == builtin["output_cost_per_token"]
    assert info["max_input_tokens"] == builtin["max_input_tokens"]


def test_deployment_model_info_of_an_unpriced_unmapped_model_reports_no_prices(make_core: MakeCore) -> None:
    """``get_model_info`` reads a missing price as 0; the deployment reports ``None`` instead."""
    name: Final = _unique()
    core: Final = make_core(model_list=[deployment("gpt", name, model=f"openai/{name}")])

    info: Final = core.registry().deployment_model_info(name, f"openai/{name}")
    assert info is not None
    assert (info["input_cost_per_token"], info["output_cost_per_token"]) == (None, None)


# --- model group info ------------------------------------------------------------------------------------------


def test_group_info_takes_the_widest_limits_highest_prices_and_any_capability(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            deployment("g", "a", model="openai/gpt-3.5-turbo"),
            deployment("g", "b", model="openai/gpt-4o"),
            deployment("g", "c", model="openai/o3-mini"),
            deployment("g", "d", model="anthropic/claude-opus-4-6"),
        ]
    )

    info: Final = core.registry().group("g")
    opus: Final = litellm.get_model_info("anthropic/claude-opus-4-6")
    assert info is not None
    assert info.providers == ["openai", "anthropic"]
    assert (info.max_input_tokens, info.max_output_tokens) == (opus["max_input_tokens"], opus["max_output_tokens"])
    assert (info.input_cost_per_token, info.output_cost_per_token) == (
        opus["input_cost_per_token"],
        opus["output_cost_per_token"],
    )
    assert info.supports_vision is True
    assert info.supports_function_calling is True
    assert info.supports_parallel_function_calling is True
    assert info.supports_web_search is True
    assert info.supports_reasoning is True
    assert info.supports_url_context is False
    assert info.supports_fast_mode is False


def test_group_info_supports_url_context_when_any_deployment_does(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a", model="openai/gpt-4o"), deployment("g", "b", model="gemini/gemini-2.5-flash")]
    )

    info: Final = core.registry().group("g")
    assert info is not None
    assert info.supports_url_context is True


@pytest.mark.parametrize(
    ("models", "fast"),
    [
        (["anthropic/claude-opus-4-8"], True),
        (["anthropic/claude-opus-4-8", "anthropic/claude-opus-4-6"], False),
    ],
    ids=["every-deployment-fast", "one-not-fast"],
)
def test_group_info_supports_fast_mode_only_when_every_deployment_does(
    make_core: MakeCore, models: list[str], fast: bool
) -> None:
    core: Final = make_core(model_list=[deployment("g", f"d{i}", model=m) for i, m in enumerate(models)])

    info: Final = core.registry().group("g")
    assert info is not None
    assert info.supports_fast_mode is fast


@pytest.mark.parametrize(
    ("models", "efforts"),
    [
        (["anthropic/claude-opus-4-8"], ("none", "minimal", "low", "medium", "high", "xhigh", "max")),
        (
            ["anthropic/claude-opus-4-8", "anthropic/claude-opus-4-6"],
            ("none", "minimal", "low", "medium", "high", "max"),
        ),
        (["anthropic/claude-sonnet-4-5-20250929", "anthropic/claude-opus-4-6"], None),
        (["anthropic/claude-opus-4-6", "anthropic/claude-sonnet-4-5-20250929"], None),
        (["openai/gpt-4o", "openai/gpt-4o"], ()),
    ],
    ids=["one", "intersection", "unknown-first-stays-unknown", "unknown-later-clears", "none-supported"],
)
def test_group_info_reasoning_efforts_are_the_intersection_or_unknown(
    make_core: MakeCore, models: list[str], efforts: tuple[str, ...] | None
) -> None:
    core: Final = make_core(model_list=[deployment("g", f"d{i}", model=m) for i, m in enumerate(models)])

    info: Final = core.registry().group("g")
    assert info is not None
    assert info.supported_reasoning_efforts == efforts


def test_group_info_supported_endpoints_are_the_union_in_deployment_order(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            deployment("g", "a", model="openai/gpt-5"),
            deployment("g", "b", model="openai/gpt-4o", model_info={"supported_endpoints": ["/v1/embeddings"]}),
        ]
    )

    info: Final = core.registry().group("g")
    assert info is not None
    assert info.supported_endpoints == ("/v1/chat/completions", "/v1/batch", "/v1/responses", "/v1/embeddings")


def test_group_info_of_a_prompt_management_model_names_its_logger_as_provider(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("prompted", "p", model="langfuse/my-prompt")])

    info: Final = core.registry().group("prompted")
    assert info is not None
    assert info.providers == ["langfuse"]


def test_group_info_of_an_alias_to_neither_a_name_nor_a_dict_is_none(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("gpt", "a")])
    core.apply(ChangeSet(assign={"model_group_alias": {"odd": 5}}))

    assert core.registry().group("odd") is None


def test_group_info_of_an_unpriced_unmapped_model_has_no_prices_or_limits(make_core: MakeCore) -> None:
    name: Final = _unique()
    core: Final = make_core(model_list=[deployment("g", name, model=f"openai/{name}")])

    info: Final = core.registry().group("g")
    assert info is not None
    assert (info.input_cost_per_token, info.output_cost_per_token, info.max_input_tokens) == (None, None, None)


# --- lookups ---------------------------------------------------------------------------------------------------


def test_from_id_of_a_deployment_id_is_its_group_name(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("gpt", "a")])

    assert core.registry().resolve("a", kind="from_id") == "gpt"


def test_from_id_on_an_empty_core_or_of_an_empty_id_is_none(make_core: MakeCore) -> None:
    assert make_core(model_list=[]).registry().resolve("x", kind="from_id") is None
    assert make_core(model_list=[deployment("gpt", "a")]).registry().resolve("", kind="from_id") is None


def test_route_candidates_of_a_routing_group_are_its_members_deployments(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("h", "b"), deployment("other", "c")],
        routing_groups=[{"group_name": "rg", "models": ["g", "h"], "routing_strategy": "simple-shuffle"}],
    )

    assert core.registry().names("rg", kind="route_candidates") == frozenset({"a", "b"})


def test_search_tool_support_reads_model_info_then_the_anthropic_route(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            deployment("g", "a", model="anthropic/claude-opus-4-6"),
            deployment("g", "b", model="openai/gpt-4o", model_info={"supports_search_tool": False}),
            deployment("g", "c", model="openai/gpt-4o"),
        ]
    )

    assert core.registry().resolve("g", kind="search_tool_support") == (True, False, None)


# --- discovered model info (aapply refresh_model_info) ----------------------------------------------------

VLLM_CARDS: Final = {"data": [{"id": "my-model", "max_model_len": 8192}, {"id": "other", "max_model_len": 4096}]}


def _vllm(
    deployment_id: str = "v", **params: object
) -> dict[str, object]:  # mutable-ok: a model_list entry  # kwargs-ok: litellm_params
    return deployment(
        "vllm", deployment_id, model="hosted_vllm/my-model", api_base="http://vllm.local", api_key="sk-vllm", **params
    )


@pytest.mark.asyncio
async def test_a_refresh_reads_the_upstream_models_endpoint_with_the_deployments_key(make_core: MakeCore) -> None:
    endpoint: Final = ModelsEndpoint(VLLM_CARDS)
    core: Final = make_core(model_list=[_vllm(extra_headers={"X-Team": "a"})])

    await core.aapply(ChangeSet(refresh_model_info=True, refresh_client=endpoint))  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

    assert endpoint.requests == [("http://vllm.local/v1/models", {"authorization": "Bearer sk-vllm", "x-team": "a"})]
    assert dict(core.registry().discovered_model_info("v")) == {
        "max_tokens": 8192,
        "max_input_tokens": 8192,
        "max_output_tokens": 8192,
    }


@pytest.mark.asyncio
async def test_discovered_limits_never_override_configured_ones_and_feed_the_group_limits(make_core: MakeCore) -> None:
    endpoint: Final = ModelsEndpoint(VLLM_CARDS)
    core: Final = make_core(model_list=[_vllm(model_info={"max_output_tokens": 1000})])

    await core.aapply(ChangeSet(refresh_model_info=True, refresh_client=endpoint))  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

    registry: Final = core.registry()
    assert dict(registry.discovered_model_info("v")) == {"max_tokens": 8192, "max_input_tokens": 8192}
    assert registry.group("vllm", field="token_limits") == (8192, 1000)
    listing: Final = registry.group("vllm", field="listing_info")
    assert listing is not None
    assert (listing.max_input_tokens, listing.max_output_tokens) == (8192, 1000)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entry",
    [
        deployment("plain", "p", model="anthropic/claude-opus-4-6", api_base="http://vllm.local"),
        deployment("wild", "w", model="hosted_vllm/*", api_base="http://vllm.local"),
        deployment("nobase", "n", model="openai/gpt-4o"),
    ],
    ids=["provider-without-discovery", "wildcard-model", "no-api-base"],
)
async def test_a_refresh_skips_deployments_it_cannot_discover(make_core: MakeCore, entry: dict[str, object]) -> None:
    endpoint: Final = ModelsEndpoint(VLLM_CARDS)
    core: Final = make_core(model_list=[entry])

    await core.aapply(ChangeSet(refresh_model_info=True, refresh_client=endpoint))  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

    assert endpoint.requests == []


@pytest.mark.asyncio
async def test_an_unreachable_models_endpoint_discovers_nothing(make_core: MakeCore) -> None:
    endpoint: Final = ModelsEndpoint(error=ConnectionError("down"))
    core: Final = make_core(model_list=[_vllm()])

    await core.aapply(ChangeSet(refresh_model_info=True, refresh_client=endpoint))  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

    assert len(endpoint.requests) == 1
    assert dict(core.registry().discovered_model_info("v")) == {}


@pytest.mark.asyncio
async def test_discovered_limits_are_forgotten_when_the_deployment_is_replaced(make_core: MakeCore) -> None:
    endpoint: Final = ModelsEndpoint(VLLM_CARDS)
    core: Final = make_core(model_list=[_vllm()])
    await core.aapply(ChangeSet(refresh_model_info=True, refresh_client=endpoint))  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

    core.apply(ChangeSet(upsert=[Deployment(**_vllm(rpm=5))]))  # pyright: ignore[reportArgumentType]

    assert dict(core.registry().discovered_model_info("v")) == {}


# --- failed upserts and replays under ignore_invalid_deployments ---------------------------------------------


def test_a_failing_upsert_of_a_new_deployment_under_ignore_invalid_deployments_adds_nothing(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=[deployment("gpt", "a")], ignore_invalid_deployments=True)
    host.set_dynamic(provider_list=["anthropic"])

    result: Final = core.apply(ChangeSet(upsert=[Deployment(**deployment("gpt", "new"))]))  # pyright: ignore[reportArgumentType]

    assert list(result.upserted) == [None]
    assert model_ids(core) == ["a"]


def test_a_replay_retries_deployments_whose_provider_did_not_resolve_and_keeps_them_out_while_it_still_fails(
    make_core: MakeCore,
) -> None:
    core: Final = make_core(
        model_list=[deployment("gpt", "a"), deployment("odd", "u", model="l11100-no-provider/x")],
        ignore_invalid_deployments=True,
    )
    assert model_ids(core) == ["a"]

    core.replay_model_cost_registrations()  # pyright: ignore[reportAttributeAccessIssue]  # public on the core

    assert model_ids(core) == ["a"]


# --- group info without any model info ------------------------------------------------------------------------


def test_group_info_without_a_cost_map_entry_falls_back_to_the_configured_mode_and_costs(make_core: MakeCore) -> None:
    """With the entries construction registered withdrawn from the cost map (as a price data reload
    does before the replay) and a model litellm does not map, the group is described from the
    deployment's own model_info: its mode, its costs (a string cost read as a float, a missing one as
    None) and the provider's supported params, with no token limits."""
    name: Final = _unique()
    core: Final = make_core(
        model_list=[
            deployment(
                "g", name, model=f"openai/{name}", model_info={"mode": "embedding", "input_cost_per_token": "0.5"}
            )
        ]
    )
    litellm.model_cost.pop(name, None)
    litellm.model_cost.pop(f"openai/{name}", None)

    info: Final = core.registry().group("g")
    assert info is not None
    assert (info.mode, info.input_cost_per_token, info.output_cost_per_token) == ("embedding", 0.5, None)
    assert (info.max_input_tokens, info.max_output_tokens) == (None, None)
    assert info.supported_openai_params
