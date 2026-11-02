"""
How the core builds its registry from ``model_list``: deployment ids, provider checks, environment
gating, ``os.environ/`` references, per-organization fan-out, wildcard and team registration, and
the pricing each deployment writes into ``litellm.model_cost``.
"""

import pytest

import litellm
from litellm.router_api import deployment_id
from tests.unit.test_router.core.registry_helpers import MakeCore, ids, team_deployment, unique_id
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment


def _bare(
    model_name: str, model: str, **params: object
) -> dict[str, object]:  # mutable-ok: a model_list entry  # kwargs-ok: litellm_params
    """A model_list entry with no ``model_info`` at all."""
    return {"model_name": model_name, "litellm_params": {"model": model, "api_key": "sk-test", **params}}


# -- deployment ids -----------------------------------------------------------------------------


def test_a_deployment_without_an_id_gets_the_stable_hash_of_its_group_and_params(make_core: MakeCore) -> None:
    core = make_core(model_list=[_bare("gpt", "openai/gpt-4o")])

    expected = deployment_id("gpt", {"model": "openai/gpt-4o", "api_key": "sk-test"})
    assert core.registry().names(kind="ids") == [expected]
    assert core.registry().has_deployment(expected)


def test_generated_ids_are_the_same_across_cores_built_from_the_same_list(make_core: MakeCore) -> None:
    first = make_core(model_list=[_bare("gpt", "openai/gpt-4o")])
    second = make_core(model_list=[_bare("gpt", "openai/gpt-4o")])

    assert first.registry().names(kind="ids") == second.registry().names(kind="ids")


def test_generated_ids_differ_when_the_group_or_params_differ(make_core: MakeCore) -> None:
    core = make_core(
        model_list=[_bare("gpt", "openai/gpt-4o"), _bare("other", "openai/gpt-4o"), _bare("gpt", "openai/gpt-4o-mini")]
    )

    assert len(set(core.registry().names(kind="ids"))) == 3


def test_a_declared_non_string_id_is_kept_as_a_string(make_core: MakeCore) -> None:
    core = make_core(
        model_list=[
            {"model_name": "gpt", "litellm_params": {"model": "openai/x", "api_key": "k"}, "model_info": {"id": 1}}
        ]
    )

    assert core.registry().names(kind="ids") == ["1"]
    found = core.registry().deployment("1")
    assert found is not None
    assert found.model_info.id == "1"


def test_two_deployments_declaring_one_id_are_both_kept_and_the_id_finds_the_last(make_core: MakeCore) -> None:
    core = make_core(model_list=[deployment("gpt", "d"), deployment("emb", "d")])

    assert [entry["model_name"] for entry in core.registry().all()] == ["gpt", "emb"]
    assert core.registry().index("model_name") == {"gpt": [0], "emb": [1]}
    found = core.registry().deployment("d")
    assert found is not None
    assert found.model_name == "emb"


# -- the stored entries -------------------------------------------------------------------------


def test_the_stored_entry_carries_litellm_param_defaults_and_model_info_flags(make_core: MakeCore) -> None:
    core = make_core(model_list=[deployment("gpt", "a")])

    (entry,) = core.registry().all()
    assert entry["model_name"] == "gpt"
    assert entry["litellm_params"]["model"] == "openai/a"
    assert entry["litellm_params"]["api_key"] == "sk-test"
    assert entry["model_info"] == {"id": "a", "db_model": False, "member_auto_router": False}


def test_top_level_rpm_and_tpm_are_copied_into_litellm_params(make_core: MakeCore) -> None:
    core = make_core(
        model_list=[
            {
                "model_name": "gpt",
                "litellm_params": {"model": "openai/x", "api_key": "k"},
                "model_info": {"id": "r"},
                "rpm": 7,
                "tpm": 9,
            }
        ]
    )

    (entry,) = core.registry().all()
    assert (entry["litellm_params"]["rpm"], entry["litellm_params"]["tpm"]) == (7, 9)
    assert (entry["rpm"], entry["tpm"]) == (7, 9)


def test_os_environ_references_in_litellm_params_are_resolved(
    make_core: MakeCore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REGISTRY_CHAR_BASE", "https://env.example")
    monkeypatch.setenv("REGISTRY_CHAR_KEY", "sk-from-env")
    core = make_core(
        model_list=[
            deployment("gpt", "a", api_key="os.environ/REGISTRY_CHAR_KEY", api_base="os.environ/REGISTRY_CHAR_BASE")
        ]
    )

    params = core.registry().all()[0]["litellm_params"]
    assert (params["api_key"], params["api_base"]) == ("sk-from-env", "https://env.example")


@pytest.mark.asyncio
async def test_resolved_environment_credentials_are_sent_to_the_provider(
    make_core: MakeCore, host: ScriptedHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REGISTRY_CHAR_BASE", "https://env.example")
    core = make_core(model_list=[deployment("gpt", "a", api_base="os.environ/REGISTRY_CHAR_BASE")])

    await core.acompletion(model="gpt", messages=[{"role": "user", "content": "hi"}])  # pyright: ignore[reportAttributeAccessIssue]  # an endpoint method

    assert [call.kwargs.get("api_base") for call in host.calls_to("acompletion")] == ["https://env.example"]


def test_a_list_of_organizations_fans_out_into_one_deployment_per_organization(make_core: MakeCore) -> None:
    core = make_core(model_list=[_bare("gpt", "openai/x", organization=["org-1", "org-2"])])

    entries = core.registry().all()
    assert [entry["litellm_params"]["organization"] for entry in entries] == ["org-1", "org-2"]
    assert len(set(ids(entries))) == 2
    assert core.registry().index("model_name") == {"gpt": [0, 1]}
    for entry in entries:  # pyright: ignore[reportAny]  # untyped in the checker's view
        found = core.registry().deployment(entry["model_info"]["id"])  # pyright: ignore[reportAny]  # untyped in the checker's view
        assert found is not None
        assert found.litellm_params.organization == entry["litellm_params"]["organization"]


def test_organization_deployments_get_the_same_ids_on_every_build(make_core: MakeCore) -> None:
    first = make_core(model_list=[_bare("gpt", "openai/x", organization=["org-1", "org-2"])])
    second = make_core(model_list=[_bare("gpt", "openai/x", organization=["org-1", "org-2"])])

    assert ids(first.registry().all()) == ids(second.registry().all())


def test_a_declared_id_stays_on_the_first_organization_and_is_suffixed_on_the_rest(make_core: MakeCore) -> None:
    entry = _bare("gpt", "openai/x", organization=["org-1", "org-2"])
    entry["model_info"] = {"id": "declared"}
    core = make_core(model_list=[entry])

    assert ids(core.registry().all()) == ["declared", "declared-org-2"]


def test_deployment_names_list_each_deployments_litellm_model(make_core: MakeCore) -> None:
    core = make_core(model_list=[deployment("gpt", "a"), deployment("gpt", "b"), deployment("emb", "e")])

    assert core.registry().names(kind="deployment_names") == ["openai/a", "openai/b", "openai/e"]


def test_every_configured_litellm_model_starts_with_zero_latency(make_core: MakeCore) -> None:
    core = make_core(model_list=[deployment("gpt", "a"), deployment("gpt", "b")])

    assert core.health().latency == {"openai/a": 0, "openai/b": 0}


@pytest.mark.parametrize("model_list", [None, []], ids=["none", "empty"])
def test_a_core_without_deployments_has_an_empty_registry(make_core: MakeCore, model_list: object) -> None:
    core = make_core(model_list=model_list)

    assert core.registry().all() == []
    assert core.registry().names(kind="ids") == []
    assert core.registry().names(kind="group_set") == set()
    assert core.registry().deployments() == []


# -- invalid deployments ------------------------------------------------------------------------


def test_an_unknown_provider_raises_bad_request_at_construction(make_core: MakeCore) -> None:
    with pytest.raises(litellm.BadRequestError, match="LLM Provider NOT provided"):
        make_core(model_list=[deployment("gpt", "a", model="notaprovider/model")])


def test_ignore_invalid_deployments_drops_the_invalid_one_and_keeps_the_rest(make_core: MakeCore) -> None:
    core = make_core(
        model_list=[deployment("gpt", "a", model="notaprovider/model"), deployment("gpt", "b")],
        ignore_invalid_deployments=True,
    )

    assert core.registry().names(kind="ids") == ["b"]


def test_an_invalid_tag_regex_raises_value_error_naming_the_model(make_core: MakeCore) -> None:
    with pytest.raises(ValueError, match=r"Invalid regex in tag_regex for model 'gpt': '\('"):
        make_core(model_list=[deployment("gpt", "a", tag_regex=["("])])


def test_ignore_invalid_deployments_drops_a_deployment_with_an_invalid_tag_regex(make_core: MakeCore) -> None:
    core = make_core(
        model_list=[deployment("gpt", "a", tag_regex=["("]), deployment("gpt", "b", tag_regex=["^ok$"])],
        ignore_invalid_deployments=True,
    )

    assert core.registry().names(kind="ids") == ["b"]


# -- environment-gated deployments --------------------------------------------------------------


def test_a_deployment_for_another_environment_is_left_out(make_core: MakeCore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_ENVIRONMENT", "staging")
    core = make_core(
        model_list=[
            deployment("prod-only", "a", model_info={"supported_environments": ["production"]}),
            deployment("both", "b", model_info={"supported_environments": ["staging", "production"]}),
        ]
    )

    assert core.registry().names(kind="ids") == ["b"]
    assert core.registry().deployments("prod-only") == []


def test_a_left_out_environment_deployment_still_names_its_group(
    make_core: MakeCore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Odd but current: the group set is read from the given model_list, not from what was added."""
    monkeypatch.setenv("LITELLM_ENVIRONMENT", "staging")
    core = make_core(
        model_list=[
            deployment("prod-only", "a", model_info={"supported_environments": ["production"]}),
            deployment("h", "b"),
        ]
    )

    assert core.registry().names(kind="group_set") == {"prod-only", "h"}
    assert core.registry().index("model_name") == {"h": [0]}


def test_supported_environments_without_litellm_environment_raises(
    make_core: MakeCore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LITELLM_ENVIRONMENT", raising=False)

    with pytest.raises(ValueError, match="Set 'supported_environments' for model but not 'LITELLM_ENVIRONMENT' set"):
        make_core(model_list=[deployment("gpt", "a", model_info={"supported_environments": ["production"]})])


@pytest.mark.parametrize(
    ("environment", "supported", "message"),
    [
        ("bogus", ["production"], "LITELLM_ENVIRONMENT must be one of"),
        ("staging", ["prod"], "supported_environments must be one of"),
        ("staging", "production", "supported_environments must be a list of"),
    ],
    ids=["unknown_environment", "unknown_supported_value", "not_a_list"],
)
def test_invalid_environment_configuration_raises(
    make_core: MakeCore, monkeypatch: pytest.MonkeyPatch, environment: str, supported: object, message: str
) -> None:
    monkeypatch.setenv("LITELLM_ENVIRONMENT", environment)

    with pytest.raises(ValueError, match=message):
        make_core(model_list=[deployment("gpt", "a", model_info={"supported_environments": supported})])


def test_ignore_invalid_deployments_drops_an_invalid_environment_deployment(
    make_core: MakeCore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LITELLM_ENVIRONMENT", raising=False)
    core = make_core(
        model_list=[
            deployment("gpt", "a", model_info={"supported_environments": ["production"]}),
            deployment("gpt", "b"),
        ],
        ignore_invalid_deployments=True,
    )

    assert core.registry().names(kind="ids") == ["b"]


@pytest.mark.parametrize(
    ("model_info", "active"),
    [
        ({}, True),
        ({"supported_environments": ["development"]}, True),
        ({"supported_environments": ["production"]}, False),
    ],
    ids=["unset", "matching", "other"],
)
def test_deployment_is_active_for_environment_follows_litellm_environment(
    make_core: MakeCore, monkeypatch: pytest.MonkeyPatch, model_info: dict[str, object], active: bool
) -> None:
    core = make_core(model_list=[deployment("gpt", "a")])
    monkeypatch.setenv("LITELLM_ENVIRONMENT", "development")
    candidate = litellm.types.router.Deployment(**deployment("gpt", "z", model_info=model_info))  # pyright: ignore[reportArgumentType]  # a model_list entry

    assert core.deployment_is_active_for_environment(candidate) is active  # pyright: ignore[reportAttributeAccessIssue]  # public on the core


# -- wildcard and team registration -------------------------------------------------------------


def test_a_wildcard_group_registers_a_pattern_and_a_provider_default_deployment(make_core: MakeCore) -> None:
    core = make_core(
        model_list=[
            deployment("openai/*", "w", model="openai/*"),
            deployment("anthropic/*", "x", model="anthropic/*"),
            deployment("gpt", "a"),
        ]
    )

    assert core.settings().provider_default_deployment_ids == ["w", "x"]
    assert ids(core.registry().deployments("openai/gpt-4o")) == ["w"]
    assert ids(core.registry().deployments("anthropic/claude-x")) == ["x"]


def test_team_deployments_are_indexed_by_team_and_public_name(make_core: MakeCore) -> None:
    core = make_core(
        model_list=[
            team_deployment("internal-1", "t1", "team-a", "pub"),
            team_deployment("internal-2", "t2", "team-a", "pub"),
            team_deployment("internal-3", "t3", "team-b", "other"),
            deployment("gpt", "g"),
        ]
    )

    assert core.registry().index("team_model") == {("team-a", "pub"): [0, 1], ("team-b", "other"): [2]}
    assert core.registry().names(kind="team_public") == frozenset({"pub", "other"})
    assert core.registry().names(kind="team_public", team_id="team-a") == frozenset({"pub"})
    assert core.registry().names(kind="team_public", team_id="nobody") == frozenset()


def test_a_team_deployment_needs_both_team_id_and_public_name_to_be_team_indexed(make_core: MakeCore) -> None:
    core = make_core(
        model_list=[
            deployment("gpt", "a", model_info={"team_id": "team-a"}),
            deployment("gpt", "b", model_info={"team_public_model_name": "pub"}),
        ]
    )

    assert core.registry().index("team_model") == {}
    assert core.registry().names(kind="team_public") == frozenset()


# -- pricing written into litellm.model_cost -----------------------------------------------------


def test_custom_pricing_is_registered_under_the_deployment_id(make_core: MakeCore) -> None:
    model_id = unique_id()
    make_core(
        model_list=[
            deployment("gpt", model_id, model="openai/gpt-4o", input_cost_per_token=1e-3, output_cost_per_token=2e-3)
        ]
    )

    entry = litellm.model_cost[model_id]
    assert (entry["input_cost_per_token"], entry["output_cost_per_token"]) == (1e-3, 2e-3)


def test_custom_input_pricing_inherits_the_backend_models_cache_read_price(make_core: MakeCore) -> None:
    model_id = unique_id()
    make_core(model_list=[deployment("gpt", model_id, model="openai/gpt-4o", input_cost_per_token=1e-3)])

    builtin = litellm.get_model_info("openai/gpt-4o")
    assert litellm.model_cost[model_id]["cache_read_input_token_cost"] == builtin["cache_read_input_token_cost"]


def test_a_user_set_cache_read_price_is_not_overwritten_by_inheritance(make_core: MakeCore) -> None:
    model_id = unique_id()
    make_core(
        model_list=[
            deployment(
                "gpt", model_id, model="openai/gpt-4o", input_cost_per_token=1e-3, cache_read_input_token_cost=7e-9
            )
        ]
    )

    assert litellm.model_cost[model_id]["cache_read_input_token_cost"] == 7e-9


def test_a_deployment_without_custom_pricing_registers_no_prices_under_its_id(make_core: MakeCore) -> None:
    model_id = unique_id()
    make_core(model_list=[deployment("gpt", model_id, model="openai/gpt-4o")])

    entry = litellm.model_cost[model_id]
    assert entry.get("input_cost_per_token") is None
    assert entry.get("output_cost_per_token") is None


def test_model_info_fields_are_registered_under_the_deployment_id(make_core: MakeCore) -> None:
    model_id = unique_id()
    make_core(model_list=[deployment("gpt", model_id, model_info={"max_input_tokens": 1000, "mode": "embedding"})])

    entry = litellm.model_cost[model_id]
    assert (entry["max_input_tokens"], entry["mode"]) == (1000, "embedding")
