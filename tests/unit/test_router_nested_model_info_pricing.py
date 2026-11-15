import logging

import pytest

import litellm
from litellm import Router
from litellm.router_api import ChangeSet
from litellm.router_rust.surface.registry import _model_info_nested_under_litellm_params
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

BACKEND_MODEL = "deepinfra/deepseek-ai/DeepSeek-V4-Flash-0731"
PUBLIC_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
INPUT_COST = 7.2e-08
OUTPUT_COST = 1.44e-07
CACHE_READ_COST = 1.44e-08
DEPLOYMENT_ID = "db-deployment-id"


def _pricing_model_info():
    return {
        "input_cost_per_token": INPUT_COST,
        "output_cost_per_token": OUTPUT_COST,
        "cache_read_input_token_cost": CACHE_READ_COST,
    }


def _nested_config():
    """The reporter's config shape: model_info nested inside litellm_params."""
    return {
        "model_name": PUBLIC_MODEL,
        "litellm_params": {
            "model": BACKEND_MODEL,
            "api_key": "fake-key",
            "extra_body": {"service_tier": "flex"},
            "model_info": _pricing_model_info(),
        },
    }


def _deployment_level_config():
    return {
        "model_name": PUBLIC_MODEL,
        "litellm_params": {
            "model": BACKEND_MODEL,
            "api_key": "fake-key",
            "extra_body": {"service_tier": "flex"},
        },
        "model_info": _pricing_model_info(),
    }


@pytest.fixture(autouse=True)
def clean_cost_map():
    """Keep registrations from bleeding across tests, and put back the entries the cost map already had."""
    from litellm.utils import _invalidate_model_cost_lowercase_map

    cost_map = litellm.model_cost
    keys = (BACKEND_MODEL, PUBLIC_MODEL, DEPLOYMENT_ID)
    original = {key: cost_map[key] for key in keys if key in cost_map}

    def _clear():
        for key in keys:
            cost_map.pop(key, None)
        _invalidate_model_cost_lowercase_map()

    _clear()
    yield
    _clear()
    cost_map.update(original)
    _invalidate_model_cost_lowercase_map()


def _deployment_pricing(router):
    deployment = router.model_list[0]
    return deployment.get("model_info", {})


def _nested_deployment(model_info):
    """A deployment built the way the proxy loads one from the DB, nested block included."""
    config = _nested_config()
    return Deployment(
        model_name=config["model_name"],
        litellm_params=LiteLLM_Params.model_validate(config["litellm_params"]),
        model_info=model_info,
    )


def test_promotes_nothing_when_no_nested_block():
    """No nested block, and a non-dict nested value, both promote nothing."""
    assert _model_info_nested_under_litellm_params({"model": BACKEND_MODEL}, {}) == {}
    assert _model_info_nested_under_litellm_params({"model_info": None}, {}) == {}
    assert _model_info_nested_under_litellm_params({"model_info": "nope"}, {}) == {}


def test_promotes_only_fields_the_deployment_does_not_define():
    """Deployment-level values win; only unset fields are promoted."""
    promoted = _model_info_nested_under_litellm_params(
        {"model_info": {"input_cost_per_token": 1.0, "output_cost_per_token": 2.0}},
        {"input_cost_per_token": INPUT_COST},
    )

    assert promoted == {"output_cost_per_token": 2.0}


def test_never_promotes_id_and_skips_none_values():
    """`id` is reserved for the generated deployment id, and None is not a value."""
    promoted = _model_info_nested_under_litellm_params(
        {
            "model_info": {
                "id": "nested-id",
                "output_cost_per_token": None,
                "input_cost_per_token": INPUT_COST,
            }
        },
        {},
    )

    assert promoted == {"input_cost_per_token": INPUT_COST}


def test_promoted_mapping_is_read_only():
    """The caller updates its own dict; the returned view must not be mutable."""
    promoted = _model_info_nested_under_litellm_params({"model_info": {"input_cost_per_token": INPUT_COST}}, {})

    with pytest.raises(TypeError):
        promoted["input_cost_per_token"] = 0.0  # type: ignore[index]


def test_nested_model_info_pricing_is_applied():
    """
    Regression test for https://github.com/BerriAI/litellm/issues/35691

    `model_info` is a declared field on GenericLiteLLMParams, so nesting it under
    `litellm_params` validates cleanly and is then silently ignored by every cost
    path, leaving the deployment with no pricing at all.
    """
    router = Router(model_list=[_nested_config()])
    model_info = _deployment_pricing(router)

    assert model_info.get("input_cost_per_token") == INPUT_COST
    assert model_info.get("output_cost_per_token") == OUTPUT_COST
    assert model_info.get("cache_read_input_token_cost") == CACHE_READ_COST


def test_nested_matches_deployment_level():
    """Both spellings must end up with the same pricing."""
    nested = _deployment_pricing(Router(model_list=[_nested_config()]))
    proper = _deployment_pricing(Router(model_list=[_deployment_level_config()]))

    for field in (
        "input_cost_per_token",
        "output_cost_per_token",
        "cache_read_input_token_cost",
    ):
        assert nested.get(field) == proper.get(field)


def test_deployment_level_model_info_wins_over_nested():
    """An explicit deployment-level value is not overwritten by a nested one."""
    config = _deployment_level_config()
    config["litellm_params"]["model_info"] = {"input_cost_per_token": 999.0}

    model_info = _deployment_pricing(Router(model_list=[config]))
    assert model_info.get("input_cost_per_token") == INPUT_COST


def test_nested_model_info_does_not_override_deployment_id():
    """The generated deployment id must survive a nested `id`."""
    config = _nested_config()
    config["litellm_params"]["model_info"]["id"] = "nested-id-should-be-ignored"

    model_info = _deployment_pricing(Router(model_list=[config]))
    assert model_info.get("id") != "nested-id-should-be-ignored"


def test_no_nested_model_info_is_unchanged():
    """Configs without the nested block keep working exactly as before."""
    model_info = _deployment_pricing(Router(model_list=[_deployment_level_config()]))
    assert model_info.get("input_cost_per_token") == INPUT_COST


@pytest.mark.asyncio
async def test_nested_model_info_produces_non_zero_response_cost():
    """
    End to end: the persisted spend is derived from `response_cost`, so pricing that
    never reaches the cost map shows up as spend 0 in LiteLLM_SpendLogs.
    """
    router = Router(model_list=[_nested_config()])

    response = await router.acompletion(
        model=PUBLIC_MODEL,
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hello",
    )

    response_cost = response._hidden_params.get("response_cost")
    assert response_cost is not None
    assert response_cost > 0


@pytest.mark.asyncio
async def test_add_deployment_applies_nested_model_info_pricing():
    """
    DB-loaded and /model/new deployments reach the router through add_deployment,
    not _create_deployment, so the nested block has to be applied there as well.
    """
    router = Router(model_list=[])
    assert router.add_deployment(_nested_deployment({"id": DEPLOYMENT_ID})) is not None

    model_info = _deployment_pricing(router)
    assert model_info.get("input_cost_per_token") == INPUT_COST
    assert model_info.get("output_cost_per_token") == OUTPUT_COST
    assert model_info.get("cache_read_input_token_cost") == CACHE_READ_COST

    # Exact, not just > 0: the shipped cost map also prices this backend model.
    response = await router.acompletion(
        model=PUBLIC_MODEL,
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hello",
    )
    usage = response.usage
    assert response._hidden_params.get("response_cost") == pytest.approx(
        usage.prompt_tokens * INPUT_COST + usage.completion_tokens * OUTPUT_COST
    )


def test_add_deployment_keeps_deployment_level_values_and_id():
    """Only unset fields are applied, and the nested `id` never re-keys the deployment."""
    deployment = _nested_deployment({"id": DEPLOYMENT_ID, "input_cost_per_token": INPUT_COST})
    deployment.litellm_params.model_info.update(id="nested-id-should-be-ignored", input_cost_per_token=999.0)

    router = Router(model_list=[])
    router.add_deployment(deployment)

    model_info = _deployment_pricing(router)
    assert model_info.get("input_cost_per_token") == INPUT_COST
    assert model_info.get("output_cost_per_token") == OUTPUT_COST
    assert model_info.get("id") == DEPLOYMENT_ID
    assert router.has_model_id(DEPLOYMENT_ID)
    assert not router.has_model_id("nested-id-should-be-ignored")


PROTECTED_MODEL_INFO = {"id": DEPLOYMENT_ID, "team_id": "team-a", "blocked": True, "db_model": True}
NESTED_INPUT_COST = 9.9e-07
ABOVE_1HR_CACHE_WRITE = "cache_creation_input_token_cost_above_1hr"
NO_PRICING = dict.fromkeys(_pricing_model_info())


def _overwriting_dunder_dict():
    """A nested `__dict__` holding a complete, valid ModelInfo with every protected field changed."""
    return {"__dict__": ModelInfo(id="attacker-id", team_id="team-b", blocked=False, db_model=False).model_dump()}


def _protected_fields(model_info):
    return {key: model_info.get(key) for key in PROTECTED_MODEL_INFO}


NESTED_NON_PRICING = {
    "team_id": "team-b",
    "blocked": False,
    "db_model": True,
    "created_by": "someone-else",
    "access_groups": ["other-group"],
    "access_via_team_ids": ["team-b"],
    "team_public_model_name": "other-public-name",
    "member_auto_router": True,
    "ptu_count": 5,
    "mode": "embedding",
    "base_model": "azure/gpt-4o",
    "my_custom_key": "custom",
    **_overwriting_dunder_dict(),
    "self": "x",
    "MAX_PTU_COUNT": 5,
}


def _config_router(nested, deployment_info=None, params=None):
    """Router(model_list=...) with one deployment whose litellm_params nest `nested`."""
    config = _nested_config()
    config["model_info"] = {"id": DEPLOYMENT_ID, **(deployment_info or {})}
    config["litellm_params"].update(params or {}, model_info=nested)
    return Router(model_list=[config])


def _added_router(nested, deployment_info=None, params=None):
    """Router.add_deployment with the same deployment."""
    config = _nested_config()
    router = Router(model_list=[])
    router.add_deployment(
        Deployment(
            model_name=PUBLIC_MODEL,
            litellm_params=LiteLLM_Params.model_validate(
                {**config["litellm_params"], **(params or {}), "model_info": nested}
            ),
            model_info={"id": DEPLOYMENT_ID, **(deployment_info or {})},
        )
    )
    return router


INSTALLS = pytest.mark.parametrize("install", (_config_router, _added_router), ids=("router_init", "add_deployment"))


def _db_deployment(api_key, nested=None):
    """A team model as a proxy pod loads it from the DB, paused by a proxy admin."""
    litellm_params = {"model": BACKEND_MODEL, "api_key": api_key}
    return Deployment(
        model_name=PUBLIC_MODEL,
        litellm_params=LiteLLM_Params.model_validate(
            litellm_params if nested is None else {**litellm_params, "model_info": nested}
        ),
        model_info=dict(PROTECTED_MODEL_INFO),
    )


def _updated_router(nested):
    """A pod's router after one DB reload installs the row and the next installs an update to it
    that rotates its key and nests `nested`, upserted as the proxy upserts DB rows."""
    router = Router(model_list=[], ignore_invalid_deployments=True)
    router.apply(ChangeSet(upsert=[_db_deployment("key-old")]))
    router.apply(ChangeSet(upsert=[_db_deployment("key-new", nested)]))
    return router


def _served(router):
    return router.registry().deployment(DEPLOYMENT_ID)


def _served_pricing(router):
    model_info = _served(router).model_info
    return {field: model_info.get(field) for field in NO_PRICING}


def _nested_warnings(caplog):
    return tuple(
        record.getMessage()
        for record in caplog.records
        if record.name == "LiteLLM Router" and "nested inside 'litellm_params'" in record.getMessage()
    )


@pytest.mark.parametrize("key", NESTED_NON_PRICING)
def test_promotes_only_mirrored_pricing_fields(key):
    """Only the pricing fields Deployment mirrors onto model_info leave a nested block."""
    promoted = _model_info_nested_under_litellm_params(
        {"model_info": {**_pricing_model_info(), key: NESTED_NON_PRICING[key]}}, {}
    )

    assert promoted == _pricing_model_info()


@pytest.mark.parametrize("key", NESTED_NON_PRICING)
def test_router_init_ignores_nested_non_pricing_fields(key):
    """Every other nested field is ignored: the deployment serves what its pricing alone gives it."""
    expected = _served(_config_router(_pricing_model_info())).model_info.model_dump()

    served = _served(_config_router({**_pricing_model_info(), key: NESTED_NON_PRICING[key]}))
    assert served.model_info.model_dump() == expected


@pytest.mark.parametrize("key", [key for key in NESTED_NON_PRICING if key not in ("db_model", "member_auto_router")])
def test_add_deployment_ignores_nested_non_pricing_fields(key):
    """As on the config path. A built Deployment always sets db_model and member_auto_router, so no nested value
    could fill them, and they are left out here."""
    expected = _served(_added_router(_pricing_model_info())).model_info.model_dump()

    served = _served(_added_router({**_pricing_model_info(), key: NESTED_NON_PRICING[key]}))
    assert served.model_info.model_dump() == expected


def test_add_deployment_nested_dunder_dict_keeps_protected_fields():
    """A nested `__dict__` must not replace id, team_id, blocked or db_model of a DB-loaded deployment."""
    deployment = _nested_deployment(dict(PROTECTED_MODEL_INFO))
    deployment.litellm_params.model_info.update(_overwriting_dunder_dict())

    router = Router(model_list=[])
    router.add_deployment(deployment)

    model_info = _deployment_pricing(router)
    assert _protected_fields(model_info) == PROTECTED_MODEL_INFO
    assert model_info.get("input_cost_per_token") == INPUT_COST
    assert router.registry().has_deployment(DEPLOYMENT_ID)
    assert not router.registry().has_deployment("attacker-id")


def test_update_with_nested_dunder_dict_keeps_the_admin_block():
    """A nested `__dict__` in a model update: the paused row stays the only copy, still blocked, keyed and owned
    as the DB says, and the update itself applies."""
    router = _updated_router({**_pricing_model_info(), **_overwriting_dunder_dict()})

    assert [dict(entry["model_info"]).get("blocked") for entry in router.registry().all()] == [True]
    served = _served(router)
    assert _protected_fields(served.model_info) == PROTECTED_MODEL_INFO
    assert served.litellm_params.api_key == "key-new"


INVALID_NESTED_PRICING = {
    "a string": ("input_cost_per_token", "not-a-number"),
    "a list": ("output_cost_per_token", [OUTPUT_COST]),
    "a dict": ("cache_read_input_token_cost", {"per_token": CACHE_READ_COST}),
}


def _warned_about_invalid(caplog, field, value):
    (warning,) = _nested_warnings(caplog)
    return f"model={PUBLIC_MODEL}:" in warning and field in warning and str(value) not in warning


@INSTALLS
@pytest.mark.parametrize(("field", "value"), INVALID_NESTED_PRICING.values(), ids=INVALID_NESTED_PRICING)
def test_invalid_nested_pricing_is_ignored_with_a_warning(install, field, value, caplog):
    """A nested value that does not validate never stops the deployment loading. Nothing from the block applies,
    and the warning names the deployment and the field, not the value."""
    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        router = install({**_pricing_model_info(), field: value})

    assert _served_pricing(router) == NO_PRICING
    assert _warned_about_invalid(caplog, field, value)


@pytest.mark.parametrize(("field", "value"), INVALID_NESTED_PRICING.values(), ids=INVALID_NESTED_PRICING)
def test_update_with_invalid_nested_pricing_still_takes_effect(field, value, caplog):
    """An update that also nests invalid pricing still rotates the key."""
    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        router = _updated_router({**_pricing_model_info(), field: value})

    assert _served(router).litellm_params.api_key == "key-new"
    assert _served_pricing(router) == NO_PRICING
    assert _warned_about_invalid(caplog, field, value)


WRONG_SHAPE_TIERS = {
    "a dict": {"range": [0, 1000], "input_cost_per_token": INPUT_COST},
    "a list of numbers": [INPUT_COST, OUTPUT_COST],
}


@INSTALLS
@pytest.mark.parametrize("tiers", WRONG_SHAPE_TIERS.values(), ids=WRONG_SHAPE_TIERS)
def test_nested_tiered_pricing_of_the_wrong_shape_is_ignored(install, tiers):
    router = install({**_pricing_model_info(), "tiered_pricing": tiers})

    assert _served(router).model_info.tiered_pricing is None
    assert _served_pricing(router) == NO_PRICING


@INSTALLS
@pytest.mark.parametrize("cost", ("inf", "nan", "1e400", float("inf")), ids=repr)
def test_non_finite_nested_cost_is_never_applied(install, cost):
    """Refused where the router's finite-cost rule reads the nested value, and never applied where that rule
    does not look."""
    with pytest.raises(ValueError, match="cannot be served"):
        install({"input_cost_per_token": cost})
    install(_pricing_model_info())
    expected = litellm.model_cost[DEPLOYMENT_ID].get(ABOVE_1HR_CACHE_WRITE)

    router = install({**_pricing_model_info(), ABOVE_1HR_CACHE_WRITE: cost})
    assert _served(router).model_info.get(ABOVE_1HR_CACHE_WRITE) is None
    assert litellm.model_cost[DEPLOYMENT_ID].get(ABOVE_1HR_CACHE_WRITE) == expected


PRECEDENCE = {
    "a deployment-level value wins": ({"input_cost_per_token": INPUT_COST}, {}, INPUT_COST, ("output_cost_per_token",)),
    "a litellm_params value wins": ({}, {"input_cost_per_token": INPUT_COST}, INPUT_COST, ("output_cost_per_token",)),
    "an explicit None is filled": (
        {"input_cost_per_token": None},
        {},
        NESTED_INPUT_COST,
        ("input_cost_per_token", "output_cost_per_token"),
    ),
}


@INSTALLS
@pytest.mark.parametrize(("deployment_info", "params", "input_cost", "applied"), PRECEDENCE.values(), ids=PRECEDENCE)
def test_nested_pricing_fills_only_fields_nothing_else_sets(
    install, deployment_info, params, input_cost, applied, caplog
):
    """The warning names exactly the fields it applies."""
    nested = {"input_cost_per_token": NESTED_INPUT_COST, "output_cost_per_token": OUTPUT_COST}
    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        router = install(nested, deployment_info, params)

    model_info = _served(router).model_info
    assert model_info.input_cost_per_token == input_cost
    assert model_info.output_cost_per_token == OUTPUT_COST
    (warning,) = _nested_warnings(caplog)
    assert tuple(field for field in nested if field in warning) == applied


RESERVED_CAPACITY = {
    "team_id": "team-a",
    "ptu_count": 2,
    "cost_per_ptu_per_hour": 1.0,
    "ptu_effective_from": "2026-01-01T00:00:00Z",
}


def test_nested_pricing_is_not_applied_under_reserved_capacity(monkeypatch, caplog):
    """A config deployment billed for reserved capacity serves its token pricing at zero, which outranks a nested
    block, so nothing from the block applies and no warning says it does."""
    monkeypatch.setenv("LITELLM_ENABLE_PTU_COST_ATTRIBUTION", "true")
    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        router = _config_router(_pricing_model_info(), RESERVED_CAPACITY)

    assert _served_pricing(router) == dict.fromkeys(NO_PRICING, 0.0)
    assert _nested_warnings(caplog) == ()
