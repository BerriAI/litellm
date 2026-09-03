"""
Tests for the router-settings write path: ``POST /config/update`` ->
``UpdateRouterConfig`` -> ``Router.update_settings``.

A setting has to clear two independent gates to take effect, and a field
missing from either one is discarded behind a 200. ``retry_policy``
(LIT-3152) was the first field found stranded; ``max_fallbacks`` and
``enable_weighted_failover`` (LIT-5880) were the next two, so this file also
pins the three lists against each other rather than only the fields that have
been reported so far.

LIT-3152. The Admin UI Model Retry Settings tab posts a global ``retry_policy``
through ``POST /config/update`` -> ``UpdateRouterConfig`` ->
``Router.update_settings``. Both the pydantic schema and
``update_settings`` were dropping the field silently:

- ``UpdateRouterConfig`` had no ``retry_policy`` attribute, so
  ``model_dump(exclude_none=True)`` returned ``{}`` for that key.
- ``Router.update_settings`` had no ``"retry_policy"`` entry in
  ``_allowed_settings``, so even when fed directly the call was a no-op
  (``Setting {} is not allowed`` debug log).

The net effect was that after saving retry counts in the UI and
reloading, every value snapped back to ``defaultRetry = num_retries``
(2 by default), exactly matching the ticket repro.

This file pins both halves of the fix.
"""

import json
from dataclasses import dataclass
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

import litellm
from litellm.constants import RUNTIME_UPDATABLE_ROUTER_SETTINGS
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.router_utils.pre_call_checks.model_rate_limit_check import ModelRateLimitingCheck
from litellm.router_utils.pre_call_checks.prompt_caching_deployment_check import PromptCachingDeploymentCheck
from litellm.types.management_endpoints import ROUTER_SETTINGS_FIELDS
from litellm.types.router import RetryPolicy, UpdateRouterConfig


@pytest.fixture(autouse=True)
def isolate_litellm_callbacks():
    callbacks_before: Final = litellm.callbacks.copy()
    yield
    litellm.callbacks = callbacks_before  # test-quality-ok: required callback-state restoration fixture


# ---------------------------------------------------------------------------
# UpdateRouterConfig schema membership (LIT-3152 part 1)
# ---------------------------------------------------------------------------


CONSTRUCTOR_COUPLED_UI_SETTINGS = frozenset({"default_litellm_params", "set_verbose"})


def _writable_ui_settings() -> set[str]:
    return {field.field_name for field in ROUTER_SETTINGS_FIELDS} - CONSTRUCTOR_COUPLED_UI_SETTINGS


def test_every_admin_ui_router_setting_is_writable():
    """``POST /config/update`` parses ``router_settings`` through
    ``UpdateRouterConfig``, which ignores undeclared keys. A setting the Admin
    UI renders but the schema omits is accepted with a 200 and silently
    discarded, so the save appears to work and nothing changes."""
    assert _writable_ui_settings() <= set(UpdateRouterConfig.model_fields)


def test_every_admin_ui_router_setting_is_applied_at_runtime():
    """Clearing the schema is only half the trip: ``update_settings`` drops
    anything outside ``RUNTIME_UPDATABLE_ROUTER_SETTINGS``, which leaves the value in
    the config row while the live Router keeps serving the old one."""
    assert _writable_ui_settings() <= RUNTIME_UPDATABLE_ROUTER_SETTINGS


def test_constructor_coupled_settings_stay_out_of_the_writable_surface():
    """Router.__init__ does more than a plain attribute write for these two:
    default_litellm_params gets timeout / max_retries / caching_groups layered on
    (and caching_groups is never retained on the Router to replay), and set_verbose
    drives the process-global router logger, so turning it off would undo
    --detailed_debug. Until update_settings reproduces that, both gates have to
    keep rejecting them, and the UI has to keep rendering them so the gap stays
    visible."""
    assert not CONSTRUCTOR_COUPLED_UI_SETTINGS & set(UpdateRouterConfig.model_fields)
    assert not CONSTRUCTOR_COUPLED_UI_SETTINGS & RUNTIME_UPDATABLE_ROUTER_SETTINGS
    assert CONSTRUCTOR_COUPLED_UI_SETTINGS <= {field.field_name for field in ROUTER_SETTINGS_FIELDS}


def test_every_runtime_applicable_router_setting_is_writable():
    """The reverse direction: a setting the Router can apply is unreachable
    through the config API unless the schema declares it."""
    assert RUNTIME_UPDATABLE_ROUTER_SETTINGS <= set(UpdateRouterConfig.model_fields)


@pytest.mark.parametrize("value", [0, -1])
def test_update_router_config_rejects_an_unusable_parallel_request_limit(value):
    """A deployment's concurrency limiter is an asyncio.Semaphore built from
    this number. A negative one raises at construction, which turns every
    later request on every deployment into a 500, and zero reads as no limiter
    at all rather than as a block. Neither is usable, so both have to be
    refused at the edge instead of poisoning the live router."""
    with pytest.raises(ValidationError):
        UpdateRouterConfig(default_max_parallel_requests=value)


def test_update_router_config_exposes_retry_policy_field():
    """retry_policy must be a declared field on UpdateRouterConfig.

    Without it, Pydantic silently strips the key from the /config/update
    payload before the proxy even calls llm_router.update_settings.
    """
    assert "retry_policy" in UpdateRouterConfig.model_fields


def test_update_router_config_accepts_retry_policy_payload():
    """The exact payload the Admin UI Model Retry Settings tab sends must
    round-trip through the schema's ``dict(exclude_none=True)`` form, since
    that is what /config/update writes to the LiteLLM_Config row."""
    payload = {
        "retry_policy": {
            "BadRequestErrorRetries": 5,
            "RateLimitErrorRetries": 7,
            "TimeoutErrorRetries": 3,
        }
    }
    cfg = UpdateRouterConfig(**payload)
    dumped = cfg.model_dump(exclude_none=True)
    assert "retry_policy" in dumped
    assert dumped["retry_policy"]["BadRequestErrorRetries"] == 5
    assert dumped["retry_policy"]["RateLimitErrorRetries"] == 7
    assert dumped["retry_policy"]["TimeoutErrorRetries"] == 3


def test_update_router_config_rejects_malformed_retry_policy():
    """The field is typed as RetryPolicy, so /config/update validates the
    payload at the boundary and rejects non-numeric counts with a 422 instead
    of silently persisting garbage the apply path would later have to drop."""
    with pytest.raises(ValidationError):
        UpdateRouterConfig(retry_policy={"BadRequestErrorRetries": "not-an-int"})


def test_update_router_config_rejects_malformed_model_group_retry_policy():
    """model_group_retry_policy is Dict[str, RetryPolicy], so each per-group
    policy is validated the same way."""
    with pytest.raises(ValidationError):
        UpdateRouterConfig(
            model_group_retry_policy={"gpt-4": {"RateLimitErrorRetries": "x"}}
        )


# ---------------------------------------------------------------------------
# Router.update_settings retry_policy path (LIT-3152 part 2)
# ---------------------------------------------------------------------------


def _build_router() -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_key": "sk-fake",
                    "api_base": "http://localhost:9999",
                },
            }
        ]
    )


def test_update_settings_adds_optional_pre_call_check_once():
    router = _build_router()

    router.update_settings(num_retries=7, optional_pre_call_checks=["prompt_caching"])
    router.update_settings(optional_pre_call_checks=["prompt_caching"])

    prompt_caching_callbacks = [
        callback for callback in router.optional_callbacks if isinstance(callback, PromptCachingDeploymentCheck)
    ]
    assert len(prompt_caching_callbacks) == 1
    assert router.num_retries == 7


def test_update_settings_clears_omitted_toggleable_pre_call_checks():
    router = _build_router()

    router.update_settings(optional_pre_call_checks=["prompt_caching"])
    router.update_settings(optional_pre_call_checks=[])

    assert not any(isinstance(callback, PromptCachingDeploymentCheck) for callback in (router.optional_callbacks or []))
    assert not any(isinstance(callback, PromptCachingDeploymentCheck) for callback in litellm.callbacks)


def test_set_optional_pre_call_checks_reconciles_callback_types():
    router = _build_router()

    router.set_optional_pre_call_checks(["prompt_caching"])
    router.set_optional_pre_call_checks([])

    assert not any(isinstance(callback, PromptCachingDeploymentCheck) for callback in (router.optional_callbacks or []))
    assert not any(isinstance(callback, PromptCachingDeploymentCheck) for callback in litellm.callbacks)


def test_remove_optional_pre_call_check_removes_local_and_global_callbacks():
    router = _build_router()

    router.set_optional_pre_call_checks(["prompt_caching"])
    router._remove_optional_callbacks_of_type(PromptCachingDeploymentCheck)

    assert not any(type(callback) is PromptCachingDeploymentCheck for callback in (router.optional_callbacks or []))
    assert not any(type(callback) is PromptCachingDeploymentCheck for callback in litellm.callbacks)


def test_remove_optional_pre_call_check_keeps_global_callback_for_another_router():
    router_a = _build_router()
    router_b = _build_router()

    router_a.update_settings(optional_pre_call_checks=["prompt_caching"])
    router_b.update_settings(optional_pre_call_checks=["prompt_caching"])

    router_a.update_settings(optional_pre_call_checks=[])

    assert not any(type(callback) is PromptCachingDeploymentCheck for callback in (router_a.optional_callbacks or []))
    assert any(type(callback) is PromptCachingDeploymentCheck for callback in (router_b.optional_callbacks or []))
    assert any(type(callback) is PromptCachingDeploymentCheck for callback in litellm.callbacks)

    router_b.update_settings(optional_pre_call_checks=[])

    assert not any(type(callback) is PromptCachingDeploymentCheck for callback in (router_b.optional_callbacks or []))
    assert not any(type(callback) is PromptCachingDeploymentCheck for callback in litellm.callbacks)


def test_remove_optional_pre_call_check_keeps_global_callback_when_second_router_clears_first():
    router_a = _build_router()
    router_b = _build_router()

    router_a.update_settings(optional_pre_call_checks=["prompt_caching"])
    router_b.update_settings(optional_pre_call_checks=["prompt_caching"])

    router_b.update_settings(optional_pre_call_checks=[])

    assert any(type(callback) is PromptCachingDeploymentCheck for callback in (router_a.optional_callbacks or []))
    assert not any(type(callback) is PromptCachingDeploymentCheck for callback in (router_b.optional_callbacks or []))
    assert any(type(callback) is PromptCachingDeploymentCheck for callback in litellm.callbacks)

    router_a.update_settings(optional_pre_call_checks=[])

    assert not any(type(callback) is PromptCachingDeploymentCheck for callback in litellm.callbacks)


def test_update_settings_replaces_toggleable_pre_call_checks():
    router = _build_router()

    router.update_settings(optional_pre_call_checks=["prompt_caching"])
    router.update_settings(optional_pre_call_checks=["enforce_model_rate_limits"])

    assert not any(isinstance(callback, PromptCachingDeploymentCheck) for callback in (router.optional_callbacks or []))
    assert not any(isinstance(callback, PromptCachingDeploymentCheck) for callback in litellm.callbacks)
    assert any(isinstance(callback, ModelRateLimitingCheck) for callback in (router.optional_callbacks or []))


@pytest.mark.asyncio
async def test_update_settings_preserves_router_budget_limiting_when_omitted(monkeypatch):
    async def _disable_periodic_sync(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "litellm.router_strategy.budget_limiter.RouterBudgetLimiting.periodic_sync_in_memory_spend_with_redis",
        _disable_periodic_sync,
    )
    router = _build_router()

    router.add_optional_pre_call_checks(["router_budget_limiting"])
    router.update_settings(optional_pre_call_checks=[])

    assert any(isinstance(callback, RouterBudgetLimiting) for callback in (router.optional_callbacks or []))


def test_update_settings_persists_retry_policy_dict():
    """When the proxy's ``_add_router_settings_from_db_config`` calls
    ``llm_router.update_settings(retry_policy={...})`` after reading the
    DB row, the dict must land on ``self.retry_policy`` as a typed
    ``RetryPolicy`` (mirroring ``Router.__init__`` semantics)."""
    router = _build_router()
    assert router.retry_policy is None  # baseline

    router.update_settings(
        retry_policy={
            "BadRequestErrorRetries": 5,
            "RateLimitErrorRetries": 7,
            "TimeoutErrorRetries": 3,
        }
    )

    assert isinstance(router.retry_policy, RetryPolicy)
    assert router.retry_policy.BadRequestErrorRetries == 5
    assert router.retry_policy.RateLimitErrorRetries == 7
    assert router.retry_policy.TimeoutErrorRetries == 3


def test_update_settings_accepts_retry_policy_object_unchanged():
    """A pre-built ``RetryPolicy`` instance must pass through verbatim so
    callers that already constructed one (e.g. tests or programmatic
    callers) keep working."""
    router = _build_router()

    policy = RetryPolicy(BadRequestErrorRetries=2)
    router.update_settings(retry_policy=policy)

    assert router.retry_policy is policy


def test_update_settings_ignores_malformed_retry_policy():
    """A non-dict, non-``RetryPolicy`` value (e.g. a YAML typo like
    ``retry_policy: 5`` reaching ``update_settings``) must not land on
    ``self.retry_policy``. ``Router.__init__`` already drops such inputs;
    the update path must match so a malformed config can't store garbage
    that ``get_num_retries_from_retry_policy`` would only choke on at
    request time."""
    router = _build_router()

    existing = RetryPolicy(BadRequestErrorRetries=4)
    router.update_settings(retry_policy=existing)
    assert router.retry_policy is existing

    for bad_value in (5, "RateLimitErrorRetries=7", ["BadRequestErrorRetries"]):
        router.update_settings(retry_policy=bad_value)
        assert router.retry_policy is existing


def test_update_settings_get_settings_round_trip_for_retry_policy():
    """``GET /get/config/callbacks`` serializes ``llm_router.get_settings()``
    back to the UI. After updating, the round-trip must reflect the new
    values rather than the pre-update sentinel."""
    router = _build_router()
    pre = router.get_settings().get("retry_policy")
    assert pre is None

    router.update_settings(
        retry_policy={
            "BadRequestErrorRetries": 5,
            "RateLimitErrorRetries": 7,
        }
    )
    post = router.get_settings().get("retry_policy")
    assert post is not None
    assert post.BadRequestErrorRetries == 5
    assert post.RateLimitErrorRetries == 7


def test_update_settings_unrelated_kwargs_still_skipped():
    """Regression guard: the new branch must not relax the
    ``_allowed_settings`` allowlist for unrelated keys. An unknown
    setting should still be dropped silently as before."""
    router = _build_router()
    router.update_settings(this_is_not_a_router_setting=123)
    assert not hasattr(router, "this_is_not_a_router_setting")


# ---------------------------------------------------------------------------
# End-to-end persist -> apply -> read-back (LIT-3152 part 3)
#
# The tests above pin each layer in isolation, so they would all still pass
# if a regression flipped ``ConfigYAML.router_settings`` back to a loose
# ``dict`` (silently dropping retry_policy on the DB write) or if
# ``_add_router_settings_from_db_config`` stopped pushing the stored row onto
# the live router. This drives the real handler chain an Admin UI save
# triggers — ``POST /config/update`` writes the LiteLLM_Config row,
# ``add_deployment`` applies it to ``llm_router``, and
# ``GET /get/config/callbacks`` serializes it back — so the round trip is
# pinned, not just the pieces.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _FakeConfigRow:
    param_name: str
    param_value: dict


class _FakeConfigTable:
    """Stand-in for ``prisma_client.db.litellm_config``.

    Reproduces the one behavior the apply path relies on: a value written as a
    JSON string by ``/config/update`` reads back as a parsed dict, which is
    what ``_add_router_settings_from_db_config``'s
    ``isinstance(param_value, dict)`` branch requires to forward the settings.
    """

    def __init__(self):
        self.rows = {}

    async def find_first(self, where):
        return self.rows.get(where["param_name"])

    async def upsert(self, where, data):
        name = where["param_name"]
        raw = (data["update"] if name in self.rows else data["create"])["param_value"]
        self.rows[name] = _FakeConfigRow(name, json.loads(raw) if isinstance(raw, str) else raw)


@pytest.mark.asyncio
async def test_config_update_persists_and_reads_back_retry_policy(monkeypatch):
    """The exact global retry_policy save the UI performs must survive the
    real ``/config/update`` -> DB -> apply -> ``/get/config/callbacks`` path,
    not snap back to the ``num_retries`` fallback the ticket reported."""
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy._types import ConfigYAML, LitellmUserRoles, UserAPIKeyAuth

    router = _build_router()
    assert router.retry_policy is None

    fake_table = _FakeConfigTable()
    prisma_client = MagicMock()
    prisma_client.db.litellm_config = fake_table

    async def _apply_router_settings(*args, **kwargs):
        await proxy_server.proxy_config._add_router_settings_from_db_config(
            config_data={}, llm_router=router, prisma_client=prisma_client
        )

    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server.proxy_config, "add_deployment", _apply_router_settings)
    monkeypatch.setattr(proxy_server.proxy_config, "get_config", AsyncMock(return_value={}))

    posted = UpdateRouterConfig(
        retry_policy=RetryPolicy(
            BadRequestErrorRetries=5,
            TimeoutErrorRetries=3,
            RateLimitErrorRetries=7,
        )
    )
    request = MagicMock()
    request.json = AsyncMock(return_value={"router_settings": {"retry_policy": posted.model_dump()}})

    await proxy_server.update_config(
        config_info=ConfigYAML(router_settings=posted),
        request=request,
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-1234"),
    )

    persisted = fake_table.rows["router_settings"].param_value["retry_policy"]
    assert persisted == {
        "BadRequestErrorRetries": 5,
        "TimeoutErrorRetries": 3,
        "RateLimitErrorRetries": 7,
    }

    assert isinstance(router.retry_policy, RetryPolicy)
    assert router.retry_policy.RateLimitErrorRetries == 7

    read_back = (
        await proxy_server.get_config(
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-1234"
            )
        )
    )["router_settings"]["retry_policy"]
    assert read_back.BadRequestErrorRetries == 5
    assert read_back.TimeoutErrorRetries == 3
    assert read_back.RateLimitErrorRetries == 7


async def _post_router_settings(monkeypatch, router, table=None, **settings):
    """Drive one Admin UI save through the real chain: ``POST /config/update``
    writes the LiteLLM_Config row, ``add_deployment`` applies it to the router.

    Returns the stored row and the table, so a caller can chain a second save
    onto the first and assert on how the two merge.
    """
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy._types import ConfigYAML, LitellmUserRoles, UserAPIKeyAuth

    fake_table = table if table is not None else _FakeConfigTable()
    prisma_client = MagicMock()
    prisma_client.db.litellm_config = fake_table

    async def _apply_router_settings(*args, **kwargs):
        await proxy_server.proxy_config._add_router_settings_from_db_config(
            config_data={}, llm_router=router, prisma_client=prisma_client
        )

    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server.proxy_config, "add_deployment", _apply_router_settings)
    monkeypatch.setattr(proxy_server.proxy_config, "get_config", AsyncMock(return_value={}))

    request = MagicMock()
    request.json = AsyncMock(return_value={"router_settings": dict(settings)})

    await proxy_server.update_config(
        config_info=ConfigYAML(router_settings=UpdateRouterConfig(**settings)),
        request=request,
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-1234"),
    )
    return fake_table.rows["router_settings"].param_value, fake_table


@pytest.mark.asyncio
async def test_config_update_persists_and_applies_max_fallbacks(monkeypatch):
    """max_fallbacks was stranded at both gates the way retry_policy was: absent
    from the schema, so /config/update returned 200 and discarded it, and absent
    from the apply allowlist, so feeding it directly was a no-op too (LIT-5880)."""
    router = _build_router()
    assert router.max_fallbacks != 9

    persisted, _ = await _post_router_settings(monkeypatch, router, max_fallbacks=9)

    assert persisted["max_fallbacks"] == 9
    assert router.max_fallbacks == 9


@pytest.mark.asyncio
async def test_config_update_persists_and_applies_enable_weighted_failover(monkeypatch):
    """enable_weighted_failover was stranded at the schema gate only: the Router
    could always apply it, but no config API payload ever reached it."""
    router = _build_router()
    assert router.enable_weighted_failover is False

    persisted, _ = await _post_router_settings(monkeypatch, router, enable_weighted_failover=True)

    assert persisted["enable_weighted_failover"] is True
    assert router.enable_weighted_failover is True


@pytest.mark.asyncio
async def test_config_update_leaves_unsent_router_settings_alone(monkeypatch):
    """The handler merges the request over the stored row, so a field the
    caller never sent must not appear in the payload and clobber what is
    already there. ``model_group_alias`` defaulted to ``{}`` and did exactly
    that on every unrelated save."""
    router = _build_router()

    _, table = await _post_router_settings(monkeypatch, router, model_group_alias={"gpt-4": "gpt-4o"})
    persisted, _ = await _post_router_settings(monkeypatch, router, num_retries=4, table=table)

    assert persisted["model_group_alias"] == {"gpt-4": "gpt-4o"}
    assert router.model_group_alias == {"gpt-4": "gpt-4o"}


def test_update_settings_rebuilds_max_parallel_request_clients():
    """The concurrency limiter is a semaphore cached per deployment on first
    use, so storing a new default_max_parallel_requests is not enough: without
    dropping the cached ones, every deployment already serving traffic keeps
    the old limit until the process restarts."""
    router = _build_router()
    deployment = router.model_list[0]

    router.update_settings(default_max_parallel_requests=2)
    assert router._get_client(deployment=deployment, kwargs={}, client_type="max_parallel_requests")._value == 2

    router.update_settings(default_max_parallel_requests=5)
    assert router._get_client(deployment=deployment, kwargs={}, client_type="max_parallel_requests")._value == 5


def test_max_parallel_requests_cache_key_addresses_the_cached_semaphore():
    """The read site and the write site have to agree on the key, or clearing
    the cache silently misses and every deployment keeps its old limit. Pinning
    the helper against what the cache actually holds is what makes the two
    sites one contract rather than two matching string literals."""
    router = _build_router()
    deployment = router.model_list[0]
    router.update_settings(default_max_parallel_requests=3)

    semaphore = router._get_client(deployment=deployment, kwargs={}, client_type="max_parallel_requests")
    key = router.max_parallel_requests_cache_key(deployment["model_info"]["id"])
    assert router.cache.get_cache(key=key, local_only=True) is semaphore

    router._clear_max_parallel_requests_clients()
    assert router.cache.get_cache(key=key, local_only=True) is None


@pytest.mark.parametrize("setting", ["timeout", "retry_after", "num_retries", "max_fallbacks"])
def test_update_settings_ignores_a_null_numeric_setting(setting):
    """A ``router_settings: {timeout: null}`` in config.yaml reaches
    update_settings as a real None. Storing it replaces an int the request path
    does arithmetic on, which surfaces much later as a TypeError inside an
    unrelated call. The stored value has to survive instead."""
    router = _build_router()
    before = getattr(router, setting)
    assert before is not None

    router.update_settings(**{setting: None})

    assert getattr(router, setting) == before
