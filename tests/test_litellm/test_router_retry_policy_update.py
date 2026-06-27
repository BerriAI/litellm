"""
Tests for the retry_policy fix on Router.update_settings (LIT-3152).

Bug: the Admin UI Model Retry Settings tab posts a global ``retry_policy``
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

import os
import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.types.router import RetryPolicy, UpdateRouterConfig

# ---------------------------------------------------------------------------
# UpdateRouterConfig schema membership (LIT-3152 part 1)
# ---------------------------------------------------------------------------


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
