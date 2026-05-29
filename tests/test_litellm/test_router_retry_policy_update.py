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
    dumped = cfg.dict(exclude_none=True)
    assert "retry_policy" in dumped
    assert dumped["retry_policy"]["BadRequestErrorRetries"] == 5
    assert dumped["retry_policy"]["RateLimitErrorRetries"] == 7
    assert dumped["retry_policy"]["TimeoutErrorRetries"] == 3


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
