"""Tests for credential resolution (api_key, app_id) helpers.

These helpers are pure functions (no SDK dependency), so tests call them
directly with explicit args instead of constructing a guardrail instance.
"""

import pytest

from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.credentials import (
    get_metadata,
    resolve_api_key,
    resolve_app_id,
)
from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.exceptions import (
    WonderFenceMissingSecrets,
)


def _data(**overrides):
    """Build a request_data dict with default admin-pinned app_id."""
    metadata = overrides.pop("metadata", None)
    if metadata is None:
        metadata = {"user_api_key_metadata": {"alice_wonderfence_app_id": "test-app"}}
    base = {"model": "gpt-4", "metadata": metadata}
    base.update(overrides)
    return base


# ----------------------------- app_id resolution -----------------------------


def test_resolve_app_id_from_request_metadata_requires_override_flag():
    data = _data(metadata={"alice_wonderfence_app_id": "from-req"})
    assert resolve_app_id(data, allow_request_metadata_override=True) == "from-req"


def test_resolve_app_id_request_metadata_ignored_when_override_disabled():
    """Request metadata is caller-controlled and must not satisfy app_id when
    the override flag is off — otherwise a user could bypass admin-pinned
    credentials by sending their own app_id in the request body."""
    data = _data(metadata={"alice_wonderfence_app_id": "from-req"})
    with pytest.raises(WonderFenceMissingSecrets, match="alice_wonderfence_app_id"):
        resolve_app_id(data, allow_request_metadata_override=False)


def test_resolve_app_id_from_key_metadata():
    data = _data(
        metadata={
            "user_api_key_metadata": {"alice_wonderfence_app_id": "from-key"},
        }
    )
    assert resolve_app_id(data, allow_request_metadata_override=False) == "from-key"


def test_resolve_app_id_from_team_metadata():
    data = _data(
        metadata={
            "user_api_key_team_metadata": {"alice_wonderfence_app_id": "from-team"},
        }
    )
    assert resolve_app_id(data, allow_request_metadata_override=False) == "from-team"


def test_resolve_app_id_key_beats_request_even_when_override_enabled():
    """With the override flag on, request metadata is still only a last-resort
    source — admin-pinned key metadata wins."""
    data = _data(
        metadata={
            "alice_wonderfence_app_id": "from-req",
            "user_api_key_metadata": {"alice_wonderfence_app_id": "from-key"},
            "user_api_key_team_metadata": {"alice_wonderfence_app_id": "from-team"},
        }
    )
    assert resolve_app_id(data, allow_request_metadata_override=True) == "from-key"


def test_resolve_app_id_team_beats_request_when_override_enabled():
    """Team metadata beats request metadata even with the override flag on."""
    data = _data(
        metadata={
            "alice_wonderfence_app_id": "from-req",
            "user_api_key_team_metadata": {"alice_wonderfence_app_id": "from-team"},
        }
    )
    assert resolve_app_id(data, allow_request_metadata_override=True) == "from-team"


def test_resolve_app_id_priority_key_over_team():
    data = _data(
        metadata={
            "user_api_key_metadata": {"alice_wonderfence_app_id": "from-key"},
            "user_api_key_team_metadata": {"alice_wonderfence_app_id": "from-team"},
        }
    )
    assert resolve_app_id(data, allow_request_metadata_override=False) == "from-key"


def test_resolve_app_id_missing_raises():
    data = _data(metadata={})
    with pytest.raises(WonderFenceMissingSecrets, match="alice_wonderfence_app_id"):
        resolve_app_id(data, allow_request_metadata_override=False)


# ----------------------------- api_key resolution -----------------------------


def test_resolve_api_key_from_request_metadata_requires_override_flag():
    data = _data(metadata={"alice_wonderfence_api_key": "from-req"})
    assert (
        resolve_api_key(
            data, default_api_key="default", allow_request_metadata_override=True
        )
        == "from-req"
    )


def test_resolve_api_key_request_metadata_ignored_when_override_disabled():
    """With override off, a caller-supplied api_key must not be honored;
    falls back to the configured default instead."""
    data = _data(metadata={"alice_wonderfence_api_key": "from-req"})
    assert (
        resolve_api_key(
            data, default_api_key="default", allow_request_metadata_override=False
        )
        == "default"
    )


def test_resolve_api_key_key_beats_request_even_when_override_enabled():
    """Admin-pinned key metadata wins over request metadata even with the
    override flag enabled."""
    data = _data(
        metadata={
            "alice_wonderfence_api_key": "from-req",
            "user_api_key_metadata": {"alice_wonderfence_api_key": "from-key"},
        }
    )
    assert (
        resolve_api_key(
            data, default_api_key="default", allow_request_metadata_override=True
        )
        == "from-key"
    )


def test_resolve_api_key_from_key_metadata():
    data = _data(
        metadata={
            "user_api_key_metadata": {"alice_wonderfence_api_key": "from-key"},
        }
    )
    assert (
        resolve_api_key(
            data, default_api_key="default", allow_request_metadata_override=False
        )
        == "from-key"
    )


def test_resolve_api_key_from_team_metadata():
    data = _data(
        metadata={
            "user_api_key_team_metadata": {"alice_wonderfence_api_key": "from-team"},
        }
    )
    assert (
        resolve_api_key(
            data, default_api_key="default", allow_request_metadata_override=False
        )
        == "from-team"
    )


def test_resolve_api_key_falls_back_to_default():
    data = _data(metadata={})
    assert (
        resolve_api_key(
            data, default_api_key="default-key", allow_request_metadata_override=False
        )
        == "default-key"
    )


def test_resolve_api_key_missing_everywhere_raises():
    data = _data(metadata={})
    with pytest.raises(WonderFenceMissingSecrets):
        resolve_api_key(
            data, default_api_key=None, allow_request_metadata_override=False
        )


# ----------------------------- metadata fallback -----------------------------


def test_resolve_reads_litellm_metadata_when_metadata_absent():
    """``get_metadata`` falls back to ``litellm_metadata`` when ``metadata``
    is missing. Use admin-controlled key metadata so it resolves without
    needing the request-override flag."""
    data = {
        "model": "gpt-4",
        "litellm_metadata": {
            "user_api_key_metadata": {"alice_wonderfence_app_id": "from-litellm-md"}
        },
    }
    assert (
        resolve_app_id(data, allow_request_metadata_override=False) == "from-litellm-md"
    )


def test_get_metadata_merges_with_litellm_metadata_winning():
    """When both buckets are present, merge them with proxy-injected
    ``litellm_metadata`` winning on key collision; caller-only keys survive."""
    data = {
        "metadata": {
            "alice_wonderfence_app_id": "caller-only",
            "shared_key": "from-caller",
        },
        "litellm_metadata": {
            "shared_key": "from-litellm",
            "user_api_key_metadata": {"alice_wonderfence_app_id": "admin-pinned"},
        },
    }
    assert get_metadata(data) == {
        "alice_wonderfence_app_id": "caller-only",
        "shared_key": "from-litellm",
        "user_api_key_metadata": {"alice_wonderfence_app_id": "admin-pinned"},
    }


def test_get_metadata_returns_empty_when_both_absent():
    assert get_metadata({}) == {}


def test_responses_route_admin_pin_beats_caller_metadata():
    """Mirror the /v1/responses shape: caller `metadata` carries a
    request-override app_id while the admin pin lives in
    `litellm_metadata.user_api_key_metadata`. The admin pin must win even with
    the override flag enabled — the caller bucket must not shadow it."""
    data = {
        "model": "gpt-4",
        "metadata": {"alice_wonderfence_app_id": "caller-override"},
        "litellm_metadata": {
            "user_api_key_metadata": {"alice_wonderfence_app_id": "admin-pinned"}
        },
    }
    assert resolve_app_id(data, allow_request_metadata_override=True) == "admin-pinned"


def test_responses_route_admin_pin_beats_caller_metadata_api_key():
    """api_key variant of the /v1/responses regression: admin-pinned key
    metadata wins over a caller-supplied request-override api_key."""
    data = {
        "model": "gpt-4",
        "metadata": {"alice_wonderfence_api_key": "caller-override"},
        "litellm_metadata": {
            "user_api_key_metadata": {"alice_wonderfence_api_key": "admin-pinned"}
        },
    }
    assert (
        resolve_api_key(
            data, default_api_key="default", allow_request_metadata_override=True
        )
        == "admin-pinned"
    )
