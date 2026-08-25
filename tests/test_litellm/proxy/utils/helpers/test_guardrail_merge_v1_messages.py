"""Reproduce issue #36085: model-level guardrails not applied on /v1/messages.

The bug: _merge_guardrails_with_existing always writes to data["metadata"],
but for /v1/messages the internal proxy state lives in data["litellm_metadata"].
This test checks whether model-level guardrails are findable by
get_guardrail_from_metadata after the merge, and whether they leak into
the Anthropic provider-facing metadata.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from litellm.proxy.utils import (
    _check_and_merge_model_level_guardrails,
    _merge_guardrails_with_existing,
)


def _router_with_deployment(guardrails, *, by_alias: bool = False):
    deployment = SimpleNamespace(litellm_params={"guardrails": guardrails})
    router = MagicMock()
    router.get_deployment.return_value = deployment
    router.get_model_list.return_value = (
        [{"litellm_params": {"guardrails": guardrails}}] if by_alias else []
    )
    return router


def _simulate_get_guardrail_from_metadata(data: dict) -> list:
    """Replicates CustomGuardrail.get_guardrail_from_metadata logic."""
    if "guardrails" in data:
        return data["guardrails"]
    for meta_key in ("metadata", "litellm_metadata"):
        meta = data.get(meta_key) or {}
        if isinstance(meta, dict) and "guardrails" in meta:
            return meta.get("guardrails") or []
    return []


def test_merge_writes_to_litellm_metadata_when_present():
    """For /v1/messages, litellm_metadata is the internal bucket.
    _merge_guardrails_with_existing should write there, not to metadata."""
    data = {
        "model": "my-model",
        "metadata": {"user_id": "user_abc"},  # provider-facing (Anthropic)
        "litellm_metadata": {"guardrails": ["key-guardrail"]},  # internal
    }
    result = _merge_guardrails_with_existing(data, ["model-guardrail"])

    # The guardrails should be in litellm_metadata, not metadata
    assert "guardrails" in result.get("litellm_metadata", {}), \
        "model-level guardrails should be merged into litellm_metadata for /v1/messages"
    assert "model-guardrail" in result["litellm_metadata"]["guardrails"]

    # metadata (provider-facing) should NOT have guardrails injected
    assert "guardrails" not in result.get("metadata", {}), \
        "guardrails should not leak into provider-facing metadata"


def test_check_and_merge_finds_guardrails_for_v1_messages():
    """Full _check_and_merge_model_level_guardrails with litellm_metadata.
    Simulates the pre_call path for /v1/messages."""
    router = _router_with_deployment(["pii-mask-presidio"], by_alias=True)
    data = {
        "model": "my-model",
        "metadata": {"user_id": "user_abc"},  # Anthropic provider-facing
        "litellm_metadata": {},  # internal, seeded by add_litellm_data_to_request
    }
    result = _check_and_merge_model_level_guardrails(
        data=data, llm_router=router, trust_client_model_info=False
    )

    # After merge, guardrails should be findable via get_guardrail_from_metadata
    requested = _simulate_get_guardrail_from_metadata(result)
    assert "pii-mask-presidio" in requested, \
        f"model-level guardrail should be in requested_guardrails, got: {requested}"


def test_merge_does_not_overwrite_provider_metadata():
    """Writing guardrails to data["metadata"] for /v1/messages would inject
    litellm-internal guardrails into the Anthropic provider-facing metadata.
    The fix must use litellm_metadata when present."""
    data = {
        "model": "my-model",
        "metadata": {"user_id": "user_abc"},
        "litellm_metadata": {},
    }
    result = _merge_guardrails_with_existing(data, ["pii-mask-presidio"])

    # Provider-facing metadata should be unchanged
    assert result["metadata"] == {"user_id": "user_abc"}, \
        f"provider-facing metadata should not be modified, got: {result['metadata']}"


def test_merge_preserves_existing_litellm_metadata_guardrails():
    """When litellm_metadata already has guardrails (from key/team metadata),
    the merge should union with them, not overwrite."""
    data = {
        "model": "my-model",
        "metadata": {"user_id": "user_abc"},
        "litellm_metadata": {"guardrails": ["key-guardrail"]},
    }
    result = _merge_guardrails_with_existing(data, ["model-guardrail"])

    litellm_meta_guardrails = result.get("litellm_metadata", {}).get("guardrails", [])
    assert "key-guardrail" in litellm_meta_guardrails, \
        "existing litellm_metadata guardrails should be preserved"
    assert "model-guardrail" in litellm_meta_guardrails, \
        "model-level guardrails should be merged into litellm_metadata"


def test_caller_metadata_guardrails_does_not_shadow_litellm_metadata():
    """veria-ai HIGH on #36085: a caller sending
    ``metadata: {"guardrails": []}`` on /v1/messages must not shadow the
    admin-authoritative guardrails merged into ``litellm_metadata``.

    ``get_guardrail_from_metadata`` checks ``metadata`` before
    ``litellm_metadata``. Without stripping ``guardrails`` from caller-supplied
    ``metadata``, the empty list would be returned and all model-level /
    key-team guardrails silently bypassed.

    This test simulates the post-strip state: ``metadata`` no longer has a
    ``guardrails`` key, so ``get_guardrail_from_metadata`` falls through to
    ``litellm_metadata`` where the admin-merged list lives.
    """
    data = {
        "model": "my-model",
        # Caller tried to inject guardrails: [] but it was stripped.
        "metadata": {"user_id": "user_abc"},
        "litellm_metadata": {"guardrails": ["pii-mask-presidio", "key-guardrail"]},
    }
    requested = _simulate_get_guardrail_from_metadata(data)
    assert "pii-mask-presidio" in requested, \
        f"admin guardrails must not be shadowed by caller metadata, got: {requested}"
    assert "key-guardrail" in requested, \
        f"key/team guardrails must not be shadowed by caller metadata, got: {requested}"


def test_caller_cannot_empty_list_bypass_guardrails():
    """Directly test the attack vector: caller sends
    ``metadata: {"guardrails": []}`` to bypass all guardrails.

    After the strip in ``add_litellm_data_to_request``, the ``guardrails`` key
    is removed from ``metadata``. The admin-merged guardrails in
    ``litellm_metadata`` are the only source ``get_guardrail_from_metadata``
    can read.
    """
    # Simulate the state AFTER add_litellm_data_to_request has stripped
    # guardrails from caller metadata AND move_guardrails_to_metadata +
    # _check_and_merge_model_level_guardrails have written admin guardrails
    # to litellm_metadata.
    data = {
        "model": "my-model",
        "metadata": {"user_id": "user_abc"},  # guardrails stripped by pre-call
        "litellm_metadata": {"guardrails": ["model-guardrail", "key-guardrail"]},
    }
    requested = _simulate_get_guardrail_from_metadata(data)
    assert len(requested) == 2, \
        f"expected 2 admin guardrails, got: {requested}"
    assert "model-guardrail" in requested
    assert "key-guardrail" in requested


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
