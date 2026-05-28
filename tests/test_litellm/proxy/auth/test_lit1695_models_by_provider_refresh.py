"""Regression tests for LIT-1695.

Bug: `litellm.models_by_provider` was built at import time as a dict whose
values include set *unions* (e.g.
``open_ai_chat_completion_models | open_ai_text_completion_models``). The
``|`` operator returns a NEW set, frozen at evaluation time. When the
underlying per-provider sets were later mutated by ``add_known_models`` (called
by the proxy ``/reload/model_cost_map`` admin endpoint) or ``register_model``,
those union snapshots became stale.

Customer-visible effect: a wildcard route like ``openai/*`` configured on the
proxy, plus a newly added model entry in ``model_prices_and_context_window.json``,
did not surface that new model in the hub/playground drop-down list — even
after the user clicked "Refresh model price listing".

Fix: refresh ``models_by_provider`` from the underlying sets at the end of
``add_known_models`` and ``register_model`` so wildcard expansion via
``get_provider_models`` / ``get_valid_models`` reflects the new state.
"""
import pytest

import litellm
from litellm.proxy.auth.model_checks import (
    get_complete_model_list,
    get_known_models_from_wildcard,
    get_provider_models,
)


# Use clearly-fictional keys so they cannot collide with the bundled
# model_prices_and_context_window.json or any other test fixture.
_NEW_OPENAI = "openai/lit1695-fictional-openai-zz"
_NEW_ANTHROPIC = "claude-lit1695-fictional-anthropic-zz"
_NEW_VERTEX = "vertex_ai/lit1695-fictional-vertex-zz"


@pytest.fixture
def _restore_provider_state():
    """Snapshot + restore per-provider sets and model_cost so individual
    tests cannot leak into each other (or into the rest of the suite)."""
    snapshots = {
        "open_ai_chat_completion_models": set(litellm.open_ai_chat_completion_models),
        "open_ai_text_completion_models": set(litellm.open_ai_text_completion_models),
        "anthropic_models": set(litellm.anthropic_models),
        "vertex_chat_models": set(litellm.vertex_chat_models),
        "vertex_text_models": set(litellm.vertex_text_models),
        "model_cost": dict(litellm.model_cost),
        "models_by_provider": {
            k: set(v) if isinstance(v, set) else v
            for k, v in litellm.models_by_provider.items()
        },
    }
    try:
        yield
    finally:
        litellm.open_ai_chat_completion_models.clear()
        litellm.open_ai_chat_completion_models.update(
            snapshots["open_ai_chat_completion_models"]
        )
        litellm.open_ai_text_completion_models.clear()
        litellm.open_ai_text_completion_models.update(
            snapshots["open_ai_text_completion_models"]
        )
        litellm.anthropic_models.clear()
        litellm.anthropic_models.update(snapshots["anthropic_models"])
        litellm.vertex_chat_models.clear()
        litellm.vertex_chat_models.update(snapshots["vertex_chat_models"])
        litellm.vertex_text_models.clear()
        litellm.vertex_text_models.update(snapshots["vertex_text_models"])
        litellm.model_cost.clear()
        litellm.model_cost.update(snapshots["model_cost"])
        # Rebuild the merged map from the now-restored per-provider sets.
        litellm._refresh_models_by_provider()


def test_helper_is_exposed():
    """The refresh helper must be on the public ``litellm`` namespace so the
    proxy ``/reload/model_cost_map`` endpoint (and ``register_model``) can call
    it across process boundaries / module imports."""
    assert callable(litellm._refresh_models_by_provider)


def test_register_model_propagates_openai_union_to_models_by_provider(
    _restore_provider_state,
):
    """``models_by_provider['openai']`` is an eager union of two underlying
    sets — the original LIT-1695 bug surface. After ``register_model``, the
    new key must appear there."""
    assert _NEW_OPENAI not in litellm.open_ai_chat_completion_models
    assert _NEW_OPENAI not in litellm.models_by_provider["openai"]

    litellm.register_model(
        {
            _NEW_OPENAI: {
                "max_tokens": 1024,
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
                "litellm_provider": "openai",
                "mode": "chat",
            }
        }
    )

    assert _NEW_OPENAI in litellm.open_ai_chat_completion_models
    assert _NEW_OPENAI in litellm.models_by_provider["openai"], (
        "register_model must refresh models_by_provider for union-backed providers"
    )


def test_add_known_models_propagates_to_union_provider(_restore_provider_state):
    """``add_known_models`` is what the proxy ``/reload/model_cost_map`` admin
    endpoint calls after refetching the price map. It must also propagate to
    ``models_by_provider`` for union-backed entries (here ``vertex_ai``)."""
    assert _NEW_VERTEX not in litellm.vertex_chat_models
    assert _NEW_VERTEX not in litellm.models_by_provider["vertex_ai"]

    litellm.add_known_models(
        {
            _NEW_VERTEX: {
                "max_tokens": 1024,
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
                "litellm_provider": "vertex_ai-chat-models",
                "mode": "chat",
            }
        }
    )

    assert _NEW_VERTEX in litellm.vertex_chat_models
    assert _NEW_VERTEX in litellm.models_by_provider["vertex_ai"], (
        "add_known_models must refresh union-backed entries in models_by_provider"
    )


def test_wildcard_expansion_sees_newly_registered_model(_restore_provider_state):
    """End-to-end through the same path the UI hub/playground dropdown uses:
    ``get_known_models_from_wildcard('openai/*')`` -> ``get_provider_models`` ->
    ``get_valid_models`` -> ``litellm.models_by_provider['openai']``."""
    pre = get_known_models_from_wildcard("openai/*")
    assert _NEW_OPENAI not in pre

    litellm.register_model(
        {
            _NEW_OPENAI: {
                "max_tokens": 1024,
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
                "litellm_provider": "openai",
                "mode": "chat",
            }
        }
    )

    post = get_known_models_from_wildcard("openai/*")
    assert _NEW_OPENAI in post, (
        "Wildcard expansion for openai/* must include models registered after "
        "module init — this is the LIT-1695 customer-visible regression."
    )


def test_complete_model_list_surfaces_new_model_for_wildcard_route(
    _restore_provider_state,
):
    """The Models / Playground page renders ``get_complete_model_list`` output
    when the proxy ``model_list`` contains a wildcard like ``openai/*``."""
    before = get_complete_model_list(
        key_models=[],
        team_models=[],
        proxy_model_list=["openai/*"],
        user_model=None,
        infer_model_from_keys=None,
        return_wildcard_routes=False,
        llm_router=None,
        model_access_groups={},
    )
    assert _NEW_OPENAI not in before

    litellm.register_model(
        {
            _NEW_OPENAI: {
                "max_tokens": 1024,
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
                "litellm_provider": "openai",
                "mode": "chat",
            }
        }
    )

    after = get_complete_model_list(
        key_models=[],
        team_models=[],
        proxy_model_list=["openai/*"],
        user_model=None,
        infer_model_from_keys=None,
        return_wildcard_routes=False,
        llm_router=None,
        model_access_groups={},
    )
    assert _NEW_OPENAI in after


def test_direct_reference_provider_still_works(_restore_provider_state):
    """``models_by_provider['anthropic']`` is a direct reference to
    ``anthropic_models`` (not a union), so mutations were ALREADY visible
    pre-fix. Guard against accidentally regressing that path by replacing the
    reference with a stale copy."""
    assert _NEW_ANTHROPIC not in litellm.models_by_provider["anthropic"]

    litellm.register_model(
        {
            _NEW_ANTHROPIC: {
                "max_tokens": 1024,
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
                "litellm_provider": "anthropic",
                "mode": "chat",
            }
        }
    )

    assert _NEW_ANTHROPIC in litellm.anthropic_models
    assert _NEW_ANTHROPIC in litellm.models_by_provider["anthropic"]


def test_refresh_helper_is_idempotent(_restore_provider_state):
    """Calling the helper multiple times in a row must not duplicate, lose, or
    permute entries."""
    before = {
        k: set(v) if isinstance(v, set) else v
        for k, v in litellm.models_by_provider.items()
    }
    litellm._refresh_models_by_provider()
    litellm._refresh_models_by_provider()
    after = {
        k: set(v) if isinstance(v, set) else v
        for k, v in litellm.models_by_provider.items()
    }
    assert before.keys() == after.keys()
    for k in before:
        assert before[k] == after[k], f"refresh changed contents for provider {k}"
