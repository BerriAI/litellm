"""Tests for the Router-level wiring of the adaptive router.

Specifically guards the four bugs found when wiring the example config
`auto_router/adaptive_router` end-to-end:

1. The `auto_router/adaptive_router` model prefix must NOT trigger the
   semantic auto-router init path (which would crash on missing fields).
2. The same prefix MUST trigger the adaptive-router init path.
3. `init_adaptive_router_deployment` must read `input_cost_per_token`
   from `litellm_params` (where users put it), not just `model_info`.
4. `Router.async_pre_routing_hook` must dispatch to the matching entry in
   `self.adaptive_routers` when the inbound model matches a configured
   adaptive-router name, returning the underlying model the bandit picked.
"""

from unittest.mock import AsyncMock

import pytest

from litellm import Router
from litellm.types.router import LiteLLM_Params, RequestType


def _params(**overrides):
    base = {"model": "auto_router/adaptive_router"}
    base.update(overrides)
    return LiteLLM_Params(**base)


# ---- Fix 1 & 2: opt-in prefix routing -----------------------------------


def test_auto_router_check_excludes_adaptive_router_prefix():
    r = Router(model_list=[])
    assert (
        r._is_auto_router_deployment(
            litellm_params=_params(model="auto_router/adaptive_router")
        )
        is False
    )


def test_auto_router_check_excludes_complexity_router_prefix():
    r = Router(model_list=[])
    assert (
        r._is_auto_router_deployment(
            litellm_params=_params(model="auto_router/complexity_router")
        )
        is False
    )


def test_auto_router_check_still_matches_plain_auto_router_prefix():
    r = Router(model_list=[])
    assert (
        r._is_auto_router_deployment(
            litellm_params=_params(model="auto_router/my-semantic-router")
        )
        is True
    )


def test_adaptive_router_check_recognizes_prefix():
    r = Router(model_list=[])
    assert (
        r._is_adaptive_router_deployment(
            litellm_params=_params(model="auto_router/adaptive_router")
        )
        is True
    )


def test_adaptive_router_check_rejects_other_prefixes():
    r = Router(model_list=[])
    assert (
        r._is_adaptive_router_deployment(litellm_params=_params(model="openai/gpt-4o"))
        is False
    )


# ---- Fix 3: cost field path --------------------------------------------


def test_init_adaptive_router_reads_cost_from_litellm_params():
    r = Router(
        model_list=[
            {
                "model_name": "smart-cheap-router",
                "litellm_params": {
                    "model": "auto_router/adaptive_router",
                    "adaptive_router_config": {
                        "available_models": ["fast", "smart"],
                    },
                },
            },
            {
                "model_name": "fast",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "input_cost_per_token": 0.00000015,
                },
                "model_info": {
                    "adaptive_router_preferences": {
                        "quality_tier": 2,
                        "strengths": [],
                    }
                },
            },
            {
                "model_name": "smart",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "input_cost_per_token": 0.0000050,
                },
                "model_info": {
                    "adaptive_router_preferences": {
                        "quality_tier": 3,
                        "strengths": ["code_generation"],
                    }
                },
            },
        ]
    )
    assert "smart-cheap-router" in r.adaptive_routers
    assert r.adaptive_routers["smart-cheap-router"].model_to_cost == {
        "fast": 0.00000015,
        "smart": 0.0000050,
    }


# ---- Fix 4: pre-routing dispatch ---------------------------------------


def _router_with_adaptive() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "smart-cheap-router",
                "litellm_params": {
                    "model": "auto_router/adaptive_router",
                    "adaptive_router_config": {
                        "available_models": ["fast", "smart"],
                    },
                },
            },
            {
                "model_name": "fast",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "input_cost_per_token": 0.00000015,
                },
                "model_info": {
                    "adaptive_router_preferences": {
                        "quality_tier": 2,
                        "strengths": [],
                    }
                },
            },
            {
                "model_name": "smart",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "input_cost_per_token": 0.0000050,
                },
                "model_info": {
                    "adaptive_router_preferences": {
                        "quality_tier": 3,
                        "strengths": ["code_generation"],
                    }
                },
            },
        ]
    )


@pytest.mark.asyncio
async def test_async_pre_routing_hook_dispatches_to_adaptive_router():
    r = _router_with_adaptive()
    ar = r.adaptive_routers["smart-cheap-router"]
    ar.pick_model = AsyncMock(return_value="smart")  # type: ignore[assignment]

    response = await r.async_pre_routing_hook(
        model="smart-cheap-router",
        request_kwargs={"metadata": {"litellm_session_id": "sess-A"}},
        messages=[{"role": "user", "content": "Write a Python function"}],
    )
    assert response is not None
    assert response.model == "smart"
    call = ar.pick_model.await_args  # type: ignore[union-attr]
    # Stateless routing: session_id is no longer passed to pick_model.
    assert "session_id" not in call.kwargs
    assert call.kwargs["request_type"] == RequestType.CODE_GENERATION


@pytest.mark.asyncio
async def test_async_pre_routing_hook_pick_model_not_passed_session_id():
    r = _router_with_adaptive()
    ar = r.adaptive_routers["smart-cheap-router"]
    ar.pick_model = AsyncMock(return_value="fast")  # type: ignore[assignment]

    response = await r.async_pre_routing_hook(
        model="smart-cheap-router",
        request_kwargs={},
        messages=[{"role": "user", "content": "hello"}],
    )
    assert response is not None
    assert response.model == "fast"
    assert "session_id" not in ar.pick_model.await_args.kwargs  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_async_pre_routing_hook_returns_none_for_unrelated_model():
    r = _router_with_adaptive()
    ar = r.adaptive_routers["smart-cheap-router"]
    ar.pick_model = AsyncMock()  # type: ignore[assignment]
    response = await r.async_pre_routing_hook(
        model="some-other-model",
        request_kwargs={},
        messages=[{"role": "user", "content": "x"}],
    )
    assert response is None
    ar.pick_model.assert_not_awaited()  # type: ignore[union-attr]


# ---- Response header surfacing -----------------------------------------


@pytest.mark.asyncio
async def test_async_pre_routing_hook_stashes_chosen_model_in_metadata():
    """
    The adaptive-router branch must record the chosen logical model on
    `request_kwargs["metadata"]` so `_acompletion` can surface it as the
    `x-litellm-adaptive-router-model` response header.
    """
    r = _router_with_adaptive()
    r.adaptive_routers["smart-cheap-router"].pick_model = AsyncMock(  # type: ignore[assignment]
        return_value="smart"
    )

    request_kwargs: dict = {"metadata": {"litellm_session_id": "sess-A"}}
    await r.async_pre_routing_hook(
        model="smart-cheap-router",
        request_kwargs=request_kwargs,
        messages=[{"role": "user", "content": "Write a Python function"}],
    )
    assert request_kwargs["metadata"]["adaptive_router_chosen_model"] == "smart"


@pytest.mark.asyncio
async def test_async_pre_routing_hook_creates_metadata_when_missing():
    """If no metadata was passed in, the hook should create one to stash the chosen model."""
    r = _router_with_adaptive()
    r.adaptive_routers["smart-cheap-router"].pick_model = AsyncMock(  # type: ignore[assignment]
        return_value="fast"
    )

    request_kwargs: dict = {}
    await r.async_pre_routing_hook(
        model="smart-cheap-router",
        request_kwargs=request_kwargs,
        messages=[{"role": "user", "content": "hello"}],
    )
    assert request_kwargs["metadata"]["adaptive_router_chosen_model"] == "fast"


# ---- Multi-router support ----------------------------------------------


def test_two_adaptive_routers_can_coexist_on_one_router():
    r = Router(
        model_list=[
            {
                "model_name": "cheap-router",
                "litellm_params": {
                    "model": "auto_router/adaptive_router",
                    "adaptive_router_config": {"available_models": ["fast"]},
                },
            },
            {
                "model_name": "premium-router",
                "litellm_params": {
                    "model": "auto_router/adaptive_router",
                    "adaptive_router_config": {"available_models": ["smart"]},
                },
            },
            {
                "model_name": "fast",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "input_cost_per_token": 0.00000015,
                },
            },
            {
                "model_name": "smart",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "input_cost_per_token": 0.0000050,
                },
            },
        ]
    )
    assert set(r.adaptive_routers.keys()) == {"cheap-router", "premium-router"}
    assert r.adaptive_routers["cheap-router"].config.available_models == ["fast"]
    assert r.adaptive_routers["premium-router"].config.available_models == ["smart"]


@pytest.mark.asyncio
async def test_async_pre_routing_hook_dispatches_to_correct_router_when_multiple():
    """Each adaptive router only handles its own router_name."""
    r = Router(
        model_list=[
            {
                "model_name": "cheap-router",
                "litellm_params": {
                    "model": "auto_router/adaptive_router",
                    "adaptive_router_config": {"available_models": ["fast"]},
                },
            },
            {
                "model_name": "premium-router",
                "litellm_params": {
                    "model": "auto_router/adaptive_router",
                    "adaptive_router_config": {"available_models": ["smart"]},
                },
            },
            {
                "model_name": "fast",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "input_cost_per_token": 0.00000015,
                },
            },
            {
                "model_name": "smart",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "input_cost_per_token": 0.0000050,
                },
            },
        ]
    )
    cheap = r.adaptive_routers["cheap-router"]
    premium = r.adaptive_routers["premium-router"]
    cheap.pick_model = AsyncMock(return_value="fast")  # type: ignore[assignment]
    premium.pick_model = AsyncMock(return_value="smart")  # type: ignore[assignment]

    cheap_response = await r.async_pre_routing_hook(
        model="cheap-router",
        request_kwargs={},
        messages=[{"role": "user", "content": "hi"}],
    )
    premium_response = await r.async_pre_routing_hook(
        model="premium-router",
        request_kwargs={},
        messages=[{"role": "user", "content": "hi"}],
    )

    assert cheap_response is not None and cheap_response.model == "fast"
    assert premium_response is not None and premium_response.model == "smart"
    cheap.pick_model.assert_awaited_once()  # type: ignore[union-attr]
    premium.pick_model.assert_awaited_once()  # type: ignore[union-attr]


def test_init_adaptive_router_rejects_duplicate_model_name():
    """Two adaptive-router deployments with the same model_name must error."""
    from litellm.types.router import AdaptiveRouterConfig, Deployment

    r = Router(model_list=[])
    cfg = {"available_models": ["fast"]}
    deployment = Deployment(
        model_name="dup-router",
        litellm_params=LiteLLM_Params(
            model="auto_router/adaptive_router",
            adaptive_router_config=cfg,
        ),
        model_info={"id": "x"},
    )
    r.init_adaptive_router_deployment(deployment=deployment)
    with pytest.raises(ValueError, match="already exists"):
        r.init_adaptive_router_deployment(deployment=deployment)


def test_finalize_adaptive_router_if_configured_initializes_and_is_idempotent():
    """`_finalize_adaptive_router_if_configured` walks the model_list, builds an
    AdaptiveRouter for each adaptive deployment, and is a safe no-op on
    re-entry (models already in self.adaptive_routers are skipped)."""
    r = Router(
        model_list=[
            {
                "model_name": "fast",
                "litellm_params": {"model": "openai/gpt-4o-mini"},
                "model_info": {"input_cost_per_token": 0.00000015},
            },
            {
                "model_name": "smart",
                "litellm_params": {"model": "openai/gpt-4o"},
                "model_info": {"input_cost_per_token": 0.0000025},
            },
            {
                "model_name": "my-router",
                "litellm_params": {
                    "model": "auto_router/adaptive_router",
                    "adaptive_router_config": {
                        "available_models": ["fast", "smart"],
                    },
                },
            },
        ]
    )

    # Router __init__ already called _finalize_adaptive_router_if_configured.
    assert "my-router" in r.adaptive_routers
    original = r.adaptive_routers["my-router"]

    # Calling again must be idempotent: the existing AdaptiveRouter instance
    # is preserved, not rebuilt.
    r._finalize_adaptive_router_if_configured()
    assert r.adaptive_routers["my-router"] is original


def test_finalize_prunes_stale_adaptive_router_hooks_from_callbacks():
    """Replacing the Router (hot-reload path) must not leave stale
    AdaptiveRouterPostCallHook instances in `litellm.callbacks` — otherwise
    every request double-fires signal recording."""
    import litellm
    from litellm.router_strategy.adaptive_router.hooks import (
        AdaptiveRouterPostCallHook,
    )

    model_list = [
        {
            "model_name": "fast",
            "litellm_params": {"model": "openai/gpt-4o-mini"},
        },
        {
            "model_name": "my-router",
            "litellm_params": {
                "model": "auto_router/adaptive_router",
                "adaptive_router_config": {"available_models": ["fast"]},
            },
        },
    ]

    # Snapshot any pre-existing AdaptiveRouterPostCallHook entries so we can
    # restore them — other tests may have registered hooks we shouldn't drop.
    pre_hooks = [
        cb for cb in litellm.callbacks if isinstance(cb, AdaptiveRouterPostCallHook)
    ]
    for cb in pre_hooks:
        litellm.callbacks.remove(cb)

    try:
        Router(model_list=model_list)
        Router(model_list=model_list)  # simulate hot-reload

        adaptive_hooks = [
            cb
            for cb in litellm.callbacks
            if isinstance(cb, AdaptiveRouterPostCallHook)
        ]
        assert len(adaptive_hooks) == 1, (
            f"expected exactly one AdaptiveRouterPostCallHook after hot-reload, "
            f"got {len(adaptive_hooks)}"
        )
    finally:
        # Best-effort cleanup: remove whatever this test added, then restore.
        for cb in list(litellm.callbacks):
            if isinstance(cb, AdaptiveRouterPostCallHook):
                litellm.callbacks.remove(cb)
        for cb in pre_hooks:
            litellm.callbacks.append(cb)


def test_finalize_adaptive_router_if_configured_noop_when_none_configured():
    """With no adaptive deployments in model_list, the finalizer leaves
    `adaptive_routers` empty."""
    r = Router(
        model_list=[
            {
                "model_name": "fast",
                "litellm_params": {"model": "openai/gpt-4o-mini"},
            }
        ]
    )
    r._finalize_adaptive_router_if_configured()
    assert r.adaptive_routers == {}
