"""
Tests for weighted-routing failover (router_settings.enable_weighted_failover).

When enabled and the routing strategy is "simple-shuffle", a retryable failure
on one deployment causes the request to re-pick a different deployment in the
SAME model group (weighted across the remaining deployments) before any
cross-group fallback runs.
"""

from collections import Counter
from typing import Final, Optional

import pytest

import litellm
from litellm import Router
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo
from litellm.utils import get_excluded_filtered_deployments


# ---------------------------------------------------------------------------
# Unit tests for get_excluded_filtered_deployments
# ---------------------------------------------------------------------------


def _make_dep(dep_id: str, weight: Optional[int] = None) -> dict:
    params: dict = {"model": "gpt-4o", "api_key": "key"}
    if weight is not None:
        params["weight"] = weight
    return {
        "model_name": "test-model",
        "litellm_params": params,
        "model_info": {"id": dep_id},
    }


class TestGetExcludedFilteredDeployments:
    def test_no_excluded_returns_all(self):
        deps = [_make_dep("a"), _make_dep("b")]
        result = get_excluded_filtered_deployments(deps, excluded_deployment_ids=None)
        assert len(result) == 2

    def test_empty_excluded_returns_all(self):
        deps = [_make_dep("a"), _make_dep("b")]
        result = get_excluded_filtered_deployments(deps, excluded_deployment_ids=[])
        assert len(result) == 2

    def test_drops_excluded(self):
        deps = [_make_dep("a"), _make_dep("b"), _make_dep("c")]
        result = get_excluded_filtered_deployments(deps, excluded_deployment_ids=["b"])
        ids = sorted(d["model_info"]["id"] for d in result)
        assert ids == ["a", "c"]

    def test_all_excluded_returns_empty(self):
        # When every healthy deployment has been excluded, the helper must
        # return an empty list so the caller raises its usual no-deployments
        # error. Returning the original list here would re-include the
        # just-failed deployment and let weighted failover re-pick it.
        deps = [_make_dep("a"), _make_dep("b")]
        result = get_excluded_filtered_deployments(deps, excluded_deployment_ids=["a", "b"])
        assert result == []

    def test_excluded_set_with_unknown_ids(self):
        deps = [_make_dep("a"), _make_dep("b")]
        result = get_excluded_filtered_deployments(deps, excluded_deployment_ids=["zzz"])
        assert len(result) == 2

    def test_handles_missing_model_info(self):
        deps = [
            {"model_name": "x", "litellm_params": {"model": "gpt-4o"}},  # no model_info
            _make_dep("b"),
        ]
        result = get_excluded_filtered_deployments(deps, excluded_deployment_ids=["b"])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Integration tests for weighted-failover end-to-end via Router
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_failover_when_flag_off():
    """Default behavior: a failure on the picked deployment surfaces to caller."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("region-A failed"),
                    "weight": 1,
                },
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "ok from B",
                    "weight": 0,  # weight=0 so A is always picked
                },
                "model_info": {"id": "B"},
            },
        ],
        routing_strategy="simple-shuffle",
        num_retries=0,
        # enable_weighted_failover defaults to False
    )

    with pytest.raises(litellm.InternalServerError):
        await router.acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )


@pytest.mark.asyncio
async def test_failover_lands_on_other_deployment_when_flag_on():
    """Flag on: when A fails, request must succeed via B in the same call."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("region-A down"),
                    "weight": 1,  # always picked first (B has weight 0)
                },
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "ok from B",
                    "weight": 0,
                },
                "model_info": {"id": "B"},
            },
        ],
        routing_strategy="simple-shuffle",
        num_retries=0,
        enable_weighted_failover=True,
    )

    response = await router.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response._hidden_params["model_id"] == "B"


@pytest.mark.asyncio
async def test_failover_chain_three_deployments():
    """A and B fail, request succeeds on C."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("A down"),
                    "weight": 1_000_000,  # A always picked first
                },
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("B down"),
                    "weight": 1,  # picked when A is excluded
                },
                "model_info": {"id": "B"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "ok from C",
                    "weight": 0,
                },
                "model_info": {"id": "C"},
            },
        ],
        routing_strategy="simple-shuffle",
        num_retries=0,
        enable_weighted_failover=True,
    )

    response = await router.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response._hidden_params["model_id"] == "C"


@pytest.mark.asyncio
async def test_failover_exhausted_raises_original_error_class():
    """When ALL deployments fail, the request raises (does not hang)."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("A down"),
                    "weight": 1,
                },
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("B down"),
                    "weight": 1,
                },
                "model_info": {"id": "B"},
            },
        ],
        routing_strategy="simple-shuffle",
        num_retries=0,
        enable_weighted_failover=True,
    )

    with pytest.raises(litellm.InternalServerError):
        await router.acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )


@pytest.mark.asyncio
async def test_failover_falls_through_to_external_fallback():
    """When all deployments in the group fail, external fallback still runs."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("A down"),
                    "weight": 1,
                },
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("B down"),
                    "weight": 1,
                },
                "model_info": {"id": "B"},
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "ok from fallback",
                },
                "model_info": {"id": "fallback"},
            },
        ],
        routing_strategy="simple-shuffle",
        num_retries=0,
        enable_weighted_failover=True,
        fallbacks=[{"test-model": ["fallback-model"]}],
    )

    response = await router.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response._hidden_params["model_id"] == "fallback"


@pytest.mark.asyncio
async def test_weights_respected_when_all_healthy():
    """With both regions healthy, the picker should still honor configured
    weights — failover must not change the steady-state load shape."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "from A",
                    "weight": 80,
                },
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "from B",
                    "weight": 20,
                },
                "model_info": {"id": "B"},
            },
        ],
        routing_strategy="simple-shuffle",
        num_retries=0,
        enable_weighted_failover=True,
    )

    counts: Counter = Counter()
    for _ in range(1000):
        resp = await router.acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )
        counts[resp._hidden_params["model_id"]] += 1

    # Expect ~80/20 split. Loose bounds to keep the test stable under CI load.
    assert counts["A"] > counts["B"] * 2  # A should heavily dominate
    assert counts["B"] > 50  # but B should still get a meaningful share


@pytest.mark.asyncio
async def test_failover_skipped_for_non_simple_shuffle():
    """Weighted failover is only wired up for `simple-shuffle`. With another
    strategy, a failure on the picked deployment must NOT silently retry the
    other deployment in the same group. Both deployments fail here to keep the
    test deterministic regardless of which one the strategy picks first.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("A down"),
                },
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("B down"),
                },
                "model_info": {"id": "B"},
            },
        ],
        routing_strategy="latency-based-routing",
        num_retries=0,
        enable_weighted_failover=True,
    )

    with pytest.raises(litellm.InternalServerError):
        await router.acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )


@pytest.mark.asyncio
async def test_failover_skipped_for_context_window_error():
    """ContextWindowExceededError must NOT trigger weighted failover —
    it has its own dedicated fallback path. Uses the router's built-in
    `mock_testing_context_fallbacks` to deterministically raise the right
    exception class.
    """
    import litellm

    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "ok from A",
                    "weight": 1,
                },
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "ok from B",
                    "weight": 1,
                },
                "model_info": {"id": "B"},
            },
        ],
        routing_strategy="simple-shuffle",
        num_retries=0,
        enable_weighted_failover=True,
    )

    with pytest.raises(litellm.ContextWindowExceededError):
        await router.acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            mock_testing_context_fallbacks=True,
        )


@pytest.mark.asyncio
async def test_user_config_two_region_failover():
    """Mirrors the user's actual proxy_server_config.yaml shape: two Azure
    regions weighted 50/50, num_retries=0. With the flag on, a failure in
    one region is recovered by the other in the same request."""
    router = Router(
        model_list=[
            {
                "model_name": "gpt-5.4-mini",
                "litellm_params": {
                    "model": "azure/deployment-eastus2",
                    "api_key": "bad",
                    "api_base": "https://eastus2.example",
                    "mock_response": Exception("eastus2 5xx"),
                    "weight": 50,
                },
                "model_info": {"id": "eastus2"},
            },
            {
                "model_name": "gpt-5.4-mini",
                "litellm_params": {
                    "model": "azure/deployment-northcentralus",
                    "api_key": "good",
                    "api_base": "https://northcentralus.example",
                    "mock_response": "ok from northcentralus",
                    "weight": 50,
                },
                "model_info": {"id": "northcentralus"},
            },
        ],
        routing_strategy="simple-shuffle",
        cooldown_time=120,
        num_retries=0,
        enable_pre_call_checks=True,
        disable_cooldowns=False,
        allowed_fails=5,
        enable_weighted_failover=True,
    )

    # Force eastus2 to be picked first by leaving its weight intact and
    # asserting we always end up on northcentralus when eastus2 errors.
    # Run several requests and ensure we never see an unhandled failure.
    successes = Counter()
    for _ in range(20):
        resp = await router.acompletion(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": "hi"}],
        )
        successes[resp._hidden_params["model_id"]] += 1

    # With one region permanently failing, every request must land on the
    # other region (either directly because it was picked first, or via
    # failover because eastus2 was picked first).
    assert successes["northcentralus"] == 20
    assert successes["eastus2"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pre_call_check", "metadata"),
    [
        ("session_affinity", {"session_id": "sess-1"}),
        ("deployment_affinity", {"user_api_key_hash": "hashed-key-1"}),
    ],
)
async def test_failover_reaches_healthy_sibling_when_affinity_pins_failed_deployment(
    pre_call_check: str, metadata: dict[str, str]
):
    router: Final = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("A down"),
                    "weight": 1,
                },
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "ok from B",
                    "weight": 0,
                },
                "model_info": {"id": "B"},
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "ok from fallback",
                },
                "model_info": {"id": "fallback"},
            },
        ],
        routing_strategy="simple-shuffle",
        num_retries=0,
        disable_cooldowns=True,
        enable_weighted_failover=True,
        optional_pre_call_checks=[pre_call_check],
        fallbacks=[{"test-model": ["fallback-model"]}],
    )

    while_a_is_down: Final = await router.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        metadata=dict(metadata),
    )
    router.upsert_deployment(
        Deployment(
            model_name="test-model",
            litellm_params=LiteLLM_Params(model="gpt-4o", api_key="bad", mock_response="ok from A", weight=1),
            model_info=ModelInfo(id="A"),
        )
    )
    after_a_recovers: Final = await router.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        metadata=dict(metadata),
    )

    assert while_a_is_down._hidden_params["model_id"] == "B"
    assert while_a_is_down._hidden_params["additional_headers"]["x-litellm-attempted-fallbacks"] == 1
    assert after_a_recovers._hidden_params["model_id"] == "A"


def _two_deployment_router(model: str) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {"model": model, "api_key": "bad", "weight": 1},
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {"model": model, "api_key": "good", "weight": 0},
                "model_info": {"id": "B"},
            },
        ],
        routing_strategy="simple-shuffle",
        num_retries=0,
        disable_cooldowns=True,
        enable_weighted_failover=True,
    )


_ENDPOINTS: Final = [
    ("aembedding", "openai/text-embedding-3-small", {"input": ["hi"]}),
    ("aimage_generation", "openai/dall-e-3", {"prompt": "a cat"}),
    ("atranscription", "openai/whisper-1", {"file": b"audio"}),
    ("aspeech", "openai/tts-1", {"input": "hi", "voice": "alloy"}),
    ("atext_completion", "openai/gpt-3.5-turbo-instruct", {"prompt": "hi"}),
    ("arerank", "cohere/rerank-english-v3.0", {"query": "q", "documents": ["d"]}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("endpoint", "model", "arguments"), _ENDPOINTS, ids=[e[0] for e in _ENDPOINTS])
async def test_weighted_failover_moves_every_endpoint_to_another_deployment(
    monkeypatch: pytest.MonkeyPatch, endpoint: str, model: str, arguments: dict[str, object]
) -> None:
    keys_called: list[object] = []
    answer: Final = object()

    async def provider(**kwargs: object) -> object:
        keys_called.append(kwargs.get("api_key"))
        if kwargs.get("api_key") == "bad":
            raise litellm.InternalServerError(message="A is down", llm_provider="openai", model=model)
        return answer

    monkeypatch.setattr(litellm, endpoint, provider)
    router: Final = _two_deployment_router(model)

    response: Final = await getattr(router, endpoint)(model="test-model", **arguments)

    assert response is answer
    assert keys_called == ["bad", "good"]


@pytest.mark.asyncio
async def test_weighted_failover_skips_the_failed_deployment_for_clientside_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_bases_called: list[object] = []
    answer: Final = object()

    async def provider(**kwargs: object) -> object:
        api_bases_called.append(kwargs.get("api_base"))
        if kwargs.get("api_base") == "http://a":
            raise litellm.InternalServerError(message="A is down", llm_provider="openai", model="m")
        return answer

    monkeypatch.setattr(litellm, "aembedding", provider)
    router: Final = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {"model": "openai/text-embedding-3-small", "api_base": "http://a", "weight": 1},
                "model_info": {"id": "A"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {"model": "openai/text-embedding-3-small", "api_base": "http://b", "weight": 0},
                "model_info": {"id": "B"},
            },
        ],
        routing_strategy="simple-shuffle",
        num_retries=0,
        disable_cooldowns=True,
        enable_weighted_failover=True,
    )

    response: Final = await router.aembedding(model="test-model", input=["hi"], api_key="tenant-key")

    assert response is answer
    assert api_bases_called == ["http://a", "http://b"]


@pytest.mark.parametrize(
    ("endpoint", "arguments"),
    [
        ("embedding", {"input": ["hi"]}),
        ("image_generation", {"prompt": "a cat"}),
    ],
)
def test_a_sync_attempt_stamps_the_failed_deployment(
    monkeypatch: pytest.MonkeyPatch, endpoint: str, arguments: dict[str, object]
) -> None:
    def provider(**kwargs: object) -> object:
        raise litellm.InternalServerError(message="down", llm_provider="openai", model="m")

    monkeypatch.setattr(litellm, endpoint, provider)
    router: Final = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {"model": "openai/m", "api_key": "k"},
                "model_info": {"id": "A"},
            }
        ],
    )

    with pytest.raises(litellm.InternalServerError) as raised:
        getattr(router, endpoint)(model="test-model", **arguments)

    assert getattr(raised.value, "failed_deployment_id", None) == "A"


@pytest.mark.asyncio
async def test_a_failed_pick_stamps_no_deployment() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {"model": "openai/text-embedding-3-small", "api_key": "k"},
                "model_info": {"id": "A"},
            }
        ],
    )

    with pytest.raises(litellm.BadRequestError, match="no healthy deployments") as raised:
        await router.aembedding(input=["hi"], model="no-such-model")

    assert getattr(raised.value, "failed_deployment_id", None) is None
