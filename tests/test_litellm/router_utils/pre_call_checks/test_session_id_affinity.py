from unittest.mock import AsyncMock, MagicMock, patch

import pytest


import json

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.constants import SESSION_DEPLOYMENT_AFFINITY_TTL_METADATA_KEY
from litellm.router_utils.pre_call_checks.deployment_affinity_check import (
    DeploymentAffinityCheck,
)


class MockResponse:
    def __init__(self, json_data, status_code):
        self._json_data = json_data
        self.status_code = status_code
        self.text = json.dumps(json_data)
        self.headers = {}

    def json(self):
        return self._json_data


@pytest.mark.asyncio
async def test_async_session_id_affinity_routes_to_same_deployment():
    """
    When session_affinity is enabled, subsequent requests from the same session id
    should route to the same deployment.
    """
    mock_response_data = {
        "id": "resp_mock-resp-123",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "azure/computer-use-preview",
        "output": [
            {
                "type": "message",
                "id": "msg_123",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hello there!", "annotations": []}
                ],
            }
        ],
        "parallel_tool_calls": True,
        "usage": {
            "input_tokens": 5,
            "output_tokens": 10,
            "total_tokens": 15,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "text": {"format": {"type": "text"}},
        "error": None,
        "previous_response_id": None,
    }

    router = litellm.Router(
        model_list=[
            {
                "model_name": "azure-computer-use-preview",
                "litellm_params": {
                    "model": "azure/computer-use-preview-1",
                    "api_key": "mock-api-key-1",
                    "api_version": "mock-api-version",
                    "api_base": "https://mock-endpoint-1.openai.azure.com",
                },
                "model_info": {"base_model": "computer-use-preview"},
            },
            {
                "model_name": "azure-computer-use-preview",
                "litellm_params": {
                    "model": "azure/computer-use-preview-2",
                    "api_key": "mock-api-key-2",
                    "api_version": "mock-api-version-2",
                    "api_base": "https://mock-endpoint-2.openai.azure.com",
                },
                "model_info": {"base_model": "computer-use-preview"},
            },
        ],
        optional_pre_call_checks=["session_affinity"],
    )

    model_group = "azure-computer-use-preview"
    session_id = "test-session-id-1"

    choice_calls = {"count": 0}

    def deterministic_choice(seq):
        choice_calls["count"] += 1
        if choice_calls["count"] == 1:
            return seq[0]
        return seq[1] if len(seq) > 1 else seq[0]

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post,
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=deterministic_choice,
        ),
    ):
        mock_post.return_value = MockResponse(mock_response_data, 200)

        first_response = await router.aresponses(
            model=model_group,
            input="Hello, how are you?",
            truncation="auto",
            litellm_metadata={"session_id": session_id},
        )
        first_model_id = first_response._hidden_params["model_id"]

        second_response = await router.aresponses(
            model=model_group,
            input="Follow-up question",
            truncation="auto",
            litellm_metadata={"session_id": session_id},
        )
        assert second_response._hidden_params["model_id"] == first_model_id


@pytest.mark.asyncio
async def test_async_session_id_affinity_priority_over_user_key():
    """
    If both session_affinity and deployment_affinity are enabled,
    session_affinity should have priority. We test this by sending different
    session ids for the same user.
    """
    cache = DualCache()
    callback = DeploymentAffinityCheck(
        cache=cache,
        ttl_seconds=123,
        enable_user_key_affinity=True,
        enable_responses_api_affinity=False,
        enable_session_id_affinity=True,
    )

    healthy_deployments = [
        {
            "model_name": "model_group",
            "litellm_params": {"model": "model_1"},
            "model_info": {"id": "deployment-1"},
        },
        {
            "model_name": "model_group",
            "litellm_params": {"model": "model_2"},
            "model_info": {"id": "deployment-2"},
        },
    ]

    await callback.cache.async_set_cache(
        DeploymentAffinityCheck.get_affinity_cache_key("model_group", "user1"),
        {"model_id": "deployment-1"},
    )

    await callback.cache.async_set_cache(
        DeploymentAffinityCheck.get_session_affinity_cache_key(
            "model_group", "session1", user_key="user1"
        ),
        {"model_id": "deployment-2"},
    )

    # Should use session mapping
    filtered = await callback.async_filter_deployments(
        model="model_group",
        healthy_deployments=healthy_deployments,
        messages=[],
        request_kwargs={
            "metadata": {"user_api_key_hash": "user1", "session_id": "session1"}
        },
    )

    assert len(filtered) == 1
    assert filtered[0]["model_info"]["id"] == "deployment-2"


MOCK_RESPONSES_API_RESPONSE = {
    "id": "resp_mock-resp-456",
    "object": "response",
    "created_at": 1741476542,
    "status": "completed",
    "model": "azure/computer-use-preview",
    "output": [],
    "usage": {
        "input_tokens": 5,
        "output_tokens": 10,
        "total_tokens": 15,
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}


def _smart_router(session_affinity=True, ttl_seconds=777, deployment_affinity=True):
    return litellm.Router(
        model_list=[
            {
                "model_name": "smart-router",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_default_model": "target-group",
                    "complexity_router_config": {
                        "session_affinity": session_affinity,
                        "deployment_affinity": deployment_affinity,
                        "session_affinity_ttl_seconds": ttl_seconds,
                        "tiers": {
                            "SIMPLE": "target-group",
                            "MEDIUM": "target-group",
                            "COMPLEX": "target-group",
                            "REASONING": "target-group",
                        },
                    },
                },
            },
            {
                "model_name": "target-group",
                "litellm_params": {
                    "model": "azure/computer-use-preview-1",
                    "api_key": "mock-api-key-1",
                    "api_version": "mock-api-version",
                    "api_base": "https://mock-endpoint-1.openai.azure.com",
                },
                "model_info": {"id": "deployment-1", "base_model": "computer-use-preview"},
            },
            {
                "model_name": "target-group",
                "litellm_params": {
                    "model": "azure/computer-use-preview-2",
                    "api_key": "mock-api-key-2",
                    "api_version": "mock-api-version-2",
                    "api_base": "https://mock-endpoint-2.openai.azure.com",
                },
                "model_info": {"id": "deployment-2", "base_model": "computer-use-preview"},
            },
        ],
    )


def _session_pin_key(session_id, user_key):
    return DeploymentAffinityCheck.get_session_affinity_cache_key(
        model_group="target-group", session_id=session_id, user_key=user_key
    )


def _cleanup_router_callbacks(router):
    for callback in router.optional_callbacks or []:
        litellm.logging_callback_manager.remove_callback_from_all_lists(callback)


async def _one_turn(router, model, session_id, key_hash):
    """One request with the shuffle forced to deployment-1, so any other landing
    deployment can only come from a pin read."""
    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post,
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=lambda seq: seq[0],
        ),
    ):
        mock_post.return_value = MockResponse(MOCK_RESPONSES_API_RESPONSE, 200)
        response = await router.aresponses(
            model=model,
            input=f"turn for {session_id} {key_hash}",
            litellm_metadata={"session_id": session_id, "user_api_key_hash": key_hash},
        )
    return response._hidden_params["model_id"]


@pytest.mark.asyncio
async def test_auto_router_session_affinity_writes_scoped_pin_and_follows_it():
    """Turn 1 persists a key-scoped deployment pin; a pin seeded to the deployment
    the shuffle would never pick is then followed, proving the read path."""
    router = _smart_router()
    try:
        served = await _one_turn(router, "smart-router", "write-session", "key-1")
        assert await router.cache.async_get_cache(key=_session_pin_key("write-session", "key-1")) == {
            "model_id": served
        }
        assert await router.cache.async_get_cache(key=_session_pin_key("write-session", None)) is None

        await router.cache.async_set_cache(
            key=_session_pin_key("read-session", "key-1"), value={"model_id": "deployment-2"}
        )
        assert await _one_turn(router, "smart-router", "read-session", "key-1") == "deployment-2"
    finally:
        _cleanup_router_callbacks(router)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,key_hash",
    [
        ("target-group", "key-1"),
        ("smart-router", "key-2"),
    ],
    ids=["direct-group-call", "different-api-key"],
)
async def test_seeded_session_pin_is_invisible_outside_its_scope(model, key_hash):
    """The pin binds (auto-routed request, api key, session): a direct call to the
    group and a different key reusing the session id must both ignore it."""
    router = _smart_router()
    try:
        await router.cache.async_set_cache(
            key=_session_pin_key("scoped-session", "key-1"), value={"model_id": "deployment-2"}
        )
        assert await _one_turn(router, model, "scoped-session", key_hash) == "deployment-1"
    finally:
        _cleanup_router_callbacks(router)


@pytest.mark.asyncio
async def test_marker_write_uses_marker_ttl_and_writes_only_the_session_pin():
    """The write hook honors the marker's TTL over the callback default and writes
    no user-key entry when only session affinity is active."""
    import time as time_module

    cache = DualCache()
    callback = DeploymentAffinityCheck(
        cache=cache,
        ttl_seconds=3600,
        enable_user_key_affinity=False,
        enable_responses_api_affinity=False,
    )

    await callback.async_pre_call_deployment_hook(
        kwargs={
            "model_info": {"id": "deployment-1"},
            "metadata": {
                "deployment_model_name": "target-group",
                "session_id": "ttl-session",
                "user_api_key_hash": "key-1",
                SESSION_DEPLOYMENT_AFFINITY_TTL_METADATA_KEY: 777,
            },
        },
        call_type=None,
    )

    session_key = _session_pin_key("ttl-session", "key-1")
    assert cache.in_memory_cache.cache_dict == {session_key: {"model_id": "deployment-1"}}
    assert cache.in_memory_cache.ttl_dict[session_key] == pytest.approx(time_module.time() + 777, abs=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_marker", ["777", True, -5, 0, None])
async def test_malformed_marker_values_do_not_enable_session_affinity(bad_marker):
    cache = DualCache()
    callback = DeploymentAffinityCheck(
        cache=cache,
        ttl_seconds=3600,
        enable_user_key_affinity=False,
        enable_responses_api_affinity=False,
    )

    await callback.async_pre_call_deployment_hook(
        kwargs={
            "model_info": {"id": "deployment-1"},
            "metadata": {
                "deployment_model_name": "target-group",
                "session_id": "bad-marker-session",
                "user_api_key_hash": "key-1",
                SESSION_DEPLOYMENT_AFFINITY_TTL_METADATA_KEY: bad_marker,
            },
        },
        call_type=None,
    )

    assert cache.in_memory_cache.cache_dict == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_user_key", [False, True], ids=["session-pin", "user-key-pin"])
async def test_concurrent_first_requests_never_flip_a_claimed_pin(enable_user_key):
    """Two overlapping first requests select different deployments before either
    write lands. Pins are first-writer-wins claims, so the second write must leave
    the stored pin unchanged instead of flipping it."""
    cache = DualCache()
    callback = DeploymentAffinityCheck(
        cache=cache,
        ttl_seconds=3600,
        enable_user_key_affinity=enable_user_key,
        enable_responses_api_affinity=False,
    )

    def racing_kwargs(deployment_id):
        metadata = {"deployment_model_name": "target-group", "user_api_key_hash": "key-1"}
        if not enable_user_key:
            metadata["session_id"] = "racing-session"
            metadata[SESSION_DEPLOYMENT_AFFINITY_TTL_METADATA_KEY] = 777
        return {"model_info": {"id": deployment_id}, "metadata": metadata}

    await callback.async_pre_call_deployment_hook(kwargs=racing_kwargs("deployment-1"), call_type=None)
    await callback.async_pre_call_deployment_hook(kwargs=racing_kwargs("deployment-2"), call_type=None)

    pinned_key = (
        DeploymentAffinityCheck.get_affinity_cache_key(model_group="target-group", user_key="key-1")
        if enable_user_key
        else _session_pin_key("racing-session", "key-1")
    )
    assert await cache.async_get_cache(key=pinned_key) == {"model_id": "deployment-1"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored_pin",
    [{"model_id": "deployment-1"}, "deployment-1"],
    ids=["dict-pin", "legacy-string-pin"],
)
async def test_in_memory_reclaim_slides_idle_window_only_for_the_stored_deployment(stored_pin):
    """The pod-local claim mirrors the Lua keepalive: the winning deployment's
    re-claim extends the pin's expiry, a losing deployment's claim touches neither
    the value nor the expiry, so no-Redis setups keep stickiness across an active
    session and ttl bounds idle time there too. Sameness is judged on the pinned
    model id, so a legacy string pin written by the Redis branch slides the same."""
    import time as time_module

    cache = DualCache()
    callback = DeploymentAffinityCheck(
        cache=cache,
        ttl_seconds=3600,
        enable_user_key_affinity=False,
        enable_responses_api_affinity=False,
    )
    pin_key = _session_pin_key("slide-session", "key-1")
    cache.in_memory_cache.set_cache(pin_key, stored_pin, ttl=10)
    first_expiry = cache.in_memory_cache.ttl_dict[pin_key]

    reclaimed = await callback._claim_pin(cache_key=pin_key, pin_value={"model_id": "deployment-1"}, ttl_seconds=777)
    assert reclaimed == "deployment-1"
    assert cache.in_memory_cache.ttl_dict[pin_key] == pytest.approx(time_module.time() + 777, abs=5)
    assert cache.in_memory_cache.ttl_dict[pin_key] > first_expiry

    lost = await callback._claim_pin(cache_key=pin_key, pin_value={"model_id": "deployment-2"}, ttl_seconds=10)
    assert lost == "deployment-1"
    assert cache.in_memory_cache.ttl_dict[pin_key] == pytest.approx(time_module.time() + 777, abs=5)


@pytest.mark.asyncio
async def test_claim_pin_uses_redis_attached_after_construction():
    """The proxy attaches Redis via Router._update_redis_cache after the Router (and
    this callback) are built. The claim must resolve the redis tier per call, or pins
    silently stay pod-local and cross-pod first-writer-wins is lost."""
    cache = DualCache()
    callback = DeploymentAffinityCheck(
        cache=cache,
        ttl_seconds=3600,
        enable_user_key_affinity=False,
        enable_responses_api_affinity=False,
    )

    captured = {}

    async def fake_runner(keys, args, client=None):
        captured["keys"] = keys
        captured["args"] = args
        return b'{"model_id": "other-pod-winner"}'

    late_redis = MagicMock()
    late_redis.async_register_script = MagicMock(return_value=fake_runner)
    cache.redis_cache = late_redis

    import time as time_module

    pin_key = _session_pin_key("late-redis-session", "key-1")
    cache.in_memory_cache.set_cache(pin_key, {"model_id": "other-pod-winner"}, ttl=10)

    claimed = await callback._claim_pin(
        cache_key=pin_key,
        pin_value={"model_id": "our-deployment"},
        ttl_seconds=777,
    )

    assert claimed == "other-pod-winner"
    assert cache.in_memory_cache.ttl_dict[pin_key] == pytest.approx(time_module.time() + 777, abs=5)
    assert captured["keys"] == (pin_key,)
    assert captured["args"] == ('{"model_id": "our-deployment"}', 777)
    assert cache.in_memory_cache.get_cache(_session_pin_key("late-redis-session", "key-1")) == {
        "model_id": "other-pod-winner"
    }


@pytest.mark.asyncio
async def test_claim_pin_falls_back_to_pod_local_when_redis_is_down():
    """A Redis outage must cost cross-pod agreement, never same-pod stickiness. The write
    hook only logs this result, so an escaping error would leave the session unpinned and
    reshuffle every turn for the whole outage. DualCache's write path, which this claim
    replaced, wrote the in-memory tier before ever touching Redis."""
    cache = DualCache()
    callback = DeploymentAffinityCheck(
        cache=cache,
        ttl_seconds=3600,
        enable_user_key_affinity=False,
        enable_responses_api_affinity=False,
    )

    async def exploding_runner(keys, args, client=None):
        raise ConnectionError("redis is down")

    down_redis = MagicMock()
    down_redis.async_register_script = MagicMock(return_value=exploding_runner)
    cache.redis_cache = down_redis

    key = _session_pin_key("outage-session", "key-1")
    claimed = await callback._claim_pin(cache_key=key, pin_value={"model_id": "our-deployment"}, ttl_seconds=777)

    assert claimed == "our-deployment"
    assert cache.in_memory_cache.get_cache(key) == {"model_id": "our-deployment"}

    second = await callback._claim_pin(cache_key=key, pin_value={"model_id": "another-deployment"}, ttl_seconds=777)
    assert second == "our-deployment"


@pytest.mark.asyncio
async def test_marker_session_affinity_read_and_write_agree_for_wildcard_groups():
    """Wildcard deployments keep the literal pattern as model_name on both the read
    path and the write path, so the marker-gated pin round-trips through one key."""
    callback = DeploymentAffinityCheck(
        cache=DualCache(),
        ttl_seconds=3600,
        enable_user_key_affinity=False,
        enable_responses_api_affinity=False,
    )
    request_kwargs = {
        "model_info": {"id": "wild-deployment-2"},
        "metadata": {
            "deployment_model_name": "openai/*",
            "session_id": "wild-session",
            "user_api_key_hash": "key-1",
            SESSION_DEPLOYMENT_AFFINITY_TTL_METADATA_KEY: 777,
        },
    }

    await callback.async_pre_call_deployment_hook(kwargs=request_kwargs, call_type=None)
    filtered = await callback.async_filter_deployments(
        model="openai/gpt-4o",
        healthy_deployments=[
            {
                "model_name": "openai/*",
                "litellm_params": {"model": "openai/gpt-4o"},
                "model_info": {"id": f"wild-deployment-{i}"},
            }
            for i in (1, 2)
        ],
        messages=[],
        request_kwargs=request_kwargs,
    )

    assert [d["model_info"]["id"] for d in filtered] == ["wild-deployment-2"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,session_affinity,deployment_affinity,expect_marker",
    [
        ("smart-router", False, True, True),
        ("smart-router", True, False, True),
        ("smart-router", False, False, False),
        ("target-group", False, True, False),
    ],
    ids=[
        "deployment-affinity-stamps",
        "session-affinity-implies-deployment-pin",
        "both-off-no-stamp",
        "non-auto-routed-clears",
    ],
)
async def test_pre_routing_hook_stamps_or_clears_the_marker_per_attempt(
    model, session_affinity, deployment_affinity, expect_marker
):
    """Every routing attempt writes or clears the marker, so a fallback from an
    auto-routed group to a plain group cannot carry a stale marker. session_affinity
    implies the deployment pin: a session frozen onto one group must not re-shuffle
    across that group's deployments."""
    router = _smart_router(session_affinity=session_affinity, deployment_affinity=deployment_affinity)
    try:
        request_kwargs = {
            "metadata": {SESSION_DEPLOYMENT_AFFINITY_TTL_METADATA_KEY: 111},
            "litellm_metadata": {SESSION_DEPLOYMENT_AFFINITY_TTL_METADATA_KEY: 111},
        }
        await router.async_pre_routing_hook(
            model=model,
            request_kwargs=request_kwargs,
            messages=[{"role": "user", "content": "Hello"}],
        )
        if expect_marker:
            assert request_kwargs["litellm_metadata"][SESSION_DEPLOYMENT_AFFINITY_TTL_METADATA_KEY] == 777
        else:
            assert SESSION_DEPLOYMENT_AFFINITY_TTL_METADATA_KEY not in request_kwargs["metadata"]
            assert SESSION_DEPLOYMENT_AFFINITY_TTL_METADATA_KEY not in request_kwargs["litellm_metadata"]
    finally:
        _cleanup_router_callbacks(router)


def test_complexity_router_with_deployment_affinity_registers_affinity_callback():
    enabled = _smart_router()
    session_only = _smart_router(session_affinity=True, deployment_affinity=False)
    disabled = _smart_router(session_affinity=False, deployment_affinity=False)
    try:
        assert [
            (cb.enable_user_key_affinity, cb.enable_responses_api_affinity, cb.enable_session_id_affinity)
            for cb in enabled.optional_callbacks or []
            if isinstance(cb, DeploymentAffinityCheck)
        ] == [(False, False, False)]
        assert any(isinstance(cb, DeploymentAffinityCheck) for cb in session_only.optional_callbacks or [])
        assert not any(isinstance(cb, DeploymentAffinityCheck) for cb in disabled.optional_callbacks or [])
    finally:
        _cleanup_router_callbacks(enabled)
        _cleanup_router_callbacks(session_only)
        _cleanup_router_callbacks(disabled)
