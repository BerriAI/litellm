"""Tests for the LRU client cache + SDK loader + guardrail initializer."""

import sys
from unittest.mock import AsyncMock, Mock

import pytest


# ----------------------------- LRU cache -----------------------------


@pytest.mark.asyncio
async def test_get_client_caches_per_api_key(install_sdk_stub):
    from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.alice_wonderfence import (
        WonderFenceGuardrail,
    )
    from litellm.types.guardrails import GuardrailEventHooks

    instances = []

    def factory(**kwargs):
        inst = Mock(close=AsyncMock())
        inst._kwargs = kwargs
        instances.append(inst)
        return inst

    install_sdk_stub(client_factory=factory)

    g = WonderFenceGuardrail(
        guardrail_name="t",
        api_key="default",
        event_hook=[GuardrailEventHooks.pre_call],
    )
    c1 = await g._get_client("key-A")
    c1_again = await g._get_client("key-A")
    c2 = await g._get_client("key-B")
    assert c1 is c1_again
    assert c1 is not c2
    assert len(instances) == 2


@pytest.mark.asyncio
async def test_get_client_lru_evicts_oldest(install_sdk_stub):
    from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.alice_wonderfence import (
        WonderFenceGuardrail,
    )
    from litellm.types.guardrails import GuardrailEventHooks

    def factory(**kwargs):
        return Mock(close=AsyncMock(), _api_key=kwargs["api_key"])

    install_sdk_stub(client_factory=factory)

    g = WonderFenceGuardrail(
        guardrail_name="t",
        api_key="default",
        max_cached_clients=2,
        event_hook=[GuardrailEventHooks.pre_call],
    )
    a = await g._get_client("A")
    b = await g._get_client("B")
    # Touching A makes B the LRU candidate.
    await g._get_client("A")
    c = await g._get_client("C")  # should evict B

    assert "A" in g._client_cache
    assert "C" in g._client_cache
    assert "B" not in g._client_cache
    # Evicted client must NOT be closed — in-flight requests may still hold a
    # reference. GC handles cleanup.
    b.close.assert_not_awaited()
    assert a is g._client_cache["A"]
    assert c is g._client_cache["C"]


@pytest.mark.asyncio
async def test_get_client_forwards_config_to_v2_client(install_sdk_stub):
    from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.alice_wonderfence import (
        WonderFenceGuardrail,
    )
    from litellm.types.guardrails import GuardrailEventHooks

    captured = []

    def factory(**kwargs):
        captured.append(kwargs)
        return Mock(close=AsyncMock())

    install_sdk_stub(client_factory=factory)

    g = WonderFenceGuardrail(
        guardrail_name="t",
        api_key="default",
        api_base="https://wf.example.com",
        api_timeout=15.4,
        platform="aws",
        connection_pool_limit=42,
        event_hook=[GuardrailEventHooks.pre_call],
    )
    await g._get_client("resolved-key")

    assert captured[0]["api_key"] == "resolved-key"
    assert captured[0]["base_url"] == "https://wf.example.com"
    assert captured[0]["api_timeout"] == 15  # rounded to int
    assert captured[0]["platform"] == "aws"
    assert captured[0]["connection_pool_limit"] == 42


# ----------------------------- initialization -----------------------------


def test_initialization_falls_back_to_env(monkeypatch, make_guardrail):
    monkeypatch.setenv("ALICE_API_KEY", "env-key")
    guardrail, _ = make_guardrail(api_key=None)
    assert guardrail.api_key == "env-key"


def test_initialization_no_default_api_key_does_not_raise(monkeypatch, make_guardrail):
    """V2 model resolves api_key per-request — init must NOT require it."""
    monkeypatch.delenv("ALICE_API_KEY", raising=False)
    guardrail, _ = make_guardrail(api_key=None)
    assert guardrail.api_key is None


def test_allow_request_metadata_override_defaults_false(make_guardrail):
    """New flag must default to False so request-body metadata cannot
    bypass admin-pinned credentials out of the box."""
    guardrail, _ = make_guardrail()
    assert guardrail.allow_request_metadata_override is False


def test_initialize_guardrail_forwards_all_params(install_sdk_stub):
    """The package-level initializer must forward every typed config field."""
    install_sdk_stub()
    from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence import (
        initialize_guardrail,
    )
    from litellm.types.guardrails import LitellmParams

    params = LitellmParams(
        guardrail="alice_wonderfence",
        mode="pre_call",
        api_key="cfg-key",
        api_base="https://wf.example.com",
        api_timeout=12.0,
        platform="aws",
        fail_open=True,
        block_message="custom block",
        debug=True,
        max_cached_clients=5,
        connection_pool_limit=20,
        allow_request_metadata_override=True,
        default_on=True,
    )
    guardrail = {"guardrail_name": "wf-init-test"}

    g = initialize_guardrail(params, guardrail)  # type: ignore[arg-type]

    assert g.api_key == "cfg-key"
    assert g.api_base == "https://wf.example.com"
    assert g.api_timeout == 12.0
    assert g.platform == "aws"
    assert g.fail_open is True
    assert g.block_message == "custom block"
    assert g._client_cache_maxsize == 5
    assert g._connection_pool_limit == 20
    assert g.allow_request_metadata_override is True


def test_initialize_guardrail_missing_name_raises(install_sdk_stub):
    """Initializer rejects guardrails without a guardrail_name."""
    install_sdk_stub()
    from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence import (
        initialize_guardrail,
    )
    from litellm.types.guardrails import LitellmParams

    params = LitellmParams(guardrail="alice_wonderfence", mode="pre_call")
    with pytest.raises(ValueError, match="requires a guardrail_name"):
        initialize_guardrail(params, {})  # type: ignore[arg-type]


def test_init_raises_when_sdk_not_installed(monkeypatch):
    """Constructor surfaces a clean ImportError when wonderfence_sdk missing."""
    monkeypatch.setitem(sys.modules, "wonderfence_sdk", None)
    from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.alice_wonderfence import (
        WonderFenceGuardrail,
    )

    with pytest.raises(ImportError, match="wonderfence-sdk"):
        WonderFenceGuardrail(guardrail_name="t")
