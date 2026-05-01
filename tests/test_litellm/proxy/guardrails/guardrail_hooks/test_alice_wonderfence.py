"""Tests for Alice WonderFence guardrail integration (V2 client + dynamic params)."""

import sys
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException


def _install_sdk_stub(monkeypatch, client_factory=None):
    """Install a stub `wonderfence_sdk` module so the guardrail can import it."""
    sdk = Mock()
    client_pkg = Mock()
    models_pkg = Mock()

    factory = client_factory or (lambda **kwargs: Mock(close=AsyncMock()))
    client_pkg.WonderFenceV2Client = Mock(side_effect=factory)
    sdk.client = client_pkg

    models_pkg.AnalysisContext = Mock(return_value=Mock())
    sdk.models = models_pkg

    monkeypatch.setitem(sys.modules, "wonderfence_sdk", sdk)
    monkeypatch.setitem(sys.modules, "wonderfence_sdk.client", client_pkg)
    monkeypatch.setitem(sys.modules, "wonderfence_sdk.models", models_pkg)
    return sdk


def _make_guardrail(monkeypatch, **overrides):
    """Build a WonderFenceGuardrail with stubbed SDK and a mock V2 client."""
    from litellm.types.guardrails import GuardrailEventHooks

    mock_client = Mock()
    mock_client.evaluate_prompt = AsyncMock()
    mock_client.evaluate_response = AsyncMock()
    mock_client.close = AsyncMock()

    _install_sdk_stub(monkeypatch, client_factory=lambda **kwargs: mock_client)

    from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.alice_wonderfence import (
        WonderFenceGuardrail,
    )

    kwargs = dict(
        guardrail_name="wonderfence-test",
        api_key="default-api-key",
        event_hook=[
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
        ],
        default_on=True,
    )
    kwargs.update(overrides)
    guardrail = WonderFenceGuardrail(**kwargs)
    return guardrail, mock_client


def _request_data(**overrides):
    metadata = overrides.pop("metadata", None)
    if metadata is None:
        metadata = {"alice_wonderfence_app_id": "test-app"}
    base = {"model": "gpt-4", "metadata": metadata}
    base.update(overrides)
    return base


# ----------------------------- resolver tests -----------------------------


def test_resolve_app_id_from_request_metadata(monkeypatch):
    guardrail, _ = _make_guardrail(monkeypatch)
    data = _request_data(metadata={"alice_wonderfence_app_id": "from-req"})
    assert guardrail._resolve_app_id(data) == "from-req"


def test_resolve_app_id_from_key_metadata(monkeypatch):
    guardrail, _ = _make_guardrail(monkeypatch)
    data = _request_data(
        metadata={
            "user_api_key_metadata": {"alice_wonderfence_app_id": "from-key"},
        }
    )
    assert guardrail._resolve_app_id(data) == "from-key"


def test_resolve_app_id_from_team_metadata(monkeypatch):
    guardrail, _ = _make_guardrail(monkeypatch)
    data = _request_data(
        metadata={
            "user_api_key_team_metadata": {"alice_wonderfence_app_id": "from-team"},
        }
    )
    assert guardrail._resolve_app_id(data) == "from-team"


def test_resolve_app_id_priority_request_over_key_over_team(monkeypatch):
    guardrail, _ = _make_guardrail(monkeypatch)
    data = _request_data(
        metadata={
            "alice_wonderfence_app_id": "from-req",
            "user_api_key_metadata": {"alice_wonderfence_app_id": "from-key"},
            "user_api_key_team_metadata": {"alice_wonderfence_app_id": "from-team"},
        }
    )
    assert guardrail._resolve_app_id(data) == "from-req"


def test_resolve_app_id_priority_key_over_team(monkeypatch):
    guardrail, _ = _make_guardrail(monkeypatch)
    data = _request_data(
        metadata={
            "user_api_key_metadata": {"alice_wonderfence_app_id": "from-key"},
            "user_api_key_team_metadata": {"alice_wonderfence_app_id": "from-team"},
        }
    )
    assert guardrail._resolve_app_id(data) == "from-key"


def test_resolve_app_id_missing_raises(monkeypatch):
    from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.alice_wonderfence import (
        WonderFenceMissingSecrets,
    )

    guardrail, _ = _make_guardrail(monkeypatch)
    data = _request_data(metadata={})
    with pytest.raises(WonderFenceMissingSecrets, match="alice_wonderfence_app_id"):
        guardrail._resolve_app_id(data)


def test_resolve_api_key_from_request_metadata(monkeypatch):
    guardrail, _ = _make_guardrail(monkeypatch, api_key="default")
    data = _request_data(metadata={"alice_wonderfence_api_key": "from-req"})
    assert guardrail._resolve_api_key(data) == "from-req"


def test_resolve_api_key_from_key_metadata(monkeypatch):
    guardrail, _ = _make_guardrail(monkeypatch, api_key="default")
    data = _request_data(
        metadata={
            "user_api_key_metadata": {"alice_wonderfence_api_key": "from-key"},
        }
    )
    assert guardrail._resolve_api_key(data) == "from-key"


def test_resolve_api_key_from_team_metadata(monkeypatch):
    guardrail, _ = _make_guardrail(monkeypatch, api_key="default")
    data = _request_data(
        metadata={
            "user_api_key_team_metadata": {"alice_wonderfence_api_key": "from-team"},
        }
    )
    assert guardrail._resolve_api_key(data) == "from-team"


def test_resolve_api_key_falls_back_to_default(monkeypatch):
    guardrail, _ = _make_guardrail(monkeypatch, api_key="default-key")
    data = _request_data(metadata={})
    assert guardrail._resolve_api_key(data) == "default-key"


def test_resolve_api_key_missing_everywhere_raises(monkeypatch):
    monkeypatch.delenv("ALICE_API_KEY", raising=False)
    guardrail, _ = _make_guardrail(monkeypatch, api_key=None)
    data = _request_data(metadata={})
    from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.alice_wonderfence import (
        WonderFenceMissingSecrets,
    )

    with pytest.raises(WonderFenceMissingSecrets):
        guardrail._resolve_api_key(data)


def test_resolve_reads_litellm_metadata_when_metadata_absent(monkeypatch):
    guardrail, _ = _make_guardrail(monkeypatch)
    data = {
        "model": "gpt-4",
        "litellm_metadata": {"alice_wonderfence_app_id": "from-litellm-md"},
    }
    assert guardrail._resolve_app_id(data) == "from-litellm-md"


# ----------------------------- LRU cache tests -----------------------------


@pytest.mark.asyncio
async def test_get_client_caches_per_api_key(monkeypatch):
    from litellm.types.guardrails import GuardrailEventHooks

    instances = []

    def factory(**kwargs):
        inst = Mock(close=AsyncMock())
        inst._kwargs = kwargs
        instances.append(inst)
        return inst

    _install_sdk_stub(monkeypatch, client_factory=factory)
    from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.alice_wonderfence import (
        WonderFenceGuardrail,
    )

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
async def test_get_client_lru_evicts_oldest(monkeypatch):
    from litellm.types.guardrails import GuardrailEventHooks

    def factory(**kwargs):
        return Mock(close=AsyncMock(), _api_key=kwargs["api_key"])

    _install_sdk_stub(monkeypatch, client_factory=factory)
    from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.alice_wonderfence import (
        WonderFenceGuardrail,
    )

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
async def test_get_client_forwards_config_to_v2_client(monkeypatch):
    from litellm.types.guardrails import GuardrailEventHooks

    captured = []

    def factory(**kwargs):
        captured.append(kwargs)
        return Mock(close=AsyncMock())

    _install_sdk_stub(monkeypatch, client_factory=factory)
    from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.alice_wonderfence import (
        WonderFenceGuardrail,
    )

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


# ----------------------------- apply_guardrail flow -----------------------------


@pytest.fixture
def guardrail_and_client(monkeypatch):
    g, c = _make_guardrail(monkeypatch)
    # Pre-seed cache so apply_guardrail uses our mock without rebuilding.
    g._client_cache["default-api-key"] = c
    return g, c


@pytest.mark.asyncio
async def test_apply_guardrail_block_action(guardrail_and_client):
    guardrail, client = guardrail_and_client
    result_obj = Mock()
    result_obj.action = "BLOCK"
    detection = Mock()
    detection.model_dump = Mock(return_value={"policy_name": "x", "confidence": 0.9})
    result_obj.detections = [detection]
    result_obj.correlation_id = "corr-1"
    client.evaluate_prompt.return_value = result_obj

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 400
    assert exc.value.detail["action"] == "BLOCK"
    assert exc.value.detail["wonderfence_correlation_id"] == "corr-1"
    assert exc.value.detail["error"] == (
        "Content violates our policies and has been blocked"
    )
    assert exc.value.detail["detections"][0]["policy_name"] == "x"


@pytest.mark.asyncio
async def test_apply_guardrail_block_uses_custom_block_message(monkeypatch):
    guardrail, client = _make_guardrail(
        monkeypatch, block_message="custom blocked text"
    )
    guardrail._client_cache["default-api-key"] = client
    result_obj = Mock()
    result_obj.action = "BLOCK"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_prompt.return_value = result_obj

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=_request_data(),
            input_type="request",
        )
    assert exc.value.detail["error"] == "custom blocked text"


@pytest.mark.asyncio
async def test_apply_guardrail_mask_replaces_last_text(guardrail_and_client):
    guardrail, client = guardrail_and_client
    result_obj = Mock()
    result_obj.action = "MASK"
    result_obj.action_text = "[REDACTED]"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_prompt.return_value = result_obj

    out = await guardrail.apply_guardrail(
        inputs={"texts": ["a", "b", "c"]},
        request_data=_request_data(),
        input_type="request",
    )
    assert out["texts"] == ["a", "b", "[REDACTED]"]


@pytest.mark.asyncio
async def test_apply_guardrail_no_action_passthrough(guardrail_and_client):
    guardrail, client = guardrail_and_client
    result_obj = Mock()
    result_obj.action = "NO_ACTION"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_prompt.return_value = result_obj

    out = await guardrail.apply_guardrail(
        inputs={"texts": ["safe"]},
        request_data=_request_data(),
        input_type="request",
    )
    assert out["texts"] == ["safe"]
    client.evaluate_prompt.assert_awaited_once()


@pytest.mark.asyncio
async def test_apply_guardrail_passes_app_id_per_call(guardrail_and_client):
    guardrail, client = guardrail_and_client
    result_obj = Mock()
    result_obj.action = "NO_ACTION"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_prompt.return_value = result_obj

    await guardrail.apply_guardrail(
        inputs={"texts": ["hi"]},
        request_data=_request_data(metadata={"alice_wonderfence_app_id": "tenant-A"}),
        input_type="request",
    )
    kwargs = client.evaluate_prompt.call_args.kwargs
    assert kwargs["app_id"] == "tenant-A"
    assert kwargs["prompt"] == "hi"
    assert kwargs["custom_fields"] is None


@pytest.mark.asyncio
async def test_apply_guardrail_response_path_passes_app_id(monkeypatch):
    guardrail, client = _make_guardrail(monkeypatch)
    guardrail._client_cache["default-api-key"] = client
    result_obj = Mock()
    result_obj.action = "NO_ACTION"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_response.return_value = result_obj

    await guardrail.apply_guardrail(
        inputs={"texts": ["resp"]},
        request_data=_request_data(metadata={"alice_wonderfence_app_id": "tenant-B"}),
        input_type="response",
    )
    kwargs = client.evaluate_response.call_args.kwargs
    assert kwargs["app_id"] == "tenant-B"
    assert kwargs["response"] == "resp"


@pytest.mark.asyncio
async def test_apply_guardrail_missing_app_id_fail_closed_returns_500(
    guardrail_and_client,
):
    """Missing app_id follows the fail_open pattern: fail_open=False → HTTP 500."""
    guardrail, _ = guardrail_and_client
    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=_request_data(metadata={}),
            input_type="request",
        )
    assert exc.value.status_code == 500
    assert "Error in Alice WonderFence Guardrail" in exc.value.detail["error"]
    assert "alice_wonderfence_app_id" in exc.value.detail["exception"]


@pytest.mark.asyncio
async def test_apply_guardrail_missing_api_key_fail_closed_returns_500(monkeypatch):
    """Missing api_key follows the fail_open pattern: fail_open=False → HTTP 500."""
    monkeypatch.delenv("ALICE_API_KEY", raising=False)
    guardrail, _ = _make_guardrail(monkeypatch, api_key=None)
    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 500
    assert "Error in Alice WonderFence Guardrail" in exc.value.detail["error"]
    assert "alice_wonderfence_api_key" in exc.value.detail["exception"]


@pytest.mark.asyncio
async def test_apply_guardrail_missing_app_id_fail_open_returns_500(monkeypatch):
    """Missing app_id is a config error: never fail-open, even with fail_open=True."""
    guardrail, _ = _make_guardrail(monkeypatch, fail_open=True)
    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=_request_data(metadata={}),
            input_type="request",
        )
    assert exc.value.status_code == 500
    assert "alice_wonderfence_app_id" in exc.value.detail["exception"]


@pytest.mark.asyncio
async def test_apply_guardrail_missing_api_key_fail_open_returns_500(monkeypatch):
    """Missing api_key is a config error: never fail-open, even with fail_open=True."""
    monkeypatch.delenv("ALICE_API_KEY", raising=False)
    guardrail, _ = _make_guardrail(monkeypatch, api_key=None, fail_open=True)
    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 500
    assert "alice_wonderfence_api_key" in exc.value.detail["exception"]


@pytest.mark.asyncio
async def test_apply_guardrail_fail_open_swallows_transport_error(monkeypatch):
    guardrail, client = _make_guardrail(monkeypatch, fail_open=True)
    guardrail._client_cache["default-api-key"] = client
    client.evaluate_prompt.side_effect = RuntimeError("network down")

    inputs = {"texts": ["original"]}
    out = await guardrail.apply_guardrail(
        inputs=inputs,
        request_data=_request_data(),
        input_type="request",
    )
    assert out["texts"] == ["original"]


@pytest.mark.asyncio
async def test_apply_guardrail_fail_closed_returns_500(guardrail_and_client):
    guardrail, client = guardrail_and_client
    client.evaluate_prompt.side_effect = RuntimeError("network down")

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 500
    assert "Error in Alice WonderFence Guardrail" in exc.value.detail["error"]


@pytest.mark.asyncio
async def test_block_not_bypassed_by_fail_open(monkeypatch):
    guardrail, client = _make_guardrail(monkeypatch, fail_open=True)
    guardrail._client_cache["default-api-key"] = client
    result_obj = Mock()
    result_obj.action = "BLOCK"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_prompt.return_value = result_obj

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["bad"]},
            request_data=_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_apply_guardrail_evaluates_only_last_text(guardrail_and_client):
    guardrail, client = guardrail_and_client
    result_obj = Mock()
    result_obj.action = "NO_ACTION"
    result_obj.detections = []
    result_obj.correlation_id = None
    client.evaluate_prompt.return_value = result_obj

    await guardrail.apply_guardrail(
        inputs={"texts": ["t1", "t2", "t3"]},
        request_data=_request_data(),
        input_type="request",
    )
    assert client.evaluate_prompt.call_count == 1
    assert client.evaluate_prompt.call_args.kwargs["prompt"] == "t3"


# ----------------------------- post_call logging_obj bridge -----------------------------


def _make_logging_obj() -> Mock:
    """Mock the LiteLLMLoggingObj surface we use: only model_call_details."""
    obj = Mock()
    obj.model_call_details = {}
    return obj


@pytest.mark.asyncio
async def test_post_call_recovers_app_id_via_logging_obj_stash(monkeypatch):
    """Reproduces the framework gap: request body metadata is dropped before
    post_call. The logging_obj stash from the prior `input_type="request"`
    call must be used to resolve app_id."""
    guardrail, client = _make_guardrail(monkeypatch)
    guardrail._client_cache["default-api-key"] = client
    request_obj = Mock()
    request_obj.action = "NO_ACTION"
    request_obj.detections = []
    request_obj.correlation_id = None
    client.evaluate_prompt.return_value = request_obj
    response_obj = Mock()
    response_obj.action = "NO_ACTION"
    response_obj.detections = []
    response_obj.correlation_id = None
    client.evaluate_response.return_value = response_obj

    logging_obj = _make_logging_obj()

    # Step 1: simulate pre_call / during_call with full request body
    # metadata — this is where the stash happens.
    await guardrail.apply_guardrail(
        inputs={"texts": ["hello"]},
        request_data=_request_data(metadata={"alice_wonderfence_app_id": "tenant-X"}),
        input_type="request",
        logging_obj=logging_obj,
    )

    # Step 2: simulate post_call as the framework actually invokes it —
    # the request body's metadata is gone (only litellm_metadata.user_api_key_*
    # would normally be present, neither populated here). Without the
    # bridge this raises; with it, we recover from logging_obj.
    out = await guardrail.apply_guardrail(
        inputs={"texts": ["llm response"]},
        request_data={"model": "gpt-4", "metadata": {}},
        input_type="response",
        logging_obj=logging_obj,
    )
    assert out["texts"] == ["llm response"]
    assert client.evaluate_response.call_args.kwargs["app_id"] == "tenant-X"


@pytest.mark.asyncio
async def test_post_call_prefers_request_data_over_stash(monkeypatch):
    """If post_call's request_data still resolves (e.g. app_id from key/team
    metadata), use it — don't fall back to the stash."""
    guardrail, client = _make_guardrail(monkeypatch)
    guardrail._client_cache["default-api-key"] = client
    request_obj = Mock()
    request_obj.action = "NO_ACTION"
    request_obj.detections = []
    request_obj.correlation_id = None
    client.evaluate_prompt.return_value = request_obj
    response_obj = Mock()
    response_obj.action = "NO_ACTION"
    response_obj.detections = []
    response_obj.correlation_id = None
    client.evaluate_response.return_value = response_obj

    logging_obj = _make_logging_obj()

    # Stash a different app_id during the request phase.
    await guardrail.apply_guardrail(
        inputs={"texts": ["hi"]},
        request_data=_request_data(
            metadata={"alice_wonderfence_app_id": "stashed-app"}
        ),
        input_type="request",
        logging_obj=logging_obj,
    )

    # Post_call request_data resolves via key metadata to a DIFFERENT app_id.
    # The resolver path must win over the stash.
    await guardrail.apply_guardrail(
        inputs={"texts": ["resp"]},
        request_data={
            "model": "gpt-4",
            "metadata": {
                "user_api_key_metadata": {"alice_wonderfence_app_id": "key-app"}
            },
        },
        input_type="response",
        logging_obj=logging_obj,
    )
    assert client.evaluate_response.call_args.kwargs["app_id"] == "key-app"


@pytest.mark.asyncio
async def test_post_call_without_prior_stash_raises(monkeypatch):
    """If neither request_data nor logging_obj has the app_id (e.g. mode is
    post_call only and app_id was supplied only in the request body), the
    error path must still fire — not silently allow."""
    guardrail, client = _make_guardrail(monkeypatch)
    guardrail._client_cache["default-api-key"] = client

    logging_obj = _make_logging_obj()  # empty model_call_details

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["resp"]},
            request_data={"model": "gpt-4", "metadata": {}},
            input_type="response",
            logging_obj=logging_obj,
        )
    assert exc.value.status_code == 500
    assert "alice_wonderfence_app_id" in exc.value.detail["exception"]


@pytest.mark.asyncio
async def test_post_call_recovers_via_sibling_stash(monkeypatch):
    """When two alice_wonderfence instances are listed in one request's
    `guardrails` array, LiteLLM only invokes one's during_call — but every
    instance runs post_call. The instance whose during_call did NOT fire
    must recover the stash written by the sibling that did."""
    g_writer, c_writer = _make_guardrail(monkeypatch, guardrail_name="writer")
    g_writer._client_cache["default-api-key"] = c_writer
    g_reader, c_reader = _make_guardrail(monkeypatch, guardrail_name="reader")
    g_reader._client_cache["default-api-key"] = c_reader
    for c in (c_writer, c_reader):
        result = Mock()
        result.action = "NO_ACTION"
        result.detections = []
        result.correlation_id = None
        c.evaluate_prompt.return_value = result
        c.evaluate_response.return_value = result

    logging_obj = _make_logging_obj()

    # Only the writer's during_call fires (simulating LiteLLM's
    # data["guardrail_to_apply"] last-write-wins behavior).
    await g_writer.apply_guardrail(
        inputs={"texts": ["hi"]},
        request_data=_request_data(metadata={"alice_wonderfence_app_id": "shared-app"}),
        input_type="request",
        logging_obj=logging_obj,
    )

    # Reader's post_call: own name not in stash, must fall back to writer's.
    await g_reader.apply_guardrail(
        inputs={"texts": ["resp"]},
        request_data={"model": "gpt-4", "metadata": {}},
        input_type="response",
        logging_obj=logging_obj,
    )
    assert c_reader.evaluate_response.call_args.kwargs["app_id"] == "shared-app"


@pytest.mark.asyncio
async def test_stash_keyed_per_guardrail_name(monkeypatch):
    """Two alice_wonderfence instances on the same logging_obj must not
    overwrite each other's stash — they're keyed by guardrail_name."""
    g1, c1 = _make_guardrail(monkeypatch, guardrail_name="alice-a")
    g1._client_cache["default-api-key"] = c1
    g2, c2 = _make_guardrail(monkeypatch, guardrail_name="alice-b")
    g2._client_cache["default-api-key"] = c2
    for c in (c1, c2):
        result = Mock()
        result.action = "NO_ACTION"
        result.detections = []
        result.correlation_id = None
        c.evaluate_prompt.return_value = result
        c.evaluate_response.return_value = result

    logging_obj = _make_logging_obj()

    # Both instances stash under the SAME logging_obj using DIFFERENT
    # request app_ids.
    await g1.apply_guardrail(
        inputs={"texts": ["hi"]},
        request_data=_request_data(metadata={"alice_wonderfence_app_id": "app-a"}),
        input_type="request",
        logging_obj=logging_obj,
    )
    await g2.apply_guardrail(
        inputs={"texts": ["hi"]},
        request_data=_request_data(metadata={"alice_wonderfence_app_id": "app-b"}),
        input_type="request",
        logging_obj=logging_obj,
    )

    # Each must recover its own value on post_call.
    await g1.apply_guardrail(
        inputs={"texts": ["resp"]},
        request_data={"model": "gpt-4", "metadata": {}},
        input_type="response",
        logging_obj=logging_obj,
    )
    await g2.apply_guardrail(
        inputs={"texts": ["resp"]},
        request_data={"model": "gpt-4", "metadata": {}},
        input_type="response",
        logging_obj=logging_obj,
    )
    assert c1.evaluate_response.call_args.kwargs["app_id"] == "app-a"
    assert c2.evaluate_response.call_args.kwargs["app_id"] == "app-b"


# ----------------------------- misc -----------------------------


def test_get_config_model(monkeypatch):
    from litellm.types.proxy.guardrails.guardrail_hooks.alice_wonderfence import (
        WonderFenceGuardrailConfigModel,
    )

    guardrail, _ = _make_guardrail(monkeypatch)
    assert guardrail.get_config_model() is WonderFenceGuardrailConfigModel


def test_initialization_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("ALICE_API_KEY", "env-key")
    guardrail, _ = _make_guardrail(monkeypatch, api_key=None)
    assert guardrail.api_key == "env-key"


def test_initialization_no_default_api_key_does_not_raise(monkeypatch):
    """V2 model resolves api_key per-request — init must NOT require it."""
    monkeypatch.delenv("ALICE_API_KEY", raising=False)
    guardrail, _ = _make_guardrail(monkeypatch, api_key=None)
    assert guardrail.api_key is None


def test_initialize_guardrail_forwards_all_params(monkeypatch):
    """The package-level initializer must forward every typed config field."""
    _install_sdk_stub(monkeypatch)
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


def test_initialize_guardrail_missing_name_raises(monkeypatch):
    """Initializer rejects guardrails without a guardrail_name."""
    _install_sdk_stub(monkeypatch)
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


def test_build_analysis_context_falls_back_to_slash_split(monkeypatch):
    """When `litellm.get_llm_provider` raises, fall back to `provider/model` split."""
    import litellm

    guardrail, _ = _make_guardrail(monkeypatch)

    def boom(model):
        raise ValueError("unknown provider")

    monkeypatch.setattr(litellm, "get_llm_provider", boom)
    guardrail._build_analysis_context({"model": "myorg/custom-llm"})

    AnalysisContext = sys.modules["wonderfence_sdk.models"].AnalysisContext
    kwargs = AnalysisContext.call_args.kwargs
    assert kwargs["provider"] == "myorg"
    assert kwargs["model_name"] == "custom-llm"


def test_recover_resolved_with_no_logging_obj_returns_none(monkeypatch):
    """_recover_resolved must short-circuit on None logging_obj."""
    guardrail, _ = _make_guardrail(monkeypatch)
    assert guardrail._recover_resolved(None) is None


def test_extract_relevant_text_uses_structured_messages(monkeypatch):
    """Request path with structured_messages routes through get_last_user_message."""
    guardrail, _ = _make_guardrail(monkeypatch)
    inputs = {
        "structured_messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "latest user msg"},
        ],
        "texts": ["unused-fallback"],
    }
    text = guardrail._extract_relevant_text(inputs, input_type="request")  # type: ignore[arg-type]
    assert text == "latest user msg"


@pytest.mark.asyncio
async def test_apply_guardrail_no_text_short_circuits(guardrail_and_client):
    """Empty inputs must skip the SDK call and return inputs unchanged."""
    guardrail, client = guardrail_and_client
    out = await guardrail.apply_guardrail(
        inputs={"texts": []},
        request_data=_request_data(),
        input_type="request",
    )
    assert out == {"texts": []}
    client.evaluate_prompt.assert_not_awaited()
    client.evaluate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_guardrail_detect_action_passes_through(guardrail_and_client):
    """DETECT action logs a warning but does not block or mutate inputs."""
    guardrail, client = guardrail_and_client
    result_obj = Mock()
    result_obj.action = "DETECT"
    result_obj.detections = []
    result_obj.correlation_id = "corr-detect"
    client.evaluate_prompt.return_value = result_obj

    out = await guardrail.apply_guardrail(
        inputs={"texts": ["watch me"]},
        request_data=_request_data(),
        input_type="request",
    )
    assert out["texts"] == ["watch me"]
    client.evaluate_prompt.assert_awaited_once()
