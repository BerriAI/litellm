"""Regression tests for deployment / per-request custom pricing robustness.

Covers the two intertwined billing bugs described in
https://github.com/BerriAI/litellm/issues/30081:

Bug 1 - custom pricing lost when ``litellm.model_cost`` is replaced
        (e.g. after a model-cost-map reload). The deployment's explicit
        ``input_cost_per_token`` / ``output_cost_per_token`` must still be
        honored, because they are threaded through the cost calculator as
        ``custom_cost_per_token`` rather than depending on the deployment-id
        entry surviving in the process-global ``litellm.model_cost``.

Bug 2 - per-request custom pricing must apply to that request only and must
        NOT overwrite (poison) the shared canonical ``model_cost`` entry used
        by all other traffic.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm import Usage
from litellm.litellm_core_utils.litellm_logging import (
    extract_custom_cost_per_second,
    extract_custom_cost_per_token,
)


def _make_model_response(model: str, prompt_tokens: int, completion_tokens: int):
    return litellm.ModelResponse(
        model=model,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


# ---------------------------------------------------------------------------
# Bug 1 - custom_cost_per_token threading
# ---------------------------------------------------------------------------
def test_response_cost_calculator_honors_zero_custom_cost_per_token():
    """A deployment priced at 0 stays at 0 even when its model_cost entry is gone.

    Simulates the post-reload state: ``router_model_id`` is not present in
    ``litellm.model_cost`` and ``custom_pricing`` is detected from the
    deployment params. With the explicit rates threaded through as
    ``custom_cost_per_token`` the cost is 0, instead of falling back to the
    model's public price.
    """
    response = _make_model_response(
        "gpt-4o", prompt_tokens=100_000, completion_tokens=10_000
    )

    cost = litellm.response_cost_calculator(
        response_object=response,
        model="gpt-4o",
        custom_llm_provider="openai",
        call_type="acompletion",
        optional_params={},
        custom_pricing=True,
        router_model_id="id-not-in-model-cost",  # wiped by a reload
        custom_cost_per_token={
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        },
    )

    assert cost == 0.0


def test_response_cost_calculator_falls_back_without_custom_cost_per_token():
    """Control: without the threaded rates the same request bills at public price.

    This is the buggy behaviour from the issue - it confirms the threading in
    :func:`test_response_cost_calculator_honors_zero_custom_cost_per_token` is
    what fixes the silent mis-billing.
    """
    response = _make_model_response(
        "gpt-4o", prompt_tokens=100_000, completion_tokens=10_000
    )

    cost = litellm.response_cost_calculator(
        response_object=response,
        model="gpt-4o",
        custom_llm_provider="openai",
        call_type="acompletion",
        optional_params={},
        custom_pricing=True,
        router_model_id="id-not-in-model-cost",
    )

    assert cost > 0.0


def test_response_cost_calculator_honors_nonzero_custom_cost_per_token():
    """Explicit non-zero rates are applied exactly, regardless of model_cost."""
    response = _make_model_response(
        "gpt-4o", prompt_tokens=1_000, completion_tokens=2_000
    )

    cost = litellm.response_cost_calculator(
        response_object=response,
        model="gpt-4o",
        custom_llm_provider="openai",
        call_type="acompletion",
        optional_params={},
        custom_pricing=True,
        router_model_id="id-not-in-model-cost",
        custom_cost_per_token={
            "input_cost_per_token": 1e-06,
            "output_cost_per_token": 2e-06,
        },
    )

    # 1000 * 1e-06 + 2000 * 2e-06 = 0.001 + 0.004
    assert cost == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# extract_custom_cost_per_token helper
# ---------------------------------------------------------------------------
def test_extract_from_top_level_litellm_params():
    result = extract_custom_cost_per_token(
        {"input_cost_per_token": 0, "output_cost_per_token": 0}
    )
    assert result == {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}


@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata", "model_info"])
def test_extract_ignores_caller_model_info(metadata_key):
    costs = {
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
        "input_cost_per_second": 0,
    }
    nested = costs if metadata_key == "model_info" else {"model_info": costs}
    assert extract_custom_cost_per_token({metadata_key: nested}) is None
    assert extract_custom_cost_per_second({metadata_key: nested}) is None


def test_extract_includes_cache_rates_when_present():
    result = extract_custom_cost_per_token(
        {
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
            "cache_read_input_token_cost": 0,
            "cache_creation_input_token_cost": 0,
        }
    )
    assert result == {
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "cache_read_input_token_cost": 0.0,
        "cache_creation_input_token_cost": 0.0,
    }


def test_extract_returns_none_for_partial_pricing():
    # Only input set -> stay on the standard pricing path (don't assume 0 output).
    assert extract_custom_cost_per_token({"input_cost_per_token": 0}) is None


def test_extract_returns_none_when_absent():
    assert extract_custom_cost_per_token(None) is None
    assert extract_custom_cost_per_token({}) is None
    assert extract_custom_cost_per_token({"model_info": {}}) is None


# ---------------------------------------------------------------------------
# extract_custom_cost_per_second
# ---------------------------------------------------------------------------
def test_extract_custom_cost_per_second_top_level():
    result = extract_custom_cost_per_second({"input_cost_per_second": 0.005})
    assert result == 0.005


def test_extract_custom_cost_per_second_zero():
    result = extract_custom_cost_per_second({"input_cost_per_second": 0})
    assert result == 0.0


def test_extract_custom_cost_per_second_ignores_untrusted_model_info():
    """Top-level model_info is caller-controlled and must not supply request prices.

    This prevents proxy clients from injecting ``model_info: {input_cost_per_second: 0}``
    to bypass spend tracking for per-second priced deployments."""
    result = extract_custom_cost_per_second(
        {"model_info": {"input_cost_per_second": 0.01}}
    )
    assert result is None


def test_extract_custom_cost_per_second_absent():
    assert extract_custom_cost_per_second(None) is None
    assert extract_custom_cost_per_second({}) is None


# ---------------------------------------------------------------------------
# extract_custom_cost_per_token — security gate
# ---------------------------------------------------------------------------
def test_extract_ignores_untrusted_model_info():
    """Top-level model_info is caller-controlled and must not supply request prices.

    This prevents proxy clients from injecting ``model_info: {input_cost_per_token: 0,
    output_cost_per_token: 0}`` to bypass spend tracking."""
    litellm_params = {
        "model_info": {
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        }
    }
    result = extract_custom_cost_per_token(litellm_params)
    assert result is None


# ---------------------------------------------------------------------------
# response_cost_calculator — custom_cost_per_second threading
# ---------------------------------------------------------------------------
def test_response_cost_calculator_honors_custom_cost_per_second():
    r"""Per-second custom pricing is threaded through the cost calculator.

    completion_cost reads timing from _response_ms on the
    response object, so timing must be set there for per-second billing.
    """
    response = _make_model_response("gpt-4o", prompt_tokens=100, completion_tokens=0)
    response._response_ms = 5000.0

    # 0.01 $/s * 5 s = 0.05
    cost = litellm.response_cost_calculator(
        response_object=response,
        model="gpt-4o",
        custom_llm_provider="openai",
        call_type="acompletion",
        optional_params={},
        custom_pricing=True,
        router_model_id="id-not-in-model-cost",
        custom_cost_per_second=0.01,
    )

    assert cost == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Bug 2 - per-request pricing must not poison the canonical model_cost entry
# ---------------------------------------------------------------------------
@pytest.fixture
def pricing_provider():
    class PricingHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps(
                {
                    "id": "chatcmpl-pricing",
                    "object": "chat.completion",
                    "created": 1,
                    "model": request["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), PricingHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_per_request_custom_pricing_does_not_poison_canonical_entry(pricing_provider):
    model = "gpt-4o"
    canonical_before = litellm.model_cost[model].copy()
    request = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "api_base": pricing_provider,
        "api_key": "fixture-key",
    }
    custom_response = litellm.completion(
        **request, input_cost_per_token=0, output_cost_per_token=0
    )
    assert custom_response.usage.total_tokens == 150
    assert custom_response._hidden_params["response_cost"] == 0.0
    assert litellm.model_cost[model] == canonical_before

    standard_response = litellm.completion(**request)
    expected = (
        100 * canonical_before["input_cost_per_token"]
        + 50 * canonical_before["output_cost_per_token"]
    )
    assert expected > 0
    assert standard_response._hidden_params["response_cost"] == pytest.approx(expected)


@pytest.mark.parametrize("pricing_source", ["litellm_params", "model_info"])
def test_router_custom_pricing_survives_cost_map_reload(
    pricing_provider, monkeypatch, pricing_source
):
    canonical_before = litellm.model_cost["gpt-4o"].copy()
    deployment = {
        "model_name": "byok",
        "litellm_params": {
            "model": "openai/gpt-4o",
            "api_base": pricing_provider,
            "api_key": "fixture-key",
        },
        "model_info": {"id": "byok-deployment"},
    }
    deployment[pricing_source].update(
        {"input_cost_per_token": 0, "output_cost_per_token": 0}
    )
    router = litellm.Router(model_list=[deployment])
    monkeypatch.setattr(litellm, "model_cost", {"gpt-4o": canonical_before})
    response = router.completion(
        model="byok", messages=[{"role": "user", "content": "hi"}]
    )
    assert response.usage.total_tokens == 150
    assert response._hidden_params["response_cost"] == 0.0
    assert litellm.model_cost["gpt-4o"] == canonical_before


@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
def test_request_metadata_cannot_override_pricing(pricing_provider, metadata_key):
    canonical = litellm.model_cost["gpt-4o"]
    response = litellm.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        api_base=pricing_provider,
        api_key="fixture-key",
        **{
            metadata_key: {
                "model_info": {
                    "id": "client-deployment",
                    "input_cost_per_token": 0,
                    "output_cost_per_token": 0,
                }
            }
        },
    )
    expected = (
        100 * canonical["input_cost_per_token"]
        + 50 * canonical["output_cost_per_token"]
    )
    assert response._hidden_params["response_cost"] == pytest.approx(expected)


def test_custom_pricing_warning_does_not_emit_model_line_breaks(
    pricing_provider, monkeypatch, caplog
):
    model = "custom-model\nFORGED_LOG_ENTRY"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {
            **litellm.model_cost["gpt-4o"],
            "litellm_provider": "openai",
            "mode": "chat",
        },
    )
    response = litellm.completion(
        model=f"openai/{model}",
        messages=[{"role": "user", "content": "hi"}],
        api_base=pricing_provider,
        api_key="fixture-key",
        input_cost_per_token=0,
        output_cost_per_token=0,
    )
    assert response._hidden_params["response_cost"] == 0
    pricing_warnings = [
        record.getMessage()
        for record in caplog.records
        if "canonical pricing" in record.getMessage()
    ]
    assert pricing_warnings
    assert all(
        "\n" not in message and "\r" not in message for message in pricing_warnings
    )
