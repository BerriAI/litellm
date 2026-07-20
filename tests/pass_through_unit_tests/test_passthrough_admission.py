"""Tests for pass-through admission control (cost-tracking enforcement).

The guard exists because pass-through forwards a client request upstream using
the proxy's own credentials. An unpriced route bills the upstream account and
records $0 against the caller's key, which both defeats budgets and corrupts
any reconciliation that splits a provider invoice by gateway-computed cost.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy.pass_through_endpoints.passthrough_admission import (  # noqa: E402
    PassthroughAdmissionError,
    _normalize_path,
    enforce_passthrough_admission,
    find_matching_capability,
)

ANTHROPIC_MESSAGES = {
    "provider": "anthropic",
    "methods": ["POST"],
    "path": "/v1/messages",
    "model_source": "body",
}
BEDROCK_CONVERSE = {
    "provider": "bedrock",
    "methods": ["POST"],
    "path": "/model/{model_id}/converse",
    "model_source": "path:model_id",
}
CAPABILITIES = [ANTHROPIC_MESSAGES, BEDROCK_CONVERSE]

ENABLED = {
    "passthrough_require_cost_tracking": True,
    "passthrough_capabilities": CAPABILITIES,
}


def _enforce(path, method="POST", provider="anthropic", body=None, settings=None):
    enforce_passthrough_admission(
        general_settings=settings if settings is not None else ENABLED,
        provider=provider,
        method=method,
        path=path,
        request_body=body if body is not None else {"model": "claude-sonnet-5"},
    )


# ---------------------------------------------------------------------------
# Disabled by default — existing deployments must not change behaviour.
# ---------------------------------------------------------------------------


def test_no_op_when_setting_absent():
    _enforce("/v1/anything/at/all", settings={})


def test_no_op_when_setting_false():
    _enforce(
        "/v1/anything",
        settings={"passthrough_require_cost_tracking": False, "passthrough_capabilities": []},
    )


# ---------------------------------------------------------------------------
# The holes a prefix-based load balancer structurally cannot close.
# Each of these was a real finding; they are the reason this guard exists.
# ---------------------------------------------------------------------------


def test_registered_capability_is_allowed():
    _enforce("/v1/messages")


def test_subtree_of_a_registered_path_is_denied():
    # /v1/messages/batches rides along on any prefix rule for /v1/messages.
    # Batch work is billed later with no job-to-key ledger, so it must not pass.
    with pytest.raises(PassthroughAdmissionError) as exc:
        _enforce("/v1/messages/batches")
    assert "not a registered capability" in str(exc.value.message)


def test_free_sibling_endpoint_is_denied():
    # count_tokens is free. It would emit a legitimate $0 row that is
    # indistinguishable from "billed but unpriced" without route classification.
    with pytest.raises(PassthroughAdmissionError):
        _enforce("/v1/messages/count_tokens")


def test_method_is_enforced():
    # GET on the same path is object management (e.g. listing stored
    # completions) — free, and a prefix rule cannot express the distinction.
    with pytest.raises(PassthroughAdmissionError):
        _enforce("/v1/messages", method="GET")


def test_path_placeholder_matches_exactly_one_segment():
    _enforce(
        "/model/anthropic.claude-sonnet-5/converse",
        provider="bedrock",
        body={},
    )
    with pytest.raises(PassthroughAdmissionError):
        _enforce("/model/a/b/converse", provider="bedrock", body={})


def test_sibling_operation_under_placeholder_is_denied():
    # invoke-with-bidirectional-stream is a separate inference operation
    # outside the verified costing surface.
    with pytest.raises(PassthroughAdmissionError):
        _enforce(
            "/model/anthropic.claude-sonnet-5/invoke-with-bidirectional-stream",
            provider="bedrock",
            body={},
        )


def test_provider_is_enforced():
    with pytest.raises(PassthroughAdmissionError):
        _enforce("/v1/messages", provider="openai")


# ---------------------------------------------------------------------------
# Pricing: a registered route still has to resolve to a real price.
# ---------------------------------------------------------------------------


def test_unpriced_model_is_denied():
    with pytest.raises(PassthroughAdmissionError) as exc:
        _enforce("/v1/messages", body={"model": "definitely-not-a-real-model-xyz"})
    assert "no explicit price entry" in str(exc.value.message)


def test_missing_model_is_denied():
    with pytest.raises(PassthroughAdmissionError):
        _enforce("/v1/messages", body={})


def test_require_priced_model_can_be_disabled_for_genuinely_free_routes():
    settings = {
        "passthrough_require_cost_tracking": True,
        "passthrough_capabilities": [{**ANTHROPIC_MESSAGES, "path": "/v1/free", "require_priced_model": False}],
    }
    _enforce("/v1/free", body={}, settings=settings)


def test_model_from_path_is_priced_check_too():
    # Bedrock reads the model from the URL, not the body. A raw ARN has no
    # price-map entry, so it must be refused rather than recorded at $0.
    with pytest.raises(PassthroughAdmissionError):
        _enforce(
            "/model/arn:aws:bedrock:us-east-1:1234:application-inference-profile%2Fabc/converse",
            provider="bedrock",
            body={},
        )


# ---------------------------------------------------------------------------
# Path handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [("//v1//messages", "/v1/messages"), ("/v1/messages/", "/v1/messages"), ("", "/"), ("/", "/")],
)
def test_normalize_path(raw, expected):
    assert _normalize_path(raw) == expected


def test_duplicate_slashes_do_not_bypass_matching():
    _enforce("//v1//messages")


def test_percent_encoded_separator_is_not_decoded():
    # Decoding %2F would let a caller synthesise a path matching a narrower
    # template than the one the upstream actually routes.
    with pytest.raises(PassthroughAdmissionError):
        _enforce("/v1/messages%2Fbatches")


def test_unknown_capability_shape_is_ignored_not_crashed():
    settings = {
        "passthrough_require_cost_tracking": True,
        "passthrough_capabilities": ["not-a-dict", ANTHROPIC_MESSAGES],
    }
    _enforce("/v1/messages", settings=settings)


def test_non_list_capabilities_is_a_config_error():
    with pytest.raises(PassthroughAdmissionError) as exc:
        _enforce(
            "/v1/messages",
            settings={
                "passthrough_require_cost_tracking": True,
                "passthrough_capabilities": {"provider": "anthropic"},
            },
        )
    assert exc.value.status_code == 500


def test_empty_capability_list_denies_everything():
    # Fail closed: enabling enforcement with nothing registered must not
    # silently allow all traffic.
    with pytest.raises(PassthroughAdmissionError):
        _enforce(
            "/v1/messages",
            settings={
                "passthrough_require_cost_tracking": True,
                "passthrough_capabilities": [],
            },
        )


def test_find_matching_capability_returns_the_match_object():
    capability, match = find_matching_capability(CAPABILITIES, "bedrock", "POST", "/model/my-model/converse")
    assert capability is BEDROCK_CONVERSE
    assert match is not None and match.group("model_id") == "my-model"


# ---------------------------------------------------------------------------
# Enforcement must be explicit, never inferred.
# ---------------------------------------------------------------------------


def test_mock_like_settings_do_not_enable_enforcement():
    """Regression: a Mock's .get() returns a truthy Mock.

    `general_settings` is not guaranteed to be a plain dict. Treating any
    truthy return as "enabled" switched admission control on by accident and
    rejected every pass-through request with a 500.
    """
    from unittest.mock import MagicMock

    _enforce("/anything/unregistered", settings=MagicMock())


def test_non_mapping_settings_are_ignored():
    # Called directly rather than through _enforce, whose settings=None
    # sentinel means "use the enabled config".
    for settings in (None, [], "true", 1, object()):
        enforce_passthrough_admission(
            general_settings=settings,
            provider="anthropic",
            method="POST",
            path="/anything/unregistered",
            request_body={},
        )


@pytest.mark.parametrize("value", [True, "true", "True", "yes", "on", "1", 1])
def test_explicit_truthy_values_enable_enforcement(value):
    with pytest.raises(PassthroughAdmissionError):
        _enforce(
            "/anything/unregistered",
            settings={"passthrough_require_cost_tracking": value, "passthrough_capabilities": []},
        )


@pytest.mark.parametrize("value", [False, "false", "no", 0, None, "", object()])
def test_non_explicit_values_leave_enforcement_off(value):
    _enforce(
        "/anything/unregistered",
        settings={"passthrough_require_cost_tracking": value, "passthrough_capabilities": []},
    )


# ---------------------------------------------------------------------------
# Provider-scoped pricing: a price existing *anywhere* is not enough.
# ---------------------------------------------------------------------------

AZURE_CHAT = {
    "provider": "azure",
    "methods": ["POST"],
    "path": "/v1/chat/completions",
    "model_source": "body",
}


@pytest.fixture()
def no_router(monkeypatch):
    """Neutralise the router-pricing path so price-map scoping is what decides."""
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "llm_router", None)


def _azure_settings():
    return {
        "passthrough_require_cost_tracking": True,
        "passthrough_capabilities": [AZURE_CHAT],
    }


def test_cross_provider_price_collision_is_denied(monkeypatch, no_router):
    # `gpt-4` is priced for openai, NOT azure. Admitting it onto an azure
    # route still records $0 (the success handler prices under the azure key).
    import litellm

    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            "gpt-4": {"litellm_provider": "openai", "input_cost_per_token": 1e-6},
            "openai/gpt-4": {"litellm_provider": "openai", "input_cost_per_token": 1e-6},
        },
    )
    with pytest.raises(PassthroughAdmissionError) as exc:
        _enforce(
            "/v1/chat/completions",
            provider="azure",
            body={"model": "gpt-4"},
            settings=_azure_settings(),
        )
    assert "no explicit price entry" in str(exc.value.message)


def test_provider_prefixed_price_key_is_authoritative(monkeypatch, no_router):
    import litellm

    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            "gpt-4": {"litellm_provider": "openai", "input_cost_per_token": 1e-6},
            "openai/gpt-4": {"litellm_provider": "openai", "input_cost_per_token": 1e-6},
            "azure/gpt-4": {"litellm_provider": "azure", "input_cost_per_token": 1e-6},
        },
    )
    _enforce(
        "/v1/chat/completions",
        provider="azure",
        body={"model": "gpt-4"},
        settings=_azure_settings(),
    )


def test_bedrock_bare_key_priced_under_bedrock_converse_family(monkeypatch, no_router):
    # bedrock and bedrock_converse are the same billing family: a bare price
    # key tagged `bedrock_converse` must admit a model on a `bedrock` capability.
    import litellm

    monkeypatch.setattr(
        litellm,
        "model_cost",
        {"us.test-model:0": {"litellm_provider": "bedrock_converse", "input_cost_per_token": 1e-6}},
    )
    _enforce("/model/us.test-model:0/converse", provider="bedrock", body={})


# ---------------------------------------------------------------------------
# Router-priced aliases: the router IS the costing path for Bedrock aliases.
# ---------------------------------------------------------------------------


class _StubRouter:
    def __init__(self, deployments):
        self._deployments = deployments

    def get_model_list(self, model_name=None):
        return self._deployments


def test_router_alias_with_explicit_deployment_cost_is_admitted(monkeypatch):
    import litellm
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(litellm, "model_cost", {})
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        _StubRouter(
            [
                {
                    "model_name": "my-alias",
                    "litellm_params": {"model": "bedrock/some-unpriced-id", "input_cost_per_token": 3e-6},
                    "model_info": {},
                }
            ]
        ),
    )
    _enforce("/model/my-alias/converse", provider="bedrock", body={})


def test_router_alias_with_unpriced_deployment_is_denied(monkeypatch):
    import litellm
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(litellm, "model_cost", {})
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        _StubRouter(
            [
                {
                    "model_name": "my-alias",
                    "litellm_params": {"model": "bedrock/some-unpriced-id"},
                    "model_info": {},
                }
            ]
        ),
    )
    with pytest.raises(PassthroughAdmissionError) as exc:
        _enforce("/model/my-alias/converse", provider="bedrock", body={})
    assert "no explicit price entry" in str(exc.value.message)


# ---------------------------------------------------------------------------
# Template matching hardening
# ---------------------------------------------------------------------------


def test_trailing_newline_path_does_not_match():
    # `$` also matches before a trailing newline, so a path ending in an
    # encoded %0A satisfied templates it should not; `\Z` does not.
    with pytest.raises(PassthroughAdmissionError):
        _enforce("/v1/messages\n")


def test_duplicate_placeholder_template_is_a_named_config_error():
    settings = {
        "passthrough_require_cost_tracking": True,
        "passthrough_capabilities": [{"methods": ["POST"], "path": "/x/{id}/y/{id}", "model_source": "path:id"}],
    }
    with pytest.raises(PassthroughAdmissionError) as exc:
        _enforce("/x/1/y/2", settings=settings)
    assert exc.value.status_code == 500
    assert "Invalid capability path template" in str(exc.value.message)


def test_provider_scoped_capability_does_not_match_unknown_provider():
    # An entry point that cannot name its provider must not satisfy a
    # provider-scoped capability — the constraint would silently not bind
    # exactly where identity is weakest.
    capability, match = find_matching_capability(CAPABILITIES, None, "POST", "/v1/messages")
    assert capability is None and match is None
    with pytest.raises(PassthroughAdmissionError) as exc:
        _enforce("/v1/messages", provider=None)
    assert "not a registered capability" in str(exc.value.message)


# ---------------------------------------------------------------------------
# Route wiring: bedrock_llm_proxy_route must enforce admission BEFORE any
# upstream/router dispatch.
# ---------------------------------------------------------------------------

BEDROCK_ROUTE_SETTINGS_ON = {
    "passthrough_require_cost_tracking": True,
    "passthrough_capabilities": [],
}


class _UpstreamReached(Exception):
    """Sentinel: control flow got past admission into upstream dispatch."""


def _wire_bedrock_route(monkeypatch, general_settings):
    from unittest.mock import MagicMock

    import litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints as llm_pt
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", general_settings)

    async def _fake_read_request_body(request=None):
        return {"model": "anthropic.claude-v9-unpriced"}

    monkeypatch.setattr(llm_pt, "_read_request_body", _fake_read_request_body)

    def _sentinel(*args, **kwargs):
        raise _UpstreamReached()

    # First thing the route does after admission + count_tokens check.
    monkeypatch.setattr(llm_pt, "is_passthrough_request_using_router_model", _sentinel)

    request = MagicMock()
    request.method = "POST"
    return llm_pt, request


@pytest.mark.asyncio
async def test_bedrock_route_enforces_admission_before_upstream(monkeypatch):
    from fastapi import HTTPException, Response

    from litellm.proxy._types import UserAPIKeyAuth

    llm_pt, request = _wire_bedrock_route(monkeypatch, BEDROCK_ROUTE_SETTINGS_ON)

    with pytest.raises(HTTPException) as exc:
        await llm_pt.bedrock_llm_proxy_route(
            endpoint="model/anthropic.claude-v9-unpriced/converse",
            request=request,
            fastapi_response=Response(),
            user_api_key_dict=UserAPIKeyAuth(),
        )
    assert exc.value.status_code == 403
    assert "not a registered capability" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_bedrock_route_flag_off_does_not_raise_admission_403(monkeypatch):
    from fastapi import Response

    from litellm.proxy._types import UserAPIKeyAuth

    llm_pt, request = _wire_bedrock_route(monkeypatch, {})

    # With enforcement off the request must sail PAST admission — it then hits
    # our upstream-dispatch sentinel, proving no admission 403 was raised.
    with pytest.raises(_UpstreamReached):
        await llm_pt.bedrock_llm_proxy_route(
            endpoint="model/anthropic.claude-v9-unpriced/converse",
            request=request,
            fastapi_response=Response(),
            user_api_key_dict=UserAPIKeyAuth(),
        )
