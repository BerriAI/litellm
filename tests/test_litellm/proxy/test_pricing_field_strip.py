"""Proxy strips client-supplied pricing parameters from request bodies.

`litellm.completion` accepts pricing fields (`input_cost_per_token`,
`output_cost_per_token`, the rest of `CustomPricingLiteLLMParams`,
`metadata.model_info`) as part of its kwarg surface. On direct SDK use that
is intentional. On the proxy, those same fields would let any caller rewrite
their own per-request cost and — via `litellm.register_model` — mutate
`litellm.model_cost` for every subsequent caller in the worker. The proxy
strips them at the boundary; an opt-in key/team flag preserves the override
for operators who actually want it.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest
from fastapi import Request

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import (
    _CLIENT_PRICING_CONTROL_FIELDS,
    _CLIENT_PRICING_METADATA_FIELDS,
    _strip_client_pricing_overrides,
    add_litellm_data_to_request,
)
from litellm.types.utils import CustomPricingLiteLLMParams

sys.path.insert(0, os.path.abspath("../../.."))


def _make_request_mock() -> Request:
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"
    return request_mock


def _user_api_key_auth(metadata=None, team_metadata=None) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="hashed-key",
        metadata=metadata or {},
        team_metadata=team_metadata or {},
        spend=0.0,
        max_budget=100.0,
        model_max_budget={},
        team_spend=0.0,
        team_max_budget=200.0,
    )


class TestStripClientPricingOverrides:
    def test_pricing_field_set_tracks_pydantic_model(self):
        # The strip set is built from the model so additions are picked up
        # automatically — this test guards against the model and the strip
        # set drifting apart if someone replaces the auto-derivation later.
        assert _CLIENT_PRICING_CONTROL_FIELDS == frozenset(
            CustomPricingLiteLLMParams.model_fields.keys()
        )
        # Sanity: the obvious top-level pricing fields are in the set.
        for field in (
            "input_cost_per_token",
            "output_cost_per_token",
            "input_cost_per_second",
            "cache_creation_input_token_cost",
        ):
            assert field in _CLIENT_PRICING_CONTROL_FIELDS

    def test_root_pricing_fields_dropped(self):
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "cache_creation_input_token_cost": 0.0,
        }
        _strip_client_pricing_overrides(data)
        assert data == {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
        }

    def test_metadata_model_info_dropped(self):
        data = {
            "model": "gpt-4",
            "metadata": {
                "user_session": "keep-me",
                "model_info": {"input_cost_per_token": 0.0},
            },
            "litellm_metadata": {
                "model_info": {"output_cost_per_token": 0.0},
            },
        }
        _strip_client_pricing_overrides(data)
        assert data["metadata"] == {"user_session": "keep-me"}
        assert data["litellm_metadata"] == {}

    def test_non_pricing_fields_untouched(self):
        data = {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 100,
            "tools": [{"type": "function"}],
            "metadata": {"trace_id": "abc"},
        }
        snapshot = {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 100,
            "tools": [{"type": "function"}],
            "metadata": {"trace_id": "abc"},
        }
        _strip_client_pricing_overrides(data)
        assert data == snapshot

    def test_metadata_strip_handles_non_dict_metadata(self):
        # Defensive — Pydantic validation would normally reject non-dict
        # metadata, but the strip mustn't crash if a malformed body sneaks in.
        _strip_client_pricing_overrides({"metadata": "not-a-dict"})
        _strip_client_pricing_overrides({"metadata": None})
        _strip_client_pricing_overrides({"litellm_metadata": ["a", "b"]})

    def test_metadata_field_set_contains_model_info(self):
        assert "model_info" in _CLIENT_PRICING_METADATA_FIELDS

    def test_strip_emits_debug_log_listing_dropped_fields(self, caplog):
        # Operators need a paper trail so they can diagnose why a previously
        # working override stopped applying after the strip landed.
        import logging

        from litellm._logging import verbose_proxy_logger

        verbose_proxy_logger.setLevel(logging.DEBUG)
        with caplog.at_level(logging.DEBUG, logger=verbose_proxy_logger.name):
            _strip_client_pricing_overrides(
                {
                    "model": "gpt-4",
                    "input_cost_per_token": 0.0,
                    "metadata": {"model_info": {"output_cost_per_token": 0.0}},
                }
            )
        log_text = " ".join(record.getMessage() for record in caplog.records)
        assert "input_cost_per_token" in log_text
        assert "metadata.model_info" in log_text
        assert "allow_client_pricing_override" in log_text

    def test_strip_does_not_log_when_no_fields_present(self, caplog):
        # No-op strips must stay silent so the log isn't filled with noise on
        # every legitimate request.
        import logging

        from litellm._logging import verbose_proxy_logger

        verbose_proxy_logger.setLevel(logging.DEBUG)
        with caplog.at_level(logging.DEBUG, logger=verbose_proxy_logger.name):
            _strip_client_pricing_overrides({"model": "gpt-4", "temperature": 0.7})
        assert not any(
            "pricing" in record.getMessage().lower() for record in caplog.records
        )


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_strips_root_pricing_fields():
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }

    updated = await add_litellm_data_to_request(
        data=data,
        request=_make_request_mock(),
        user_api_key_dict=_user_api_key_auth(),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    assert "input_cost_per_token" not in updated
    assert "output_cost_per_token" not in updated


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_strips_metadata_model_info():
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"model_info": {"input_cost_per_token": 0.0}},
    }

    updated = await add_litellm_data_to_request(
        data=data,
        request=_make_request_mock(),
        user_api_key_dict=_user_api_key_auth(),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    assert "model_info" not in updated.get("metadata", {})


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_skips_strip_with_key_opt_in():
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "input_cost_per_token": 0.0001,
        "metadata": {"model_info": {"output_cost_per_token": 0.0002}},
    }

    user_auth = _user_api_key_auth(metadata={"allow_client_pricing_override": True})
    updated = await add_litellm_data_to_request(
        data=data,
        request=_make_request_mock(),
        user_api_key_dict=user_auth,
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    assert updated["input_cost_per_token"] == 0.0001
    assert updated["metadata"]["model_info"] == {"output_cost_per_token": 0.0002}


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_strips_json_string_litellm_metadata():
    """``litellm_metadata`` may arrive as a JSON-encoded string (multipart/
    form-data or ``extra_body``). The strip has to run after the proxy parses
    it into a dict; otherwise the ``isinstance(dict)`` guard skips the field
    and ``model_info`` survives the strip via the string path.
    """
    import json

    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "litellm_metadata": json.dumps({"model_info": {"input_cost_per_token": 0.0}}),
    }

    updated = await add_litellm_data_to_request(
        data=data,
        request=_make_request_mock(),
        user_api_key_dict=_user_api_key_auth(),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    parsed_metadata = updated.get("litellm_metadata")
    assert isinstance(parsed_metadata, dict)
    assert "model_info" not in parsed_metadata


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_skips_strip_with_team_opt_in():
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "input_cost_per_token": 0.0001,
    }

    user_auth = _user_api_key_auth(
        team_metadata={"allow_client_pricing_override": True}
    )
    updated = await add_litellm_data_to_request(
        data=data,
        request=_make_request_mock(),
        user_api_key_dict=user_auth,
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    assert updated["input_cost_per_token"] == 0.0001


@pytest.mark.asyncio
async def test_global_model_cost_unmutated_after_stripped_request(monkeypatch):
    """After a stripped request, ``litellm.model_cost`` must not carry the
    caller's submitted pricing for the model. The mutation only happens when
    the pricing fields reach ``litellm.completion``; the strip prevents that."""
    snapshot = dict(litellm.model_cost)
    data = {
        "model": "test-pricing-canary-model",
        "messages": [{"role": "user", "content": "hi"}],
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }

    await add_litellm_data_to_request(
        data=data,
        request=_make_request_mock(),
        user_api_key_dict=_user_api_key_auth(),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    # The strip prevents the pricing fields from ever reaching the path that
    # would mutate the global model_cost map.
    assert "test-pricing-canary-model" not in litellm.model_cost
    # And no other entries were mutated as a side effect.
    assert litellm.model_cost == snapshot
