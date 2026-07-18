"""Proxy-level guard against URL-valued ``model`` / ``file_id`` request fields.

Some providers (HuggingFace, Oobabooga, Gemini files) accept a URL in the
identifier field and use it as the outbound destination. On the proxy that is
an SSRF primitive — guarded centrally in ``litellm_pre_call_utils`` so SDK
users keep working but proxy users default-deny.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Request

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import (
    _reject_url_valued_destinations,
    add_litellm_data_to_request,
)

sys.path.insert(0, os.path.abspath("../../.."))


class TestRejectUrlValuedDestinations:
    def test_plain_model_passes(self):
        _reject_url_valued_destinations({"model": "gpt-4"})

    def test_plain_file_id_passes(self):
        _reject_url_valued_destinations({"file_id": "files/abc123"})

    def test_no_destination_field_passes(self):
        _reject_url_valued_destinations({"messages": [{"role": "user"}]})

    def test_url_valued_model_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _reject_url_valued_destinations({"model": "https://attacker.example/v1"})
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["param"] == "model"

    def test_url_valued_file_id_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _reject_url_valued_destinations(
                {"file_id": "https://attacker.example/v1beta/files/abc"}
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["param"] == "file_id"

    def test_http_scheme_also_rejected(self):
        with pytest.raises(HTTPException):
            _reject_url_valued_destinations({"model": "http://10.0.0.1:8080/v1"})

    def test_non_string_value_ignored(self):
        # Defensive: malformed inputs (list, dict, None) shouldn't crash here;
        # downstream Pydantic validation handles the type error.
        _reject_url_valued_destinations({"model": None})
        _reject_url_valued_destinations({"model": 42})
        _reject_url_valued_destinations({"file_id": ["a", "b"]})

    def test_allowlisted_host_passes(self, monkeypatch):
        monkeypatch.setattr(
            litellm,
            "provider_url_destination_allowed_hosts",
            ["trusted.example"],
        )
        _reject_url_valued_destinations({"model": "https://trusted.example/v1"})

    def test_allowlisted_origin_rejects_mismatched_scheme(self, monkeypatch):
        monkeypatch.setattr(
            litellm,
            "provider_url_destination_allowed_hosts",
            ["https://trusted.example"],
        )
        _reject_url_valued_destinations({"model": "https://trusted.example/v1"})
        with pytest.raises(HTTPException):
            _reject_url_valued_destinations({"model": "http://trusted.example/v1"})

    def test_allowlisted_host_port_rejects_other_ports(self, monkeypatch):
        monkeypatch.setattr(
            litellm,
            "provider_url_destination_allowed_hosts",
            ["trusted.example:8443"],
        )
        _reject_url_valued_destinations({"model": "https://trusted.example:8443/v1"})
        with pytest.raises(HTTPException):
            _reject_url_valued_destinations({"model": "https://trusted.example/v1"})

    def test_userinfo_in_url_rejected_even_when_host_allowlisted(self, monkeypatch):
        # Embedded credentials in URL are an exfil channel — must never pass.
        monkeypatch.setattr(
            litellm,
            "provider_url_destination_allowed_hosts",
            ["trusted.example"],
        )
        with pytest.raises(HTTPException):
            _reject_url_valued_destinations(
                {"model": "https://user:pass@trusted.example/v1"}
            )


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


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_rejects_url_valued_model():
    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-key",
        metadata={},
        team_metadata={},
        spend=0.0,
        max_budget=100.0,
        model_max_budget={},
        team_spend=0.0,
        team_max_budget=200.0,
    )
    data = {"model": "https://attacker.example/v1", "messages": []}

    with pytest.raises(HTTPException) as exc_info:
        await add_litellm_data_to_request(
            data=data,
            request=_make_request_mock(),
            user_api_key_dict=user_api_key_dict,
            proxy_config=MagicMock(),
            general_settings={},
            version="test-version",
        )
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["param"] == "model"
