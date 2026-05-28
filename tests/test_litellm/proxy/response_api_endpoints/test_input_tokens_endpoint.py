"""Tests for POST /v1/responses/input_tokens (LIT-2828).

Verifies the new Responses-API input-token-counting endpoint:
- 200 on string input and list-of-input-items input.
- 200 with extra ``instructions`` rolled into the system message.
- 400 when ``model`` or ``input`` is missing.
- Pre-fix regression: the path used to be claimed by GET
  ``/v1/responses/{response_id}``, so POST returned 405.
"""

import sys
import types
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def app_with_router():
    # Patch the auth dependency BEFORE the router is imported so FastAPI
    # captures the no-op replacement at decorator time. The replacement
    # must be parameterless or FastAPI will try to bind query params to it.
    from litellm.proxy.auth import user_api_key_auth as auth_mod

    async def _fake_auth():
        return auth_mod.UserAPIKeyAuth(
            api_key="sk-test", token="x", user_id="test"
        )

    auth_mod.user_api_key_auth = _fake_auth

    # Reload the endpoints module so the new dependency is bound.
    sys.modules.pop("litellm.proxy.response_api_endpoints.endpoints", None)
    from litellm.proxy.response_api_endpoints import endpoints as ep_mod  # noqa: E402

    app = FastAPI()
    app.include_router(ep_mod.router)
    return app, ep_mod


@pytest.fixture()
def client(app_with_router):
    app, _ = app_with_router
    return TestClient(app)


@pytest.fixture()
def fake_token_counter():
    """Patch proxy_server.token_counter to return a deterministic count."""
    from litellm.types.utils import TokenCountResponse

    async def _counter(request, call_endpoint=False):
        msg_count = len(request.messages or [])
        return TokenCountResponse(
            total_tokens=11 * msg_count + 4,
            request_model=request.model,
            model_used=request.model,
            tokenizer_type="openai_api",
        )

    with patch("litellm.proxy.proxy_server.token_counter", new=_counter):
        yield


HEADERS = {"Authorization": "Bearer sk-test", "Content-Type": "application/json"}


class TestResponsesInputTokens:
    def test_string_input_returns_200_with_input_tokens(
        self, client, fake_token_counter
    ):
        r = client.post(
            "/v1/responses/input_tokens",
            headers=HEADERS,
            json={"model": "openai/gpt-4o", "input": "Hello, how are you?"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "input_tokens" in body
        # 1 user message * 11 + 4 = 15
        assert body["input_tokens"] == 15

    def test_list_of_items_input_returns_200(self, client, fake_token_counter):
        r = client.post(
            "/v1/responses/input_tokens",
            headers=HEADERS,
            json={
                "model": "openai/gpt-4o",
                "instructions": "You are a helpful assistant.",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Tell me about GPUs."}
                        ],
                    },
                    {"role": "assistant", "content": "Sure."},
                ],
            },
        )
        assert r.status_code == 200, r.text
        # instructions (system) + 2 messages = 3 * 11 + 4 = 37
        assert r.json() == {"input_tokens": 37}

    def test_alias_path_responses_input_tokens(self, client, fake_token_counter):
        r = client.post(
            "/responses/input_tokens",
            headers=HEADERS,
            json={"model": "openai/gpt-4o", "input": "hi"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["input_tokens"] == 15

    def test_openai_alias_path(self, client, fake_token_counter):
        r = client.post(
            "/openai/v1/responses/input_tokens",
            headers=HEADERS,
            json={"model": "openai/gpt-4o", "input": "hi"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["input_tokens"] == 15

    def test_missing_model_returns_400(self, client, fake_token_counter):
        r = client.post(
            "/v1/responses/input_tokens",
            headers=HEADERS,
            json={"input": "hi"},
        )
        assert r.status_code == 400
        assert "model" in r.text

    def test_missing_input_returns_400(self, client, fake_token_counter):
        r = client.post(
            "/v1/responses/input_tokens",
            headers=HEADERS,
            json={"model": "openai/gpt-4o"},
        )
        assert r.status_code == 400
        assert "input" in r.text

    def test_unknown_input_item_shape_is_skipped(
        self, client, fake_token_counter
    ):
        """Unknown item types should not crash the counter; they are skipped."""
        r = client.post(
            "/v1/responses/input_tokens",
            headers=HEADERS,
            json={
                "model": "openai/gpt-4o",
                "input": [
                    {"type": "image_url", "image_url": "ignored"},
                    {"role": "user", "content": "actual text"},
                ],
            },
        )
        assert r.status_code == 200, r.text
        # only the second item (1 message) counts.
        assert r.json() == {"input_tokens": 15}

    def test_tools_are_forwarded_to_token_counter(self, client):
        from litellm.types.utils import TokenCountResponse

        captured = {}

        async def _counter(request, call_endpoint=False):
            captured["tools"] = request.tools
            captured["messages"] = request.messages
            captured["call_endpoint"] = call_endpoint
            return TokenCountResponse(
                total_tokens=42,
                request_model=request.model,
                model_used=request.model,
                tokenizer_type="openai_api",
            )

        with patch("litellm.proxy.proxy_server.token_counter", new=_counter):
            tools = [{"type": "function", "function": {"name": "x"}}]
            r = client.post(
                "/v1/responses/input_tokens",
                headers=HEADERS,
                json={
                    "model": "openai/gpt-4o",
                    "input": "weather?",
                    "tools": tools,
                },
            )
            assert r.status_code == 200, r.text
            assert r.json() == {"input_tokens": 42}
            assert captured["tools"] == tools
            assert captured["messages"] == [
                {"role": "user", "content": "weather?"}
            ]
            # Provider-specific counting requested.
            assert captured["call_endpoint"] is True


    def test_instructions_without_input_returns_400(self, client, fake_token_counter):
        """Regression: a payload with only ``instructions`` (no ``input``) must 400.

        Pre-fix the validation looked at the post-normalisation messages list,
        so an instructions-only payload produced one message and silently
        bypassed the required-``input`` check. The validation is now done
        against the raw body so this returns 400 as expected.
        """
        r = client.post(
            "/v1/responses/input_tokens",
            headers=HEADERS,
            json={"model": "openai/gpt-4o", "instructions": "You are helpful."},
        )
        assert r.status_code == 400, r.text
        assert "input" in r.text

    def test_empty_input_string_returns_400(self, client, fake_token_counter):
        r = client.post(
            "/v1/responses/input_tokens",
            headers=HEADERS,
            json={"model": "openai/gpt-4o", "input": ""},
        )
        assert r.status_code == 400, r.text
        assert "input" in r.text

    def test_empty_input_list_returns_400(self, client, fake_token_counter):
        r = client.post(
            "/v1/responses/input_tokens",
            headers=HEADERS,
            json={"model": "openai/gpt-4o", "input": []},
        )
        assert r.status_code == 400, r.text
        assert "input" in r.text



    def test_proxy_exception_preserves_status_code(self, client, monkeypatch):
        """A ProxyException raised by token_counter must surface as its original
        HTTP status code, not be masked as a 500.

        Regression for the Greptile P2 in PR #29215: prior to the fix, the
        generic ``except Exception`` block caught ProxyException and rewrote
        it to 500, dropping the original 401/403/429 etc. that drives client
        retry / billing logic.
        """
        from litellm.proxy import proxy_server
        from litellm.proxy._types import ProxyException

        async def raise_proxy_429(*args, **kwargs):
            raise ProxyException(
                message="rate limit exceeded",
                type="rate_limit_error",
                param=None,
                code="429",
            )

        monkeypatch.setattr(proxy_server, "token_counter", raise_proxy_429)
        r = client.post(
            "/v1/responses/input_tokens",
            headers=HEADERS,
            json={"model": "openai/gpt-4o", "input": "hello"},
        )
        assert r.status_code == 429, r.text
        assert "rate limit" in r.text

    def test_proxy_exception_non_numeric_code_falls_back_to_500(
        self, client, monkeypatch
    ):
        """A ProxyException with a non-numeric ``code`` falls back to 500 so
        FastAPI doesn't crash when the upstream code can't be coerced to int."""
        from litellm.proxy import proxy_server
        from litellm.proxy._types import ProxyException

        async def raise_proxy_garbage(*args, **kwargs):
            raise ProxyException(
                message="boom",
                type="server_error",
                param=None,
                code="not-a-number",
            )

        monkeypatch.setattr(proxy_server, "token_counter", raise_proxy_garbage)
        r = client.post(
            "/v1/responses/input_tokens",
            headers=HEADERS,
            json={"model": "openai/gpt-4o", "input": "hi"},
        )
        assert r.status_code == 500, r.text

class TestNormalizeResponsesInputToMessages:
    """Direct unit tests for the input-shape normalizer helper."""

    def _norm(self, **kw):
        from litellm.proxy.response_api_endpoints.endpoints import (
            _normalize_responses_input_to_messages,
        )

        return _normalize_responses_input_to_messages(
            instructions=kw.get("instructions"),
            input_value=kw.get("input_value"),
        )

    def test_string_input(self):
        assert self._norm(input_value="hi") == [
            {"role": "user", "content": "hi"}
        ]

    def test_string_input_with_instructions(self):
        assert self._norm(instructions="sys", input_value="hi") == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]

    def test_empty_string_input_returns_empty(self):
        assert self._norm(input_value="") == []

    def test_list_with_str_content(self):
        out = self._norm(
            input_value=[{"role": "user", "content": "hi"}]
        )
        assert out == [{"role": "user", "content": "hi"}]

    def test_list_with_typed_content_parts(self):
        out = self._norm(
            input_value=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "hello "},
                        {"type": "input_text", "text": "world"},
                    ],
                }
            ]
        )
        assert out == [{"role": "user", "content": "hello world"}]

    def test_missing_role_defaults_to_user(self):
        out = self._norm(input_value=[{"content": "hi"}])
        assert out == [{"role": "user", "content": "hi"}]

    def test_unknown_part_types_are_skipped(self):
        out = self._norm(
            input_value=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url"},  # no "text"
                        {"type": "input_text", "text": "kept"},
                    ],
                }
            ]
        )
        assert out == [{"role": "user", "content": "kept"}]

    def test_non_dict_items_are_skipped(self):
        out = self._norm(input_value=["just a string", None, 42])
        assert out == []

    def test_none_input_returns_empty(self):
        assert self._norm(input_value=None) == []
