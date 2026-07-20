"""Live e2e: the gateway's authorization and error-shape contract.

A virtual key may only call models in its allow-list and route groups in its
allowed_routes; both denials are a 403 raised before any provider is touched. A
syntactically valid request naming a non-existent model is a 400 with a JSON body,
never forwarded and never a 5xx. Migrated from
litellm-regression-tests/tests/test_access_control.py: the source asserted 401 for
the disallowed-model case against an older proxy, but the current contract
(auth_checks.py) is a 403 key_model_access_denied, and the unknown-route check is
replaced by a stronger route-permission check (an llm-only key rejected from a
management route).
"""

from __future__ import annotations

import json

import pytest

from access_control_client import (
    AccessControlClient,
    MODEL_ACCESS_DENIED_MARKER,
    ROUTE_NOT_ALLOWED_MARKER,
)
from e2e_config import unique_marker
from e2e_http import Success, UnauthorizedError, UnknownApiError, unwrap
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, LiteLLMParamsBody
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e

ALLOWED_MODEL = "gemini-2.5-flash"
DISALLOWED_MODEL = "gpt-5.5"
VIRTUAL_KEY_BACKEND = "anthropic/claude-haiku-4-5-20251001"


def _is_json(body: str) -> bool:
    try:
        json.loads(body)
        return True
    except ValueError:
        return False



class TestAccessControl:
    def test_disallowed_model_is_denied_403(
        self, client: AccessControlClient, resources: ResourceManager
    ) -> None:
        key = resources.key(models=[ALLOWED_MODEL])
        result = client.chat_status(
            key, DISALLOWED_MODEL, f"capital of France? {unique_marker()}"
        )
        assert result.status_code == 403, (
            f"key limited to {ALLOWED_MODEL!r} calling {DISALLOWED_MODEL!r} must be "
            f"denied 403, got {result.status_code}: {result.body[:300]}"
        )
        assert MODEL_ACCESS_DENIED_MARKER in result.body, (
            f"403 body must be a model-access denial, got: {result.body[:300]}"
        )

    def test_llm_only_key_forbidden_from_management_route_403(
        self, client: AccessControlClient, resources: ResourceManager
    ) -> None:
        key = client.llm_only_key()
        resources.defer(lambda: client.delete_key(key))
        result = client.create_model_status(key, f"e2e-forbidden-{unique_marker()}")
        assert result.status_code == 403, (
            f"llm-only key calling a management route must be denied 403, got "
            f"{result.status_code}: {result.body[:300]}"
        )
        assert ROUTE_NOT_ALLOWED_MARKER in result.body, (
            f"403 body must be a route-permission denial, got: {result.body[:300]}"
        )

    def test_unknown_model_returns_400(
        self, client: AccessControlClient, resources: ResourceManager
    ) -> None:
        key = resources.key()
        result = client.chat_status(
            key, f"nonexistent-model-{unique_marker()}", "hi this is a test"
        )
        assert result.status_code == 400, (
            f"unknown model must be rejected 400 before forwarding, got "
            f"{result.status_code}: {result.body[:300]}"
        )
        assert _is_json(result.body), f"400 body must be valid JSON: {result.body[:300]}"


class TestVirtualKeyAuth:
    """Virtual-key auth the way OpenAI-compatible clients send it: a real key
    must reach chat, a forged bearer must be rejected before the provider."""

    @pytest.mark.covers(
        "other.auth.virtual_key.valid_allows",
        "other.auth.virtual_key.invalid_denied",
        exercised_on=[],
    )
    def test_valid_key_allows_and_invalid_key_denied(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-auth-chat-{unique_marker()}"
        model_id = proxy.create_model(
            model,
            LiteLLMParamsBody(model=VIRTUAL_KEY_BACKEND, api_key="os.environ/ANTHROPIC_API_KEY"),
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        key = resources.key()

        ok = unwrap(
            proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=f"Reply with one word. {unique_marker()}",
                        )
                    ],
                    max_tokens=16,
                ),
            )
        )
        assert ok.choices, f"valid key must complete chat: {ok}"

        bad = proxy.chat(
            "sk-e2e-forged-not-a-real-key",
            ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content="should not run")],
                max_tokens=8,
            ),
        )
        match bad:
            case UnauthorizedError():
                return
            case UnknownApiError(status_code=status) if status in (401, 403):
                return
            case Success():
                pytest.fail("forged bearer must not reach a successful completion")
            case _:
                pytest.fail(f"forged bearer must be auth-denied, got {bad}")
