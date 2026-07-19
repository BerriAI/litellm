"""Live e2e: virtual-key auth the way OpenAI-compatible clients send it.

A real key must reach chat; a forged bearer must be rejected before the provider.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import Success, UnauthorizedError, UnknownApiError, unwrap
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, LiteLLMParamsBody
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e

BACKEND = "anthropic/claude-haiku-4-5-20251001"


class TestVirtualKeyAuth:
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
            LiteLLMParamsBody(model=BACKEND, api_key="os.environ/ANTHROPIC_API_KEY"),
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
