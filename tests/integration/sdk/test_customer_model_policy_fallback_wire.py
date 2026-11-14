import json
from typing import Final

import pytest
from integration._support.openai_wire import chat_reply
from integration._support.wire import Reply, Request, wire_server

import litellm
from litellm import Router
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.fallback_model_access import RouterFallbackAccessCheck
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup


def _peer(request: Request) -> Reply:
    if request.target.startswith("/primary/"):
        return Reply(
            status=401,
            body=json.dumps(
                {"error": {"message": "controlled primary failure", "type": "invalid_request_error"}}
            ).encode(),
        )
    return chat_reply("chatcmpl-customer-fallback", "gpt-4o-mini", "controlled fallback", stream=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_fallback", (False, True))
@pytest.mark.parametrize("enforce_key_access", (False, True))
async def test_customer_policy_survives_the_actual_router_fallback(
    allow_fallback: bool, enforce_key_access: bool
) -> None:
    original: Final = UserAPIKeyAuth(models=["primary", "backup"])
    scoped: Final = original.with_customer_model_scopes((("primary", "backup") if allow_fallback else ("primary",),))
    carried: Final = LiteLLMProxyRequestSetup.auth_for_request_metadata(scoped)
    with wire_server(_peer) as wire:
        router: Final = Router(
            model_list=[
                {
                    "model_name": name,
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "k",
                        "api_base": f"{wire.url}/{name}",
                        "max_retries": 0,
                    },
                }
                for name in ("primary", "backup")
            ],
            fallbacks=[{"primary": ["backup"]}],
            num_retries=0,
            fallback_access_check=RouterFallbackAccessCheck(is_enforced=lambda: enforce_key_access),
        )
        if allow_fallback:
            response: Final = await router.acompletion(
                model="primary",
                messages=[{"role": "user", "content": "controlled"}],
                metadata={"user_api_key_auth": carried},
            )
            assert response.choices[0].message.content == "controlled fallback"
        else:
            with pytest.raises(litellm.AuthenticationError):
                await router.acompletion(
                    model="primary",
                    messages=[{"role": "user", "content": "controlled"}],
                    metadata={"user_api_key_auth": carried},
                )
        posted: Final = tuple(request for request in wire.drain() if request.method == "POST")
        attempts: Final = tuple(request.target for request in posted)
        assert all(b"customer_model_scopes" not in request.body for request in posted)
    assert attempts == (
        ("/primary/chat/completions", "/backup/chat/completions") if allow_fallback else ("/primary/chat/completions",)
    )
    assert original.customer_model_scopes == ()
