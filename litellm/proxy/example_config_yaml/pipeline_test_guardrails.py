"""
Test guardrails for pipeline E2E testing.

- StrictFilter: blocks any message containing "bad" (case-insensitive)
- PermissiveFilter: always passes (simulates an advanced guardrail that is more lenient)
"""

from typing import Optional, Union

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral


class StrictFilter(CustomGuardrail):
    """Blocks any message containing the word 'bad'."""

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Optional[Union[Exception, str, dict]]:
        for msg in data.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, str) and "bad" in content.lower():
                verbose_proxy_logger.info("StrictFilter: BLOCKED - found 'bad'")
                raise HTTPException(
                    status_code=400,
                    detail="StrictFilter: content contains forbidden word 'bad'",
                )
        verbose_proxy_logger.info("StrictFilter: PASSED")
        return data


class PermissiveFilter(CustomGuardrail):
    """Always passes - simulates a lenient advanced guardrail."""

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Optional[Union[Exception, str, dict]]:
        verbose_proxy_logger.info("PermissiveFilter: PASSED (always passes)")
        return data


class AlwaysBlockFilter(CustomGuardrail):
    """Always blocks - for testing full escalation->block path."""

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Optional[Union[Exception, str, dict]]:
        verbose_proxy_logger.info("AlwaysBlockFilter: BLOCKED")
        raise HTTPException(
            status_code=400,
            detail="AlwaysBlockFilter: all content blocked",
        )
