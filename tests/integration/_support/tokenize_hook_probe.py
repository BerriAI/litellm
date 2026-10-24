from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Final

import httpx

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.tokenize import TokenizeResponse


async def _record(
    stage: str,
    data: Mapping[str, object],
    user_api_key_dict: UserAPIKeyAuth,
    response: TokenizeResponse | None = None,
    call_type: str | None = None,
) -> None:
    metadata: Final = data.get("metadata")
    body: Final = {
        "stage": stage,
        "call_id": data.get("litellm_call_id"),
        "call_type": call_type or data.get("call_type"),
        "prompt": data.get("prompt"),
        "user_id": user_api_key_dict.user_id,
        "user": data.get("user"),
        "metadata": (
            {"request_marker": metadata.get("request_marker")}
            if isinstance(metadata, Mapping)
            else None
        ),
        "count": response.count if response is not None else None,
        "model": response.model if response is not None else None,
        "token_ids": response.token_ids if response is not None else None,
        "tokens": response.tokens if response is not None else None,
        "hidden_params": dict(response._hidden_params) if response is not None else None,
    }
    async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
        probe_response: Final = await client.post(os.environ["TOKENIZE_HOOK_PROBE_URL"], json=body)
        probe_response.raise_for_status()


class TokenizeHookProbe(CustomLogger):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: object,
        data: dict[str, object],
        call_type: str,
    ) -> dict[str, object] | None:
        if call_type != "atokenize":
            return None
        rewritten: Final = {**data, "prompt": "rewritten by tokenization hook"}
        await _record("pre", rewritten, user_api_key_dict, call_type=call_type)
        return rewritten

    async def async_post_call_success_hook(
        self,
        data: dict[str, object],
        user_api_key_dict: UserAPIKeyAuth,
        response: object,
    ) -> object | None:
        if not isinstance(response, TokenizeResponse):
            return None
        await _record("post", data, user_api_key_dict, response)
        return response.model_copy(update={"model": "hook-replaced-model", "count": 777})


probe: Final = TokenizeHookProbe()
