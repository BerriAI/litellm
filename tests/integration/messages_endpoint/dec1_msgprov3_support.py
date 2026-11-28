"""Shared rig for the third round of `/v1/messages` per-provider contracts (`test_dec1_msgprov3_*`).

Round one (`dec1_msgprov_support.py`) and round two (`dec1_msgprov2_support.py`) built the owned-peer
deployments for Bedrock InvokeModel and Converse, Vertex AI Claude, Azure AI Foundry and the chat bridge. This
module only adds what round three needs on top of them, and imports everything else unchanged:

- `raw_path`, the request path exactly as the provider received it (percent escapes kept), because Bedrock's
  model id encoding (`:` kept as is, an ARN or a `model_id` override percent-encoded) is only visible there;
- `bridged_deployment`, one deployment per provider that `/v1/messages` reaches through the chat bridge
  (`LiteLLMMessagesToCompletionTransformationHandler`): Bedrock Converse, Gemini AI Studio, hosted vLLM and
  OpenAI (which lands on the Responses API for the `gpt-5.4-mini` family);
- a minimal OpenAI Responses answer and echo, since the OpenAI bridge talks `/responses`, not chat.

Every upstream is an owned loopback peer (`wire_server`); no request leaves the machine.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from typing import Final
from urllib.parse import urlsplit

from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.messages_endpoint.dec1_messages_support import Rig
from integration.messages_endpoint.dec1_msgprov2_support import HOSTED_VLLM_BACKEND, HOSTED_VLLM_KEY
from integration.messages_endpoint.dec1_msgprov_support import (
    BEDROCK_MODEL_ID,
    BEDROCK_TOKEN,
    chat_echo,
    converse_echo,
    path_of,
    provider_deployment,
)
from integration.providers.dec1_gemini_support import FLASH, GEMINI_KEY, gemini_echo
from pydantic import JsonValue

OPENAI_BACKEND: Final = "gpt-5.4-mini"
OPENAI_KEY: Final = "synthetic-openai-key"

BRIDGED_PROVIDERS: Final = ("bedrock_converse", "gemini", "hosted_vllm", "openai")
"""Providers whose `/v1/messages` traffic goes through the chat bridge (`Target.CHAT_COMPLETIONS` or
`Target.RESPONSES`, both `Rollout.PYTHON_ONLY` in `litellm/rust_bridge/catalog.py:203-207`)."""

BRIDGED_SPEND_PROVIDER: Final[Mapping[str, str]] = {
    "bedrock_converse": "bedrock",
    "gemini": "gemini",
    "hosted_vllm": "hosted_vllm",
    "openai": "openai",
}
"""The `custom_llm_provider` column each bridged provider writes on its SpendLogs row."""


def raw_path(request: Request) -> str:
    """The request path with its percent escapes intact, as Bedrock's router sees it."""
    return urlsplit(request.target).path


def responses_json(*, text: str = "pong", input_tokens: int = 11, output_tokens: int = 7) -> dict[str, JsonValue]:
    """A completed OpenAI Responses API answer with one assistant message."""
    identity: Final = uuid.uuid4().hex
    return {
        "id": f"resp_{identity}",
        "object": "response",
        "created_at": 1789788253,
        "status": "completed",
        "model": OPENAI_BACKEND,
        "output": [
            {
                "type": "message",
                "id": f"msg_{identity}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def responses_echo(_request: Request) -> Reply:
    return Reply(body=json.dumps(responses_json()).encode())


def is_model_call(provider: str, request: Request) -> bool:
    """Whether a peer request is the provider's inference call, not a model-list probe."""
    if request.method != "POST":
        return False
    path: Final = path_of(request)
    match provider:
        case "gemini":
            return ":generateContent" in path or ":streamGenerateContent" in path
        case "openai":
            return path.endswith(("/responses", "/chat/completions"))
        case _:
            return True


def bridged_deployment(
    gateway: Gateway,
    provider: str,
    respond: Callable[[Request], Reply] | None = None,
    **params: JsonValue,
) -> AbstractContextManager[Rig]:
    """One deployment of a bridged provider behind an owned peer; `respond` defaults to that provider's echo.

    Every echo reports 11 prompt and 7 completion tokens, so with the round-one `PRICES` one call costs 0.025.
    """
    match provider:
        case "bedrock_converse":
            return provider_deployment(
                gateway,
                respond or converse_echo,
                model=f"bedrock/converse/{BEDROCK_MODEL_ID}",
                api_key=BEDROCK_TOKEN,
                aws_region_name="us-east-1",
                **params,
            )
        case "gemini":
            return provider_deployment(
                gateway,
                respond or gemini_echo,
                model=f"gemini/{FLASH}",
                base_suffix="/v1beta",
                api_key=GEMINI_KEY,
                **params,
            )
        case "hosted_vllm":
            return provider_deployment(
                gateway,
                respond or chat_echo(HOSTED_VLLM_BACKEND),
                model=f"hosted_vllm/{HOSTED_VLLM_BACKEND}",
                base_suffix="/v1",
                api_key=HOSTED_VLLM_KEY,
                **params,
            )
        case "openai":
            return provider_deployment(
                gateway,
                respond or responses_echo,
                model=f"openai/{OPENAI_BACKEND}",
                api_key=OPENAI_KEY,
                **params,
            )
        case _:
            raise AssertionError(provider)


def model_calls(rig: Rig, provider: str) -> tuple[Request, ...]:
    """Inference calls the peer received for this rig (drains the queue)."""
    return tuple(request for request in rig.wire.drain() if is_model_call(provider, request))
