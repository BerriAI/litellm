import json
from collections.abc import AsyncIterable, Callable, Mapping
from typing import Final, Literal

import httpx
import respx
from pydantic import TypeAdapter

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.router import Router
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

__all__ = [
    "REPLY_TEXT",
    "CountingGuardrail",
    "drain",
    "guarded_router",
    "mock_provider",
    "post_call_guardrail",
    "pre_call_guardrail",
]

_PRIMARY_API_BASE: Final = "https://primary.invalid/v1"
REPLY_TEXT: Final = "hi from the model"

_CHAT_REPLY: Final = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": REPLY_TEXT}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
_CHAT_SSE: Final = (
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":0,"model":"gpt-4o",'
    f'"choices":[{{"index":0,"delta":{{"role":"assistant","content":"{REPLY_TEXT}"}},"finish_reason":null}}]}}\n\n'
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":0,"model":"gpt-4o",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)
_RESPONSES_REPLY: Final = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "status": "completed",
    "model": "gpt-4o",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": REPLY_TEXT, "annotations": ()}],
        }
    ],
    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
}
_RESPONSES_SSE: Final = (
    f"data: {json.dumps({'type': 'response.created', 'sequence_number': 0, 'response': {**_RESPONSES_REPLY, 'status': 'in_progress', 'output': []}})}\n\n"
    f"data: {json.dumps({'type': 'response.output_text.delta', 'sequence_number': 1, 'item_id': 'msg_1', 'output_index': 0, 'content_index': 0, 'delta': REPLY_TEXT})}\n\n"
    f"data: {json.dumps({'type': 'response.completed', 'sequence_number': 2, 'response': _RESPONSES_REPLY})}\n\n"
)
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])


class CountingGuardrail(CustomGuardrail):
    checked: tuple[str, ...] = ()

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: object = None,
    ) -> GenericGuardrailAPIInputs:
        self.checked = (*self.checked, " ".join(inputs.get("texts") or ()))
        return inputs


def post_call_guardrail(name: str = "G", default_on: bool = False) -> CountingGuardrail:
    return CountingGuardrail(guardrail_name=name, event_hook=GuardrailEventHooks.post_call, default_on=default_on)


def pre_call_guardrail(name: str) -> CountingGuardrail:
    return CountingGuardrail(guardrail_name=name, event_hook=GuardrailEventHooks.pre_call)


def _deployment(model_name: str, guardrails: tuple[str, ...], failing: bool = False) -> dict[str, object]:
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": "openai/gpt-4o",
            "api_key": "sk-test",
            **({"api_base": _PRIMARY_API_BASE} if failing else {}),
            **({"guardrails": list(guardrails)} if guardrails else {}),
        },
    }


def guarded_router(primary_guardrails: tuple[str, ...], fallback_guardrails: tuple[str, ...] | None = None) -> Router:
    if fallback_guardrails is None:
        return Router(model_list=[_deployment("gpt", primary_guardrails)])
    return Router(
        model_list=[
            _deployment("gpt", primary_guardrails, failing=True),
            _deployment("gpt-backup", fallback_guardrails),
        ],
        fallbacks=[{"gpt": ["gpt-backup"]}],
        num_retries=0,
    )


def _reply(json_reply: Mapping[str, object], sse_reply: str) -> Callable[[httpx.Request], httpx.Response]:
    def reply(request: httpx.Request) -> httpx.Response:
        if _JSON_OBJECT.validate_json(request.content).get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse_reply.encode())
        return httpx.Response(200, json=dict(json_reply))

    return reply


def mock_provider(provider: respx.MockRouter) -> None:
    provider.post(f"{_PRIMARY_API_BASE}/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": {"message": "primary down"}})
    )
    provider.post(f"{_PRIMARY_API_BASE}/responses").mock(
        return_value=httpx.Response(500, json={"error": {"message": "primary down"}})
    )
    provider.post("https://api.openai.com/v1/chat/completions").mock(side_effect=_reply(_CHAT_REPLY, _CHAT_SSE))
    provider.post("https://api.openai.com/v1/responses").mock(side_effect=_reply(_RESPONSES_REPLY, _RESPONSES_SSE))


async def drain(stream: AsyncIterable[object]) -> str:
    return "".join([str(chunk) async for chunk in stream])
