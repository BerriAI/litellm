import json
import re
from typing import Final

import pytest
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.guardrails.guardrail_hooks.presidio import OPTIONAL_PresidioPIIMasking
from litellm.types.guardrails import PiiAction, PiiEntityType
from tests._wait_helpers import await_until

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_CARD_NUMBER: Final = re.compile(r"\d{4}-\d{4}-\d{4}-\d{4}")
_CHAT_BODY: Final = {
    "id": "c",
    "object": "chat.completion",
    "created": 1,
    "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "144"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}
_CHAT_FRAMES: Final = (
    b'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m",'
    b'"choices":[{"index":0,"delta":{"role":"assistant","content":"144"},"finish_reason":null}]}\n\n',
    b'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m",'
    b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
    b"data: [DONE]\n\n",
)


def _presidio(request: Request) -> Reply:
    text: Final = str(_JSON_OBJECT.validate_json(request.body)["text"])
    cards: Final = tuple(_CARD_NUMBER.finditer(text))
    if request.target.endswith("/analyze"):
        spans: Final = [
            {"entity_type": "CREDIT_CARD", "start": card.start(), "end": card.end(), "score": 0.9} for card in cards
        ]
        return Reply(body=json.dumps(spans).encode())
    anonymized: Final = {
        "text": _CARD_NUMBER.sub("<CREDIT_CARD>", text),
        "items": [{"entity_type": "CREDIT_CARD", "operator": "replace"} for _ in cards],
    }
    return Reply(body=json.dumps(anonymized).encode())


def _upstream(request: Request) -> Reply:
    if _JSON_OBJECT.validate_json(request.body).get("stream") is True:
        return Reply(content_type="text/event-stream", chunks=_CHAT_FRAMES)
    return Reply(body=json.dumps(_CHAT_BODY).encode())


class _LoggedInputs(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: Final[list[object]] = []

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.inputs.append(kwargs.get("messages"))


@pytest.mark.parametrize(("entrypoint", "stream"), [("agenerate_content", False), ("agenerate_content_stream", True)])
async def test_generate_content_adapter_prompt_is_masked_by_a_logging_only_guardrail(
    monkeypatch: pytest.MonkeyPatch, entrypoint: str, stream: bool
) -> None:
    logged: Final = _LoggedInputs()
    with wire_server(_presidio) as presidio, wire_server(_upstream) as upstream:
        guardrail: Final = OPTIONAL_PresidioPIIMasking(
            guardrail_name="mask-logs",
            default_on=True,
            logging_only=True,
            presidio_analyzer_api_base=presidio.url,
            presidio_anonymizer_api_base=presidio.url,
            pii_entities_config={PiiEntityType.CREDIT_CARD: PiiAction.MASK},
        )
        monkeypatch.setattr(litellm, "callbacks", [guardrail, logged])
        router: Final = litellm.Router(
            model_list=[
                {
                    "model_name": "grok-group",
                    "litellm_params": {"model": "hosted_vllm/grok", "api_base": f"{upstream.url}/v1", "api_key": "k"},
                }
            ]
        )
        response: Final = await getattr(router, entrypoint)(
            model="grok-group", contents=[{"role": "user", "parts": [{"text": "My card is 4111-1111-1111-1111"}]}]
        )
        if stream:
            async for _ in response:
                pass
        await await_until(lambda: bool(logged.inputs), message="the request's success callback ran")
        await guardrail._close_http_session()

    assert logged.inputs == [[{"role": "user", "content": "My card is <CREDIT_CARD>"}]]
