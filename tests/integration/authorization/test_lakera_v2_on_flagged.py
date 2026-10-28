import json
import uuid
from contextlib import ExitStack
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_CARD: Final = "4111 1111 1111 1111"
_PROMPT: Final = f"Please update the billing on my account to card {_CARD} and confirm when done."
_MASKED_PROMPT: Final = "Please update the billing on my account to card [MASKED CREDIT_CARD] and confirm when done."
_PII_ONLY_VERDICT: Final = json.dumps(
    {
        "flagged": True,
        "breakdown": [
            {"detector_type": "pii/credit_card", "detected": True, "message_id": 0},
            {"detector_type": "prompt_attack", "detected": False, "message_id": 0},
        ],
        "payload": [
            {
                "detector_type": "pii/credit_card",
                "start": _PROMPT.index(_CARD),
                "end": _PROMPT.index(_CARD) + len(_CARD),
                "text": _CARD,
                "message_id": 0,
            }
        ],
    }
).encode()
_PROVIDER_RESPONSE: Final = json.dumps(
    {
        "id": "chatcmpl-lakera-on-flagged",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "billing updated"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
).encode()


class _GuardrailCreated(BaseModel):
    model_config = ConfigDict(frozen=True)

    guardrail_id: str
    guardrail_name: str


def _create_lakera_guardrail(gateway: Gateway, cleanups: ExitStack, on_flagged: str, api_base: str) -> str:
    created: Final = gateway.request(
        "POST",
        "/guardrails",
        {
            "guardrail": {
                "guardrail_name": f"lakera-{on_flagged}-{uuid.uuid4().hex}",
                "litellm_params": {
                    "guardrail": "lakera_v2",
                    "mode": "pre_call",
                    "default_on": False,
                    "api_base": api_base,
                    "api_key": "synthetic-lakera-key",
                    "on_flagged": on_flagged,
                },
            }
        },
    )
    assert created.status_code == 200, created.text
    payload: Final = _GuardrailCreated.model_validate_json(created.content)
    cleanups.callback(gateway.request, "DELETE", f"/guardrails/{payload.guardrail_id}")
    return payload.guardrail_name


def _forwarded_prompts(requests: tuple[Request, ...]) -> tuple[JsonValue, ...]:
    return tuple(_JSON_OBJECT.validate_json(request.body)["messages"] for request in requests)


@pytest.mark.parametrize(
    ("on_flagged", "status", "forwarded"),
    [
        ("block", 400, ()),
        ("mask", 200, ([{"role": "user", "content": _MASKED_PROMPT}],)),
    ],
)
def test_lakera_v2_pii_only_detection_follows_on_flagged(
    gateway: Gateway, on_flagged: str, status: int, forwarded: tuple[JsonValue, ...]
) -> None:
    with (
        wire_server(lambda _request: Reply(body=_PROVIDER_RESPONSE)) as provider,
        wire_server(lambda _request: Reply(body=_PII_ONLY_VERDICT)) as lakera,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{provider.url}/v1", api_key="synthetic-provider-key")
        guardrail: Final = _create_lakera_guardrail(gateway, scenario.cleanups, on_flagged, lakera.url)

        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": _PROMPT}], "guardrails": [guardrail]},
        )

        assert response.status_code == status, response.text
        assert tuple(request.target for request in lakera.drain()) == ("/v2/guard",)
        assert _forwarded_prompts(provider.drain()) == forwarded
        if status == 400:
            error: Final = object_value(_JSON_OBJECT.validate_json(response.content)["error"])
            assert (error["message"], error["code"]) == ("Violated guardrail policy", "400"), response.text
