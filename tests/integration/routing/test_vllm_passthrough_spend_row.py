import json
import uuid
from collections.abc import Callable
from typing import Final

import pytest
from integration._support.anthropic_sse import user_prompt
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.openai_wire import answering_model_discovery, chat_reply, posted_targets
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

_SERVED_MODEL: Final = "Qwen/Qwen2.5-0.5B-Instruct"
_TEXT: Final = "Hello"
_INPUT_COST: Final = 0.001
_OUTPUT_COST: Final = 0.002
_UPSTREAM_PROMPT_TOKENS: Final = 5
_UPSTREAM_COMPLETION_TOKENS: Final = 3


def _vllm_upstream(identity: str, marker: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/v1/chat/completions"), request
        body: Final = object_value(json.loads(request.body))
        assert user_prompt(body) == marker, body
        return chat_reply(identity, _SERVED_MODEL, _TEXT, stream=body.get("stream") is True)

    return answering_model_discovery(respond)


def _spend_rows(request_id: str) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(
            'SELECT status, call_type, model, model_group, custom_llm_provider, prompt_tokens, completion_tokens, spend '
            'FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
            (request_id,),
        ),
        lambda rows: len(rows) >= 1,
        seconds=70,
    )


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_vllm_passthrough_to_a_router_model_writes_one_spend_row_with_the_upstream_usage(
    gateway: Gateway, stream: bool
) -> None:
    marker: Final = "vllm-spend-" + uuid.uuid4().hex
    identity: Final = f"chatcmpl-{marker}"
    with wire_server(_vllm_upstream(identity, marker)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"hosted_vllm/{_SERVED_MODEL}",
            api_base=wire.url + "/v1",
            input_cost_per_token=_INPUT_COST,
            output_cost_per_token=_OUTPUT_COST,
        )
        response: Final = gateway.request(
            "POST",
            "/vllm/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker}], **({"stream": True} if stream else {})},
        )
        assert response.status_code == 200, response.text
        assert identity in response.text, response.text
        assert posted_targets(wire) == ("/v1/chat/completions",)

        rows: Final = _spend_rows(identity)
        assert len(rows) == 1, rows
        assert rows[0] == {
            "status": "success",
            "call_type": "allm_passthrough_route",
            "model": f"hosted_vllm/{_SERVED_MODEL}",
            "model_group": model,
            "custom_llm_provider": "hosted_vllm",
            "prompt_tokens": _UPSTREAM_PROMPT_TOKENS,
            "completion_tokens": _UPSTREAM_COMPLETION_TOKENS,
            "spend": pytest.approx(_UPSTREAM_PROMPT_TOKENS * _INPUT_COST + _UPSTREAM_COMPLETION_TOKENS * _OUTPUT_COST),
        }, rows
