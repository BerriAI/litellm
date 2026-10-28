import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server

INPUT_COST_PER_TOKEN: Final = 0.001
OUTPUT_COST_PER_TOKEN: Final = 0.002
ROUND_USAGE: Final = ((10, 5), (20, 10), (30, 15))
ROUND_SPEND: Final = tuple(
    prompt * INPUT_COST_PER_TOKEN + completion * OUTPUT_COST_PER_TOKEN for prompt, completion in ROUND_USAGE
)
SESSION_SPEND: Final = sum(ROUND_SPEND)


def _round_reply(request: Request, prompt: str, usage: tuple[int, int]) -> Reply:
    assert request.method == "POST" and request.target == "/chat/completions", request.target
    assert json.loads(request.body) == {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
    }, request.body
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "round answer"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)},
            }
        ).encode()
    )


@pytest.mark.covers("spend.logs_ui.multi_round_session_total_spend_sums_every_round")
def test_logs_ui_session_total_spend_sums_every_round_of_a_multi_round_session(gateway: Gateway) -> None:
    session_id: Final = f"session-{uuid.uuid4().hex}"
    prompts: Final = tuple(f"round-{index}-{uuid.uuid4().hex}" for index in range(len(ROUND_USAGE)))
    usage_by_prompt: Final = dict(zip(prompts, ROUND_USAGE, strict=True))

    def provider(request: Request) -> Reply:
        prompt: Final = str(json.loads(request.body)["messages"][0]["content"])
        return _round_reply(request, prompt, usage_by_prompt[prompt])

    with wire_server(provider) as wire, gateway.scenario() as scenario:
        alias: Final = scenario.model(
            api_base=wire.url,
            input_cost_per_token=INPUT_COST_PER_TOKEN,
            output_cost_per_token=OUTPUT_COST_PER_TOKEN,
            num_retries=0,
        )
        request_ids: Final = tuple(_completed_round(gateway, alias, session_id, prompt) for prompt in prompts)
        assert len(wire.drain()) == len(ROUND_USAGE)
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT request_id, spend FROM "LiteLLM_SpendLogs" WHERE session_id=%s ORDER BY "startTime"',
                (session_id,),
            ),
            lambda values: len(values) == len(ROUND_USAGE),
            seconds=70,
        )
        assert [row["request_id"] for row in rows] == list(request_ids), rows
        assert [float(row["spend"]) for row in rows] == pytest.approx(list(ROUND_SPEND)), rows
        now: Final = datetime.now(timezone.utc)
        logs: Final = gateway.request(
            "GET",
            "/spend/logs/ui",
            params={
                "session_id": session_id,
                "start_date": (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
                "end_date": (now + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        assert logs.status_code == 200, logs.text
        page: Final = logs.json()
        assert page["total"] == len(ROUND_USAGE), logs.text
        assert sorted(row["request_id"] for row in page["data"]) == sorted(request_ids), logs.text
        assert [row["session_total_count"] for row in page["data"]] == [len(ROUND_USAGE)] * len(ROUND_USAGE), logs.text
        assert [row["session_total_spend"] for row in page["data"]] == pytest.approx(
            [SESSION_SPEND] * len(ROUND_USAGE)
        ), logs.text


def _completed_round(gateway: Gateway, alias: str, session_id: str, prompt: str) -> str:
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": alias, "messages": [{"role": "user", "content": prompt}], "litellm_trace_id": session_id},
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])
