import json
from hashlib import sha256
from pathlib import Path
from typing import Final, Literal
from uuid import uuid4

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.openai_wire import answering_model_discovery
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.shares_worker_state

PROMPT_TOKENS: Final = 5
COMPLETION_TOKENS: Final = 3
INPUT_RATE: Final = 0.001
OUTPUT_RATE: Final = 0.002
TOKEN_PRICE: Final = PROMPT_TOKENS * INPUT_RATE + COMPLETION_TOKENS * OUTPUT_RATE
OVERFLOWING_INPUT_RATE: Final = 1e308
POISONED_UPSTREAM: Final = "poisoned-upstream"
CONTROL_UPSTREAM: Final = "control-upstream"
DAILY_TOTALS: Final = "SELECT model_group, SUM(spend)::float8 AS spend, SUM(api_requests)::int8 AS api_requests"
DAILY_USER_QUERY: Final = f'{DAILY_TOTALS} FROM "LiteLLM_DailyUserSpend" WHERE api_key=%s GROUP BY model_group'
DAILY_TEAM_QUERY: Final = f'{DAILY_TOTALS} FROM "LiteLLM_DailyTeamSpend" WHERE api_key=%s GROUP BY model_group'
DAILY_TAG_QUERY: Final = f'{DAILY_TOTALS} FROM "LiteLLM_DailyTagSpend" WHERE api_key=%s AND tag=%s GROUP BY model_group'

Poison = Literal["reported_nan_cost", "overflowing_price"]


def _frame(payload: dict[str, JsonValue]) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def _stream(upstream_model: str, reported_cost: float | None) -> tuple[bytes, ...]:
    chunk: Final = {"id": f"chatcmpl-{uuid4().hex}", "object": "chat.completion.chunk", "created": 1}
    usage: Final[dict[str, JsonValue]] = {
        "prompt_tokens": PROMPT_TOKENS,
        "completion_tokens": COMPLETION_TOKENS,
        "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
    }
    reported_usage: Final = usage if reported_cost is None else {**usage, "cost": reported_cost}
    return (
        _frame({**chunk, "model": upstream_model, "choices": [{"index": 0, "delta": {"content": "ok"}}]}),
        _frame(
            {
                **chunk,
                "model": upstream_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": reported_usage,
            }
        ),
        b"data: [DONE]\n\n",
    )


def _deployment(alias: str, upstream_model: str, wire_url: str, input_rate: float) -> dict[str, JsonValue]:
    return {
        "model_name": alias,
        "litellm_params": {
            "model": f"openai/{upstream_model}",
            "api_base": f"{wire_url}/v1",
            "api_key": "integration-provider-key",
            "input_cost_per_token": input_rate,
            "output_cost_per_token": OUTPUT_RATE,
        },
    }


def _proxy_config(directory: Path, deployments: tuple[dict[str, JsonValue], ...]) -> Path:
    config: Final = directory / "non_finite_cost_config.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": list(deployments),
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "disable_spend_logs": False,
                    "proxy_batch_write_at": 1,
                    "proxy_batch_polling_interval": 1,
                },
                "router_settings": {"disable_cooldowns": True},
            }
        )
    )
    return config


def _by_model_group(rows: list[dict[str, JsonValue]]) -> dict[str, float]:
    return {str(row["model_group"]): float(str(row["spend"])) for row in rows}


def _both_requests_rows(query: str, parameters: tuple[str, ...]) -> list[dict[str, JsonValue]]:
    return eventually(lambda: read_rows(query, parameters), lambda rows: len(rows) == 2, seconds=70)


@pytest.mark.parametrize("poison", ["reported_nan_cost", "overflowing_price"])
@pytest.mark.timeout(240)
def test_a_non_finite_cost_never_blocks_the_next_requests_spend_rows(
    gateway: Gateway, tmp_path: Path, poison: Poison
) -> None:
    poisoned_alias: Final = f"non-finite-cost-{uuid4().hex}"
    control_alias: Final = f"finite-cost-{uuid4().hex}"
    tag: Final = f"non-finite-cost-{uuid4().hex}"
    poisoned_spend: Final = TOKEN_PRICE if poison == "reported_nan_cost" else 0.0
    poisoned_input_rate: Final = INPUT_RATE if poison == "reported_nan_cost" else OVERFLOWING_INPUT_RATE

    def respond(request: Request) -> Reply:
        assert request.target.endswith("/chat/completions"), request.target
        upstream_model: Final = json.loads(request.body)["model"]
        reported_cost: Final = (
            float("nan") if upstream_model == POISONED_UPSTREAM and poison == "reported_nan_cost" else None
        )
        return Reply(content_type="text/event-stream", chunks=_stream(upstream_model, reported_cost))

    with wire_server(answering_model_discovery(respond)) as wire:
        config: Final = _proxy_config(
            tmp_path,
            (
                _deployment(poisoned_alias, POISONED_UPSTREAM, wire.url, poisoned_input_rate),
                _deployment(control_alias, CONTROL_UPSTREAM, wire.url, INPUT_RATE),
            ),
        )
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
            team: Final = scenario.team()
            key: Final = scenario.key(user_id=f"non-finite-cost-{uuid4().hex}", team_id=team)
            digest: Final = sha256(key.encode()).hexdigest()
            responses: Final = tuple(
                candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": alias,
                        "messages": [{"role": "user", "content": f"non-finite cost {uuid4().hex}"}],
                        "stream": True,
                        "metadata": {"tags": [tag]},
                    },
                    key=key,
                )
                for alias in (poisoned_alias, control_alias)
            )
            assert all(response.status_code == 200 for response in responses), [response.text for response in responses]

            expected: Final = {poisoned_alias: pytest.approx(poisoned_spend), control_alias: pytest.approx(TOKEN_PRICE)}
            spend_logs: Final = _both_requests_rows(
                'SELECT model_group, spend FROM "LiteLLM_SpendLogs" WHERE api_key=%s', (digest,)
            )
            assert _by_model_group(spend_logs) == expected, spend_logs
            daily_rows: Final = (
                _both_requests_rows(DAILY_USER_QUERY, (digest,)),
                _both_requests_rows(DAILY_TEAM_QUERY, (digest,)),
                _both_requests_rows(DAILY_TAG_QUERY, (digest, tag)),
            )
            assert [_by_model_group(rows) for rows in daily_rows] == [expected] * 3, daily_rows
            assert [{row["api_requests"] for row in rows} for rows in daily_rows] == [{1}] * 3, daily_rows
            key_total: Final = poisoned_spend + TOKEN_PRICE
            key_spend: Final = eventually(
                lambda: read_rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest,)),
                lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) == pytest.approx(key_total),
                seconds=70,
                return_last_on_timeout=True,
            )
            assert float(str(key_spend[0]["spend"])) == pytest.approx(key_total), key_spend
