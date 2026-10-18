"""A native Responses WebSocket runs the proxy's pre_call guardrails on every ``response.create`` frame, bills each
turn's spend row with the guardrail cost of that turn alone, and bills a blocked frame's guardrail scan as a failed
turn (LIT-9744)"""

import asyncio
import hashlib
import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import websockets
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.responses_ws_peer import PROVIDER_MODEL, USAGE, create, responses_peer, turn
from integration._support.tls import write_self_signed_cert
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.timeout(2 * graceful_stop_seconds() + 120), pytest.mark.shares_worker_state]

_INPUT_COST_PER_TOKEN: Final = 0.002
_OUTPUT_COST_PER_TOKEN: Final = 0.004
_COST_PER_WORD: Final = 0.1
_MARKER: Final = " [guarded]"
_BLOCKED_WORD: Final = "forbidden"
_GUARDRAIL_MODULE: Final = f"""
from litellm.integrations.custom_guardrail import CustomGuardrail


class TurnGuardrail(CustomGuardrail):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        blocks = [block for item in data.get("input") or [] for block in item.get("content") or []]
        words = sum(len(block["text"].split()) for block in blocks)
        blocked = any({_BLOCKED_WORD!r} in block["text"] for block in blocks)
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response={{"words": words}},
            request_data=data,
            guardrail_status="guardrail_intervened" if blocked else "success",
            tracing_detail={{"guardrail_cost": round({_COST_PER_WORD} * words, 10)}},
        )
        if blocked:
            raise ValueError("turn guardrail blocked the request")
        for block in blocks:
            block["text"] = block["text"] + {_MARKER!r}
        return data
"""


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    cert: tuple[Path, Path]


def _config(directory: Path) -> Path:
    (directory / "turn_guardrail.py").write_text(_GUARDRAIL_MODULE)
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["guardrails"] = [
        {
            "guardrail_name": "turn-guardrail",
            "litellm_params": {"guardrail": "turn_guardrail.TurnGuardrail", "mode": "pre_call", "default_on": True},
        }
    ]
    path: Final = directory / "responses-ws-guardrail.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("responses-ws-guardrail")
    cert: Final = write_self_signed_cert(directory)
    with (
        gateway_from_environment() as shared,
        owned_proxy(shared, directory, {"SSL_CERT_FILE": str(cert[0])}, config=_config(directory)) as owned,
    ):
        yield _Rig(owned, cert)


def _nested(frame: str) -> str:
    """The ``{"type": "response.create", "response": {...}}`` wire shape the WS layer accepts next to the flat one."""
    flat: Final = json.loads(frame)
    return json.dumps({"type": flat["type"], "response": {key: value for key, value in flat.items() if key != "type"}})


async def _session(
    proxy_url: str, key: str, model: str, frames: tuple[str, ...]
) -> tuple[tuple[dict[str, JsonValue], ...], ...]:
    proxy: Final = proxy_url.rstrip("/").replace("http://", "ws://")
    async with websockets.connect(
        f"{proxy}/v1/responses?model={model}",
        additional_headers={"Authorization": f"Bearer {key}"},
        open_timeout=10,
    ) as connection:
        return tuple([await turn(connection, frame) for frame in frames])


def _spend_rows(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT spend, status FROM "LiteLLM_SpendLogs" WHERE api_key = %s ORDER BY "startTime"',
        (hashlib.sha256(key.encode()).hexdigest(),),
    )


def _key_spend(key: str) -> float:
    (row,) = read_rows(
        'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token = %s', (hashlib.sha256(key.encode()).hexdigest(),)
    )
    return float(row["spend"])


def _llm_cost() -> float:
    return USAGE["input_tokens"] * _INPUT_COST_PER_TOKEN + USAGE["output_tokens"] * _OUTPUT_COST_PER_TOKEN


def test_every_turn_runs_the_pre_call_guardrail_and_bills_only_its_own_guardrail_cost(rig: _Rig) -> None:
    with responses_peer(rig.cert) as peer, rig.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{PROVIDER_MODEL}",
            api_base=peer.url,
            input_cost_per_token=_INPUT_COST_PER_TOKEN,
            output_cost_per_token=_OUTPUT_COST_PER_TOKEN,
        )
        key: Final = scenario.key(models=[model])
        tag: Final = uuid.uuid4().hex
        texts: Final = (
            f"one {tag}",
            f"one two {tag}",
            f"one two three {tag}",
            f"{_BLOCKED_WORD} {tag}",
            f"one {tag}",
            f"{_BLOCKED_WORD} again {tag}",
        )
        frames: Final = tuple(create(model, text) for text in texts[:-1]) + (_nested(create(model, texts[-1])),)
        turns: Final = asyncio.run(_session(str(rig.gateway.client.base_url), key, model, frames))

        assert [events[-1]["type"] for events in turns] == [
            "response.completed",
            "response.completed",
            "response.completed",
            "error",
            "response.completed",
            "error",
        ], [[event["type"] for event in events] for events in turns]
        assert "blocked" in str(turns[3][-1]["error"]), turns[3]

        forwarded: Final = tuple(peer.frames.get_nowait() for _ in range(peer.frames.qsize()))
        assert [frame["input"][0]["content"][0]["text"] for frame in forwarded] == [
            f"{text}{_MARKER}" for text in texts if _BLOCKED_WORD not in text
        ], forwarded

        rows: Final = eventually(lambda: _spend_rows(key), lambda values: len(values) == 6, seconds=70)
        assert [row["status"] for row in rows] == ["success", "success", "success", "failure", "success", "failure"], (
            rows
        )
        expected: Final = [
            (0.0 if _BLOCKED_WORD in text else _llm_cost()) + _COST_PER_WORD * len(text.split()) for text in texts
        ]
        assert [float(row["spend"]) for row in rows] == pytest.approx(expected), rows
        total: Final = sum(expected)
        assert eventually(
            lambda: _key_spend(key), lambda spend: abs(spend - total) < 1e-6, seconds=70
        ) == pytest.approx(total)
