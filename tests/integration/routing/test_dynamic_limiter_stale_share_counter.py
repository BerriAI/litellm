"""A team's usage above its share in one window must not follow it into the next: every worker asks Redis, and
none keeps the count it read for the share metric."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from integration.routing.test_payment_required_mapping import is_model_info_probe, model_list_reply
from integration.routing.test_per_model_rate_limit_fallbacks import chat, model_entry, proxy_config

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 180)

WINDOW_SECONDS: Final = 20
MODEL_TPM: Final = 3000
TEAM_TPM: Final = 100
REPLY_TOKENS: Final = 120
SATURATION_THRESHOLD: Final = 0.4
IDLE_CALLS: Final = 8
WARMUP_CALLS: Final = int(MODEL_TPM * SATURATION_THRESHOLD) // REPLY_TOKENS + 1


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    primary: str
    shared: str


def metered_reply(request: Request) -> Reply:
    if is_model_info_probe(request):
        return model_list_reply()
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl_stale_share",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "gpt-4.1-mini",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "metered"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": REPLY_TOKENS - 20, "total_tokens": REPLY_TOKENS},
            }
        ).encode()
    )


def provider_error_reply(request: Request) -> Reply:
    if is_model_info_probe(request):
        return model_list_reply()
    return Reply(status=500, body=b'{"error": {"message": "scripted provider outage", "type": "server_error"}}')


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    suffix: Final = uuid.uuid4().hex[:10]
    primary: Final = f"stale-share-primary-{suffix}"
    shared: Final = f"stale-share-shared-{suffix}"
    directory: Final = tmp_path_factory.mktemp("stale-share-counter")
    with ExitStack() as stack:
        rig_gateway: Final = stack.enter_context(gateway_from_environment())
        primary_wire: Final = stack.enter_context(wire_server(provider_error_reply))
        shared_wire: Final = stack.enter_context(wire_server(metered_reply))
        base: Final = proxy_config(
            directory,
            model_list=(model_entry(primary, primary_wire), model_entry(shared, shared_wire, tpm=MODEL_TPM)),
            fallbacks=({primary: [shared]},),
            general_settings={},
        )
        configuration: Final = yaml.safe_load(Path(base).read_text())
        configuration["litellm_settings"] = {
            **configuration["litellm_settings"],
            "callbacks": ["dynamic_rate_limiter_v3", "prometheus"],
            "priority_reservation": {"interactive": 0.6, "batch": 0.4},
            "priority_reservation_settings": {
                "default_priority": 1.0,
                "saturation_threshold": SATURATION_THRESHOLD,
                "model_limit_borrowing_config": {"borrow_idle_capacity": ["team"]},
            },
        }
        Path(base).write_text(yaml.safe_dump(configuration))
        owned: Final = stack.enter_context(
            owned_proxy_process(
                rig_gateway,
                directory,
                {"LITELLM_RATE_LIMIT_WINDOW_SIZE": str(WINDOW_SECONDS)},
                config=base,
                workers=2,
            )
        )
        yield Rig(owned.gateway, primary, shared)


def _wait_until(moment: float) -> None:
    time.sleep(max(0.0, moment - time.time()))


def _served(rig: Rig, key: str) -> int:
    replies: Final = tuple(chat(rig.gateway, key, rig.shared, f"warm-{uuid.uuid4().hex}") for _ in range(WARMUP_CALLS))
    return sum(reply.status_code == 200 for reply in replies)


def test_a_fallback_is_served_in_the_window_after_the_team_went_over_its_share(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        team: Final = scenario.team(metadata={"model_tpm_limit": {rig.shared: TEAM_TPM}})
        key: Final = scenario.key(team_id=team)
        other_key: Final = scenario.key(team_id=scenario.team(metadata={"priority": "batch"}))
        model_window_start: Final = time.time()
        first_warm_up: Final = _served(rig, other_key)
        _wait_until(model_window_start + WINDOW_SECONDS / 2)
        over_share: Final = chat(rig.gateway, key, rig.shared, f"over-{uuid.uuid4().hex}")
        share_window_end: Final = time.time() + WINDOW_SECONDS
        _wait_until(model_window_start + WINDOW_SECONDS + 1.5)
        idle: Final = tuple(chat(rig.gateway, key, rig.shared, f"idle-{uuid.uuid4().hex}") for _ in range(IDLE_CALLS))
        _wait_until(share_window_end + 1)
        second_warm_up: Final = _served(rig, other_key)
        fallback: Final = chat(rig.gateway, key, rig.primary, f"fallback-{uuid.uuid4().hex}")

    assert min(first_warm_up, second_warm_up) * REPLY_TOKENS >= MODEL_TPM * SATURATION_THRESHOLD
    assert over_share.status_code == 200, over_share.text
    assert REPLY_TOKENS > TEAM_TPM
    assert tuple(r.status_code for r in idle) == (200,) * IDLE_CALLS, [r.text for r in idle]
    assert fallback.status_code == 200, fallback.text
    assert fallback.headers["x-litellm-model-group"] == rig.shared, dict(fallback.headers)
