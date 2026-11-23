"""`general_settings.guardrail_log_outcomes`, driven through a real proxy.

The provider is the only double, an owned wire peer. The proxy, its guardrail registry, Postgres and Redis run
for real with two workers. Every check reads `LiteLLM_SpendLogGuardrailIndex` and `LiteLLM_DailyGuardrailMetrics`,
which the spend log writer fills after the request has finished
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value, string_value
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server

pytestmark: Final = pytest.mark.shares_worker_state

AWS_KEY: Final = "AKIA" + "SYNTHETIC0000000"
PASSES: Final = "outcomes-passes"
MASKS: Final = "outcomes-masks"
BLOCKS: Final = "outcomes-blocks"
FLAGS: Final = "outcomes-flags"
KEPT_OUTCOMES: Final = ("blocked", "flagged", "masked")
UNSET: Final = "unset"


def _provider(request: Request) -> Reply:
    assert request.target.endswith("/v1/chat/completions"), request.target
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-5.4-mini",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "synthetic"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        ).encode()
    )


def _content_filter(name: str, mode: str, action: str) -> dict[str, object]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "litellm_content_filter",
            "mode": mode,
            "default_on": False,
            "patterns": [
                {"pattern_type": "regex", "pattern": "AKIA[0-9A-Z]{16}", "name": "aws_access_key", "action": action}
            ],
        },
    }


def _config(root: Path, name: str, outcomes: tuple[str, ...] | None, spend_log_writer: str) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"]["cache"] = False
    config["general_settings"]["use_spend_logs_ledger"] = spend_log_writer == "rust"
    if outcomes is not None:
        config["general_settings"]["guardrail_log_outcomes"] = list(outcomes)
    config["guardrails"] = [
        _content_filter(PASSES, "pre_call", "BLOCK"),
        _content_filter(MASKS, "pre_call", "MASK"),
        _content_filter(BLOCKS, "pre_call", "BLOCK"),
        _content_filter(FLAGS, "logging_only", "BLOCK"),
    ]
    path: Final = root / name
    path.write_text(yaml.safe_dump(config))
    return path


@dataclass(frozen=True, slots=True)
class Counters:
    evaluated: int
    passed: int
    blocked: int
    flagged: int


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    provider: Wire
    model: str

    def guardrail_id(self, name: str) -> str:
        listed: Final = self.proxy.get("/v2/guardrails/list")["guardrails"]
        assert isinstance(listed, list), listed
        matches: Final = tuple(object_value(row) for row in listed if object_value(row)["guardrail_name"] == name)
        assert len(matches) == 1, listed
        return string_value(matches[0]["guardrail_id"])

    def chat(self, text: str, guardrail: str) -> httpx.Response:
        return self.proxy.client.post(
            "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": text}],
                "guardrails": [guardrail],
            },
            headers={"Authorization": f"Bearer {self.proxy.key}", "Connection": "close"},
        )

    def counters(self, guardrail_id: str) -> Counters:
        rows: Final = self.proxy.rows(
            "SELECT COALESCE(SUM(requests_evaluated), 0)::int AS evaluated, COALESCE(SUM(passed_count), 0)::int AS passed, "
            "COALESCE(SUM(blocked_count), 0)::int AS blocked, COALESCE(SUM(flagged_count), 0)::int AS flagged "
            'FROM "LiteLLM_DailyGuardrailMetrics" WHERE guardrail_id = %s',
            (guardrail_id,),
        )
        row: Final = rows[0]
        return Counters(
            int(str(row["evaluated"])), int(str(row["passed"])), int(str(row["blocked"])), int(str(row["flagged"]))
        )

    def index_rows_added(
        self, guardrail: str, texts: tuple[str, ...], statuses: tuple[int, ...], added: Counters
    ) -> int:
        """Send one request per text through `guardrail`, wait until the daily counters grew by `added`, and
        return how many index rows the guardrail gained. The worker database outlives each proxy, so every
        check is a difference from what was there before."""
        guardrail_id: Final = self.guardrail_id(guardrail)
        before: Final = self.counters(guardrail_id)
        indexed: Final = self.indexed_requests(guardrail_id)
        assert tuple(self.chat(text, guardrail).status_code for text in texts) == statuses
        expected: Final = Counters(
            before.evaluated + added.evaluated,
            before.passed + added.passed,
            before.blocked + added.blocked,
            before.flagged + added.flagged,
        )
        eventually(lambda: self.counters(guardrail_id), lambda seen: seen == expected, seconds=90)
        return self.indexed_requests(guardrail_id) - indexed

    def indexed_requests(self, guardrail_id: str) -> int:
        rows: Final = self.proxy.rows(
            'SELECT COUNT(*)::int AS total FROM "LiteLLM_SpendLogGuardrailIndex" WHERE guardrail_id = %s',
            (guardrail_id,),
        )
        return int(str(rows[0]["total"]))


def _rig(root: Path, outcomes: tuple[str, ...] | None, spend_log_writer: str = "rust") -> Iterator[Rig]:
    with (
        gateway_from_environment() as gateway,
        wire_server(_provider) as provider,
        owned_proxy_process(
            gateway,
            root,
            {},
            config=_config(root, "guardrail-log-outcomes.yaml", outcomes, spend_log_writer),
            workers=2,
        ) as owned,
        owned.gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model="openai/gpt-5.4-mini", api_base=provider.url + "/v1", api_key="synthetic-openai-key"
        )
        yield Rig(owned.gateway, provider, model)


@pytest.fixture(scope="module")
def rig(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    """One proxy at a time: the owned proxies share the worker database and the guardrail ids, so two alive at
    once would move each other's counts."""
    outcomes: Final = None if request.param == UNSET else KEPT_OUTCOMES
    writer: Final = "rust" if request.param == UNSET else str(request.param)
    yield from _rig(tmp_path_factory.mktemp("guardrail-log-outcomes"), outcomes, writer)


kept_outcomes: Final = pytest.mark.parametrize("rig", ("rust", "python"), indirect=True)
unset: Final = pytest.mark.parametrize("rig", (UNSET,), indirect=True)


@kept_outcomes
def test_a_passed_run_is_counted_but_writes_no_index_row(rig: Rig) -> None:
    clean: Final = tuple(f"clean prompt {index}" for index in range(3))
    assert rig.index_rows_added(PASSES, clean, (200, 200, 200), Counters(3, 3, 0, 0)) == 0


@kept_outcomes
def test_a_masked_run_keeps_its_index_row(rig: Rig) -> None:
    rig.provider.drain()
    texts: Final = ("clean prompt", f"the key is {AWS_KEY}")
    assert rig.index_rows_added(MASKS, texts, (200, 200), Counters(2, 2, 0, 0)) == 1
    forwarded: Final = tuple(request.body.decode() for request in rig.provider.drain())
    assert any("the key is" in body for body in forwarded), forwarded
    assert not any(AWS_KEY in body for body in forwarded), forwarded


@kept_outcomes
def test_a_blocked_run_keeps_its_index_row(rig: Rig) -> None:
    texts: Final = ("clean prompt", f"the key is {AWS_KEY}")
    assert rig.index_rows_added(BLOCKS, texts, (200, 400), Counters(2, 1, 1, 0)) == 1


@kept_outcomes
def test_a_flagged_run_keeps_its_index_row(rig: Rig) -> None:
    guardrail_id: Final = rig.guardrail_id(FLAGS)
    before: Final = rig.counters(guardrail_id)
    flagged: Final = rig.chat(f"the key is {AWS_KEY}", FLAGS)
    assert flagged.status_code == 200, flagged.text
    response_id: Final = string_value(object_value(flagged.json())["id"])

    eventually(lambda: rig.counters(guardrail_id), lambda seen: seen.flagged == before.flagged + 1, seconds=90)
    rows: Final = rig.proxy.rows(
        'SELECT request_id FROM "LiteLLM_SpendLogGuardrailIndex" WHERE guardrail_id = %s AND request_id = %s',
        (guardrail_id, response_id),
    )
    assert len(rows) == 1, rows


def _rows_added_by_passed_runs(rig: Rig, runs: int) -> int:
    clean: Final = tuple(f"clean prompt {index}" for index in range(runs))
    return rig.index_rows_added(MASKS, clean, (200,) * runs, Counters(runs, runs, 0, 0))


@unset
def test_every_run_keeps_its_index_row_when_the_setting_is_unset(rig: Rig) -> None:
    clean: Final = tuple(f"clean prompt {index}" for index in range(3))
    assert rig.index_rows_added(PASSES, clean, (200, 200, 200), Counters(3, 3, 0, 0)) == 3


@unset
def test_outcomes_saved_and_cleared_through_the_config_api_apply_on_every_worker(rig: Rig) -> None:
    assert _rows_added_by_passed_runs(rig, 2) == 2

    rejected: Final = rig.proxy.request(
        "POST", "/config/update", {"general_settings": {"guardrail_log_outcomes": ["blocked", "no-such-outcome"]}}
    )
    assert rejected.status_code == 422, rejected.text
    saved: Final = rig.proxy.request(
        "POST", "/config/update", {"general_settings": {"guardrail_log_outcomes": ["blocked", "masked"]}}
    )
    assert saved.status_code == 200, saved.text
    listed: Final = rig.proxy.request("GET", "/config/list", params={"config_type": "general_settings"})
    assert listed.status_code == 200, listed.text
    fields: Final = tuple(
        object_value(field) for field in listed.json() if object_value(field)["field_name"] == "guardrail_log_outcomes"
    )
    assert [(field["field_type"], field["field_value"]) for field in fields] == [("List", ["blocked", "masked"])]

    eventually(lambda: _rows_added_by_passed_runs(rig, 4), lambda added: added == 0, seconds=180)
    assert _rows_added_by_passed_runs(rig, 8) == 0
    assert rig.index_rows_added(MASKS, (f"the key is {AWS_KEY}",), (200,), Counters(1, 1, 0, 0)) == 1

    cleared: Final = rig.proxy.request(
        "POST", "/config/field/delete", {"config_type": "general_settings", "field_name": "guardrail_log_outcomes"}
    )
    assert cleared.status_code == 200, cleared.text
    eventually(lambda: _rows_added_by_passed_runs(rig, 4), lambda added: added == 4, seconds=180)
    assert _rows_added_by_passed_runs(rig, 8) == 8
