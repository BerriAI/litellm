"""`general_settings.guardrail_log_skip_passed`, driven through a real proxy.

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
PASSES: Final = "skip-passed-passes"
MASKS: Final = "skip-passed-masks"
BLOCKS: Final = "skip-passed-blocks"
FLAGS: Final = "skip-passed-flags"
REWRITES: Final = "skip-passed-rewrites"
EDITS_IN_PLACE: Final = "skip-passed-edits-in-place"
REWRITES_RESPONSE: Final = "skip-passed-rewrites-response"
VENDOR_BLOCKS: Final = "skip-passed-vendor-blocks"
UNSET: Final = "unset"
CHAT: Final = "chat"
CHAT_STREAM: Final = "chat-stream"
MESSAGES: Final = "messages"
RESPONSES: Final = "responses"


def _chat_completion() -> dict[str, object]:
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5.4-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "synthetic"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _chat_completion_stream() -> bytes:
    completion_id: Final = "chatcmpl-" + uuid.uuid4().hex
    chunk: Final = {"id": completion_id, "object": "chat.completion.chunk", "created": 1, "model": "gpt-5.4-mini"}
    events: Final = (
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "synthetic"}}]},
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {**chunk, "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
    )
    return b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + b"data: [DONE]\n\n"


def _response() -> dict[str, object]:
    return {
        "id": "resp_" + uuid.uuid4().hex,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-5.4-mini",
        "output": [
            {
                "type": "message",
                "id": "msg_" + uuid.uuid4().hex,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "synthetic", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


def _provider(request: Request) -> Reply:
    if request.target.endswith("/v1/responses"):
        return Reply(body=json.dumps(_response()).encode())
    assert request.target.endswith("/v1/chat/completions"), request.target
    if json.loads(request.body).get("stream") is True:
        return Reply(chunks=(_chat_completion_stream(),), content_type="text/event-stream")
    return Reply(body=json.dumps(_chat_completion()).encode())


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


_REWRITING_CODE: Final = (
    "def apply_guardrail(inputs, request_data, input_type):\n"
    '    texts = list(inputs.get("texts") or [])\n'
    f'    masked = [text.replace("{AWS_KEY}", "[MASKED]") for text in texts]\n'
    "    if masked != texts:\n"
    "        return modify(texts=masked)\n"
    "    return allow()\n"
)


def _rewriting_guardrail(name: str) -> dict[str, object]:
    """A guardrail that masks through the shared guardrail logging, which reports no entity counts."""
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "custom_code",
            "mode": "pre_call",
            "default_on": False,
            "custom_code": _REWRITING_CODE,
        },
    }


_HOOKS_MODULE: Final = "skip_passed_hooks"
_HOOKS_CODE: Final = f'''from litellm.integrations.custom_guardrail import CustomGuardrail, log_guardrail_information


class EditsInPlace(CustomGuardrail):
    @log_guardrail_information
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        for message in data["messages"]:
            message["content"] = message["content"].replace("{AWS_KEY}", "[MASKED]")


class RewritesResponse(CustomGuardrail):
    @log_guardrail_information
    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        if "{AWS_KEY}" in str(data["messages"]):
            response.choices[0].message.content = "[MASKED]"
        return response
'''


def _hook_guardrail(name: str, hook: str, mode: str) -> dict[str, object]:
    """A guardrail whose hook returns no changed value: it edits what it was handed."""
    return {
        "guardrail_name": name,
        "litellm_params": {"guardrail": f"{_HOOKS_MODULE}.{hook}", "mode": mode, "default_on": False},
    }


def _vendor(request: Request) -> Reply:
    """The Lakera guard API: it flags a prompt attack on the AWS key and passes everything else."""
    assert request.target.endswith("/v2/guard"), request.target
    flagged: Final = AWS_KEY in request.body.decode()
    breakdown: Final = [{"detector_type": "prompt_attack", "detected": True, "message_id": 0}] if flagged else []
    return Reply(body=json.dumps({"flagged": flagged, "breakdown": breakdown, "payload": []}).encode())


def _vendor_guardrail(name: str, api_base: str) -> dict[str, object]:
    """A guardrail that logs its verdict with a success status and then rejects the request."""
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "lakera_v2",
            "mode": "pre_call",
            "default_on": False,
            "api_key": "synthetic-lakera-key",
            "api_base": api_base,
            "on_flagged": "block",
        },
    }


def _config(root: Path, name: str, skip_passed: bool | None, spend_log_writer: str, vendor_url: str) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"]["cache"] = False
    config["general_settings"]["use_spend_logs_ledger"] = spend_log_writer == "rust"
    if skip_passed is not None:
        config["general_settings"]["guardrail_log_skip_passed"] = skip_passed
    config["guardrails"] = [
        _content_filter(PASSES, "pre_call", "BLOCK"),
        _content_filter(MASKS, "pre_call", "MASK"),
        _content_filter(BLOCKS, "pre_call", "BLOCK"),
        _content_filter(FLAGS, "logging_only", "BLOCK"),
        _rewriting_guardrail(REWRITES),
        _hook_guardrail(EDITS_IN_PLACE, "EditsInPlace", "pre_call"),
        _hook_guardrail(REWRITES_RESPONSE, "RewritesResponse", "post_call"),
        _vendor_guardrail(VENDOR_BLOCKS, vendor_url),
    ]
    (root / f"{_HOOKS_MODULE}.py").write_text(_HOOKS_CODE)
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

    def send(self, endpoint: str, text: str, guardrail: str) -> httpx.Response:
        messages: Final = [{"role": "user", "content": text}]
        path, body = {
            CHAT: ("/v1/chat/completions", {"messages": messages}),
            CHAT_STREAM: ("/v1/chat/completions", {"messages": messages, "stream": True}),
            MESSAGES: ("/v1/messages", {"messages": messages, "max_tokens": 32}),
            RESPONSES: ("/v1/responses", {"input": text}),
        }[endpoint]
        return self.proxy.client.post(
            path,
            json={"model": self.model, "guardrails": [guardrail], **body},
            headers={"Authorization": f"Bearer {self.proxy.key}", "Connection": "close"},
        )

    def chat(self, text: str, guardrail: str) -> httpx.Response:
        return self.send(CHAT, text, guardrail)

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
        self,
        guardrail: str,
        texts: tuple[str, ...],
        statuses: tuple[int, ...],
        added: Counters,
        endpoint: str = CHAT,
        logged_as: str | None = None,
    ) -> int:
        """Send one request per text through `guardrail`, wait until the daily counters grew by `added`, and
        return how many index rows the guardrail gained. The worker database outlives each proxy, so every
        check is a difference from what was there before. `logged_as` is the id the rows carry when it is
        not the guardrail's listed id."""
        guardrail_id: Final = logged_as or self.guardrail_id(guardrail)
        before: Final = self.counters(guardrail_id)
        indexed: Final = self.indexed_requests(guardrail_id)
        assert tuple(self.send(endpoint, text, guardrail).status_code for text in texts) == statuses
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


def _rig(root: Path, skip_passed: bool | None, spend_log_writer: str = "rust") -> Iterator[Rig]:
    with (
        gateway_from_environment() as gateway,
        wire_server(_provider) as provider,
        wire_server(_vendor) as vendor,
        owned_proxy_process(
            gateway,
            root,
            {},
            config=_config(root, "guardrail-log-skip-passed.yaml", skip_passed, spend_log_writer, vendor.url),
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
    if request.param == UNSET:
        yield from _rig(tmp_path_factory.mktemp("guardrail-log-skip-passed"), None)
        return
    yield from _rig(tmp_path_factory.mktemp("guardrail-log-skip-passed"), True, str(request.param))


skips_passed: Final = pytest.mark.parametrize("rig", ("rust", "python"), indirect=True)
unset: Final = pytest.mark.parametrize("rig", (UNSET,), indirect=True)


@skips_passed
def test_a_passed_run_is_counted_but_writes_no_index_row(rig: Rig) -> None:
    clean: Final = tuple(f"clean prompt {index}" for index in range(3))
    assert rig.index_rows_added(PASSES, clean, (200, 200, 200), Counters(3, 3, 0, 0)) == 0


@skips_passed
def test_a_masked_run_keeps_its_index_row(rig: Rig) -> None:
    rig.provider.drain()
    texts: Final = ("clean prompt", f"the key is {AWS_KEY}")
    assert rig.index_rows_added(MASKS, texts, (200, 200), Counters(2, 2, 0, 0)) == 1
    forwarded: Final = tuple(request.body.decode() for request in rig.provider.drain())
    assert any("the key is" in body for body in forwarded), forwarded
    assert not any(AWS_KEY in body for body in forwarded), forwarded


@skips_passed
def test_a_masked_run_that_reports_no_entity_count_keeps_its_index_row(rig: Rig) -> None:
    """The shared guardrail logging records no guardrail id, so these rows are keyed by the guardrail's name."""
    rig.provider.drain()
    texts: Final = ("clean prompt", f"the key is {AWS_KEY}")
    assert rig.index_rows_added(REWRITES, texts, (200, 200), Counters(2, 2, 0, 0), logged_as=REWRITES) == 1
    forwarded: Final = tuple(request.body.decode() for request in rig.provider.drain())
    assert any("the key is [MASKED]" in body for body in forwarded), forwarded
    assert not any(AWS_KEY in body for body in forwarded), forwarded


@skips_passed
def test_a_pre_call_hook_that_edits_the_request_in_place_keeps_its_index_row(rig: Rig) -> None:
    rig.provider.drain()
    texts: Final = ("clean prompt", f"the key is {AWS_KEY}")
    added: Final = rig.index_rows_added(
        EDITS_IN_PLACE, texts, (200, 200), Counters(2, 2, 0, 0), logged_as=EDITS_IN_PLACE
    )
    assert added == 1
    forwarded: Final = tuple(request.body.decode() for request in rig.provider.drain())
    assert any("the key is [MASKED]" in body for body in forwarded), forwarded
    assert not any(AWS_KEY in body for body in forwarded), forwarded


@skips_passed
def test_a_post_call_hook_that_rewrites_the_response_keeps_its_index_row(rig: Rig) -> None:
    rewritten: Final = rig.chat(f"the key is {AWS_KEY}", REWRITES_RESPONSE)
    assert rewritten.status_code == 200, rewritten.text
    assert "[MASKED]" in rewritten.text, rewritten.text
    texts: Final = ("clean prompt", f"the key is {AWS_KEY}")
    added: Final = rig.index_rows_added(
        REWRITES_RESPONSE, texts, (200, 200), Counters(2, 2, 0, 0), logged_as=REWRITES_RESPONSE
    )
    assert added == 1


@skips_passed
def test_a_vendor_block_logged_with_a_success_status_keeps_its_index_row(rig: Rig) -> None:
    """The vendor guardrail logs a success status before it rejects, so the daily counters read both runs as passed."""
    texts: Final = ("clean prompt", f"the key is {AWS_KEY}")
    added: Final = rig.index_rows_added(VENDOR_BLOCKS, texts, (200, 400), Counters(2, 2, 0, 0), logged_as=VENDOR_BLOCKS)
    assert added == 1


@skips_passed
def test_a_blocked_run_keeps_its_index_row(rig: Rig) -> None:
    texts: Final = ("clean prompt", f"the key is {AWS_KEY}")
    assert rig.index_rows_added(BLOCKS, texts, (200, 400), Counters(2, 1, 1, 0)) == 1


@skips_passed
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


@skips_passed
@pytest.mark.parametrize("endpoint", (CHAT_STREAM, MESSAGES, RESPONSES))
def test_every_endpoint_drops_the_passed_run_and_keeps_the_blocked_one(rig: Rig, endpoint: str) -> None:
    clean: Final = ("clean prompt 0", "clean prompt 1")
    assert rig.index_rows_added(PASSES, clean, (200, 200), Counters(2, 2, 0, 0), endpoint) == 0
    assert rig.index_rows_added(BLOCKS, (f"the key is {AWS_KEY}",), (400,), Counters(1, 0, 1, 0), endpoint) == 1


@skips_passed
def test_a_value_set_in_the_config_file_cannot_be_changed_through_the_config_api(rig: Rig) -> None:
    changed: Final = rig.proxy.request(
        "POST", "/config/update", {"general_settings": {"guardrail_log_skip_passed": False}}
    )
    assert changed.status_code == 400, changed.text
    assert "guardrail_log_skip_passed" in changed.text
    assert rig.index_rows_added(PASSES, ("clean prompt",), (200,), Counters(1, 1, 0, 0)) == 0


def _rows_added_by_passed_runs(rig: Rig, runs: int) -> int:
    clean: Final = tuple(f"clean prompt {index}" for index in range(runs))
    return rig.index_rows_added(MASKS, clean, (200,) * runs, Counters(runs, runs, 0, 0))


@unset
def test_every_run_keeps_its_index_row_when_the_setting_is_unset(rig: Rig) -> None:
    clean: Final = tuple(f"clean prompt {index}" for index in range(3))
    assert rig.index_rows_added(PASSES, clean, (200, 200, 200), Counters(3, 3, 0, 0)) == 3


@unset
def test_the_setting_saved_and_cleared_through_the_config_api_applies_on_every_worker(rig: Rig) -> None:
    assert _rows_added_by_passed_runs(rig, 2) == 2

    rejected: Final = rig.proxy.request(
        "POST", "/config/update", {"general_settings": {"guardrail_log_skip_passed": ["blocked"]}}
    )
    assert rejected.status_code == 422, rejected.text
    saved: Final = rig.proxy.request(
        "POST", "/config/update", {"general_settings": {"guardrail_log_skip_passed": True}}
    )
    assert saved.status_code == 200, saved.text
    listed: Final = rig.proxy.request("GET", "/config/list", params={"config_type": "general_settings"})
    assert listed.status_code == 200, listed.text
    fields: Final = tuple(
        object_value(field)
        for field in listed.json()
        if object_value(field)["field_name"] == "guardrail_log_skip_passed"
    )
    assert [(field["field_type"], field["field_value"]) for field in fields] == [("Boolean", True)]

    eventually(lambda: _rows_added_by_passed_runs(rig, 4), lambda added: added == 0, seconds=180)
    assert _rows_added_by_passed_runs(rig, 8) == 0
    assert rig.index_rows_added(MASKS, (f"the key is {AWS_KEY}",), (200,), Counters(1, 1, 0, 0)) == 1

    turned_off: Final = rig.proxy.request(
        "POST", "/config/update", {"general_settings": {"guardrail_log_skip_passed": False}}
    )
    assert turned_off.status_code == 200, turned_off.text
    eventually(lambda: _rows_added_by_passed_runs(rig, 4), lambda added: added == 4, seconds=180)
    assert _rows_added_by_passed_runs(rig, 8) == 8

    cleared: Final = rig.proxy.request(
        "POST", "/config/field/delete", {"config_type": "general_settings", "field_name": "guardrail_log_skip_passed"}
    )
    assert cleared.status_code == 200, cleared.text
