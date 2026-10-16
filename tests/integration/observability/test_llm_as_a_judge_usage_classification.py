"""LIT-8332: llm_as_a_judge gains sample_rate (per litellm_call_id) and max_input_chars."""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, list_value, object_value
from integration._support.database import read_rows, write_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

WORK_MODEL: Final = "lit8332-work-model"
JUDGE_MODEL: Final = "lit8332-judge-model"
CALL_ID_HEADER: Final = "x-litellm-call-id"
RUN_TOKEN: Final = uuid.uuid4().hex[:8]
CRITERIA: Final = [{"name": "work", "description": "Is this request work related?", "weight": 100}]
BASE_JUDGE_PARAMS: Final = {
    "guardrail": "llm_as_a_judge",
    "judge_model": JUDGE_MODEL,
    "criteria": CRITERIA,
    "on_failure": "log",
    "default_on": True,
}


def _chat_completion(content: str, model: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        ).encode()
    )


def _verdict(score: float) -> str:
    return json.dumps(
        {
            "verdicts": [
                {
                    "criterion_name": "work",
                    "score": score,
                    "reasoning": "synthetic scripted verdict",
                    "passed": score >= 80,
                    "weight": 100,
                }
            ],
            "overall_score": score,
        }
    )


def _judge_scoring(score: float) -> Callable[[Request], Reply]:
    def handler(request: Request) -> Reply:
        assert request.target == "/v1/chat/completions", request.target
        return _chat_completion(_verdict(score), model=JUDGE_MODEL)

    return handler


def _work_handler(request: Request) -> Reply:
    assert request.target == "/v1/chat/completions", request.target
    return _chat_completion("synthetic work answer", model=WORK_MODEL)


def _write_config(
    tmp_path: Path, work_url: str, judge_url: str, guardrail_params: Mapping[str, JsonValue]
) -> tuple[Path, str]:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["model_list"] = [
        {
            "model_name": WORK_MODEL,
            "litellm_params": {
                "model": f"openai/{WORK_MODEL}",
                "api_base": f"{work_url}/v1",
                "api_key": "synthetic-work-key",
            },
        },
        {
            "model_name": JUDGE_MODEL,
            "litellm_params": {
                "model": f"openai/{JUDGE_MODEL}",
                "api_base": f"{judge_url}/v1",
                "api_key": "synthetic-judge-key",
            },
        },
    ]
    config.pop("vector_store_registry", None)
    config["litellm_settings"]["cache"] = False
    config["litellm_settings"].pop("cache_params", None)
    guardrail_name: Final = f"lit8332-{uuid.uuid4().hex}"
    config["guardrails"] = [{"guardrail_name": guardrail_name, "litellm_params": dict(guardrail_params)}]
    path: Final = tmp_path / "judge.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, guardrail_name


def _send(candidate: Gateway, call_id: str, *, key: str | None = None, text: str = "synthetic request") -> None:
    response: Final = candidate.request(
        "POST",
        "/v1/chat/completions",
        {"model": WORK_MODEL, "messages": [{"role": "user", "content": text}]},
        key=key,
        headers={CALL_ID_HEADER: call_id},
    )
    assert response.status_code == 200, response.text


def _spend_rows(call_ids: Sequence[str]) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT request_id, litellm_call_id, metadata FROM "LiteLLM_SpendLogs" WHERE litellm_call_id = ANY(%s)',
        (list(call_ids),),
    )


def _meta(row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return object_value(row["metadata"])


def _eval_entries(meta: Mapping[str, JsonValue]) -> list[dict[str, JsonValue]]:
    raw: Final = meta.get("eval_information")
    if raw is None:
        return []
    items: Final = raw if isinstance(raw, list) else [raw]
    return [object_value(item) for item in items]


def _guardrail_entries(meta: Mapping[str, JsonValue], name: str) -> list[dict[str, JsonValue]]:
    raw: Final = meta.get("guardrail_information")
    if raw is None:
        return []
    items: Final = raw if isinstance(raw, list) else [raw]
    return [object_value(item) for item in items if object_value(item).get("guardrail_name") in (None, name)]


def _statuses(row: Mapping[str, JsonValue], name: str) -> list[JsonValue]:
    return [entry.get("guardrail_status") for entry in _guardrail_entries(_meta(row), name)]


def _judge_prompt_chars(request: Request, letters: str) -> Mapping[str, int]:
    body: Final = json.loads(request.body)
    user_prompt: Final = str(body["messages"][-1].get("content", ""))
    return {letter: user_prompt.count(letter) for letter in letters}


def _scaffold_chars(labels: str, letters: str) -> int:
    return sum(labels.count(letter) for letter in letters)


def test_sample_rate_bounds_judge_calls_and_marks_not_run(gateway: Gateway, tmp_path: Path) -> None:
    total: Final = 200
    sample_rate: Final = 0.25
    with wire_server(_judge_scoring(95)) as judge, wire_server(_work_handler) as work:
        config, gname = _write_config(
            tmp_path,
            work.url,
            judge.url,
            {**BASE_JUDGE_PARAMS, "mode": "pre_call", "sample_rate": sample_rate},
        )
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            call_ids: Final = tuple(f"lit8332-sample-{RUN_TOKEN}-{i}" for i in range(total))
            for call_id in call_ids:
                _send(candidate, call_id)
            rows: Final = eventually(lambda: _spend_rows(call_ids), lambda values: len(values) == total, seconds=180)
            judged_rows: Final = [row for row in rows if _eval_entries(_meta(row))]
            skipped_rows: Final = [row for row in rows if not _eval_entries(_meta(row))]
            judged_calls: Final = len(judge.drain())
            assert judged_calls == len(judged_rows), (
                f"{judged_calls} judge upstream calls but {len(judged_rows)} spend rows carry eval_information"
            )
            sigma: Final = math.sqrt(total * sample_rate * (1 - sample_rate))
            assert total * sample_rate - 4 * sigma <= judged_calls <= total * sample_rate + 4 * sigma, (
                f"{judged_calls} judged of {total} at sample_rate {sample_rate} "
                f"(expected within [{total * sample_rate - 4 * sigma:.1f}, {total * sample_rate + 4 * sigma:.1f}])"
            )
            assert skipped_rows, f"all {total} requests were judged at sample_rate {sample_rate}"
            for row in skipped_rows:
                assert _statuses(row, gname) == ["not_run"], (
                    f"not-sampled {row['litellm_call_id']} statuses {_statuses(row, gname)}, expected ['not_run']"
                )
            assert len(work.drain()) == total


def test_sample_decision_depends_only_on_call_id(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_judge_scoring(95)) as judge, wire_server(_work_handler) as work:
        config, gname = _write_config(
            tmp_path,
            work.url,
            judge.url,
            {**BASE_JUDGE_PARAMS, "mode": "pre_call", "sample_rate": 0.5},
        )
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
            probe_ids: Final = tuple(f"lit8332-probe-{RUN_TOKEN}-{i}" for i in range(40))
            for call_id in probe_ids:
                _send(candidate, call_id)
            probe_rows: Final = eventually(
                lambda: _spend_rows(probe_ids), lambda values: len(values) == len(probe_ids), seconds=180
            )
            by_id: Final = {str(row["litellm_call_id"]): bool(_eval_entries(_meta(row))) for row in probe_rows}
            sampled: Final = [call_id for call_id, judged in by_id.items() if judged]
            unsampled: Final = [call_id for call_id, judged in by_id.items() if not judged]
            assert sampled and unsampled, f"probe produced only sampled={len(sampled)} unsampled={len(unsampled)}"
            user_a: Final = scenario.user()
            user_b: Final = scenario.user()
            key_a: Final = scenario.key(user_id=user_a, models=[WORK_MODEL])
            key_b: Final = scenario.key(user_id=user_b, models=[WORK_MODEL])
            for call_id in (sampled[0], unsampled[0]):
                judge.drain()
                for key in (key_a, key_b):
                    _send(candidate, call_id, key=key)
                    _send(candidate, call_id, key=key)
                rows: Final = eventually(
                    lambda call_id=call_id: read_rows(
                        'SELECT metadata FROM "LiteLLM_SpendLogs" WHERE litellm_call_id = %s',
                        (call_id,),
                    ),
                    lambda values: len(values) == 5,
                    seconds=120,
                )
                outcomes: Final = frozenset(bool(_eval_entries(_meta(row))) for row in rows)
                assert outcomes == frozenset({by_id[call_id]}), (
                    f"call id {call_id} decided {outcomes} across two keys and repeats, probe said {by_id[call_id]}"
                )
                judge_calls: Final = len(judge.drain())
                expected_calls: Final = 4 if by_id[call_id] else 0
                assert judge_calls == expected_calls, (
                    f"call id {call_id}: {judge_calls} judge calls, expected {expected_calls}"
                )


def test_default_sample_rate_judges_every_request(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_judge_scoring(95)) as judge, wire_server(_work_handler) as work:
        config, gname = _write_config(tmp_path, work.url, judge.url, {**BASE_JUDGE_PARAMS, "mode": "pre_call"})
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            call_ids: Final = tuple(f"lit8332-default-{RUN_TOKEN}-{i}" for i in range(10))
            for call_id in call_ids:
                _send(candidate, call_id)
            judge_calls: Final = len(judge.drain())
            assert judge_calls == len(call_ids), (
                f"{judge_calls} judge calls for {len(call_ids)} requests with no sample_rate"
            )
            rows: Final = eventually(
                lambda: _spend_rows(call_ids), lambda values: len(values) == len(call_ids), seconds=120
            )
            for row in rows:
                assert _statuses(row, gname) == ["success"], (
                    f"{row['litellm_call_id']} statuses {_statuses(row, gname)}"
                )


def test_sample_rate_zero_judges_nothing(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_judge_scoring(95)) as judge, wire_server(_work_handler) as work:
        config, gname = _write_config(
            tmp_path,
            work.url,
            judge.url,
            {**BASE_JUDGE_PARAMS, "mode": "pre_call", "sample_rate": 0},
        )
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            call_ids: Final = tuple(f"lit8332-zero-{RUN_TOKEN}-{i}" for i in range(10))
            for call_id in call_ids:
                _send(candidate, call_id)
            rows: Final = eventually(
                lambda: _spend_rows(call_ids), lambda values: len(values) == len(call_ids), seconds=120
            )
            assert len(judge.drain()) == 0, "judge called at sample_rate 0"
            for row in rows:
                meta: Final = _meta(row)
                assert not _eval_entries(meta), f"sample_rate 0 row carried eval_information: {meta}"
                assert _statuses(row, gname) == ["not_run"], _statuses(row, gname)


def test_max_input_chars_caps_judged_request_context(gateway: Gateway, tmp_path: Path) -> None:
    cap: Final = 500
    sent: Final = [
        {"role": "system", "content": "a" * 2000},
        {"role": "user", "content": "b" * 2000},
        {"role": "assistant", "content": "c" * 2000},
        {"role": "user", "content": "d" * 3000},
    ]
    with wire_server(_judge_scoring(95)) as judge, wire_server(_work_handler) as work:
        config, gname = _write_config(
            tmp_path,
            work.url,
            judge.url,
            {**BASE_JUDGE_PARAMS, "mode": "pre_call", "sample_rate": 1, "max_input_chars": cap},
        )
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": WORK_MODEL, "messages": sent},
                headers={CALL_ID_HEADER: "lit8332-truncate-pre-" + RUN_TOKEN},
            )
            assert response.status_code == 200, response.text
            judge_requests: Final = eventually(lambda: judge.drain(), lambda v: len(v) == 1, seconds=60)
            counts: Final = _judge_prompt_chars(judge_requests[0], "abcd")
            scaffold: Final = _scaffold_chars(
                "Criteria to evaluate:\n- work (weight 100%): Is this request work related?\n\n"
                "Conversation:\nLatest request turn to evaluate:\n",
                "abcd",
            )
            assert sum(counts.values()) - scaffold <= cap, (
                f"judge saw {counts} of capped letters ({scaffold} in fixed labels), cap {cap}"
            )
            assert counts["d"] > 0, "latest user turn missing from judge prompt"
            work_requests: Final = eventually(lambda: work.drain(), lambda v: len(v) == 1, seconds=60)
            assert json.loads(work_requests[0].body)["messages"] == sent


def test_max_input_chars_caps_judged_response_context(gateway: Gateway, tmp_path: Path) -> None:
    cap: Final = 500
    sent: Final = [
        {"role": "system", "content": "a" * 2000},
        {"role": "user", "content": "b" * 2000},
        {"role": "assistant", "content": "c" * 2000},
        {"role": "user", "content": "d" * 3000},
    ]

    def reply_e(request: Request) -> Reply:
        return _chat_completion("e" * 3000, model=WORK_MODEL)

    with wire_server(_judge_scoring(95)) as judge, wire_server(reply_e) as work:
        config, gname = _write_config(
            tmp_path,
            work.url,
            judge.url,
            {**BASE_JUDGE_PARAMS, "mode": "post_call", "sample_rate": 1, "max_input_chars": cap},
        )
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": WORK_MODEL, "messages": sent},
                headers={CALL_ID_HEADER: "lit8332-truncate-post-" + RUN_TOKEN},
            )
            assert response.status_code == 200, response.text
            judge_requests: Final = eventually(lambda: judge.drain(), lambda v: len(v) == 1, seconds=120)
            counts: Final = _judge_prompt_chars(judge_requests[0], "abcde")
            scaffold: Final = _scaffold_chars(
                "Criteria to evaluate:\n- work (weight 100%): Is this request work related?\n\n"
                "Conversation:\nAssistant response to evaluate:\n",
                "abcde",
            )
            assert sum(counts.values()) - scaffold <= cap, (
                f"judge saw {counts} of capped letters ({scaffold} in fixed labels), cap {cap}"
            )
            assert counts["e"] > 0, "response text missing from judge prompt"


def test_two_judges_on_one_request_both_record_eval_information(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_judge_scoring(95)) as judge, wire_server(_work_handler) as work:
        config, gname = _write_config(
            tmp_path,
            work.url,
            judge.url,
            {**BASE_JUDGE_PARAMS, "mode": "pre_call", "sample_rate": 1.0},
        )
        loaded: Final = yaml.safe_load(config.read_text())
        loaded["guardrails"].append(
            {
                "guardrail_name": f"{gname}-post",
                "litellm_params": {**BASE_JUDGE_PARAMS, "mode": "post_call", "sample_rate": 1.0},
            }
        )
        config.write_text(yaml.safe_dump(loaded))
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            call_id: Final = f"lit8332-dual-{RUN_TOKEN}"
            _send(candidate, call_id)
            rows: Final = eventually(
                lambda: _spend_rows([call_id]),
                lambda values: len(values) == 1,
                seconds=120,
            )
            entries: Final = _eval_entries(_meta(rows[0]))
            assert len(entries) == 2, (
                f"spend row eval_information has {len(entries)} entries, expected 2 "
                f"(one per pre_call + post_call judge): {entries}"
            )
            assert len(judge.drain()) == 2


def test_log_only_judge_failure_never_blocks(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_judge_scoring(0)) as judge, wire_server(_work_handler) as work:
        config, gname = _write_config(
            tmp_path,
            work.url,
            judge.url,
            {**BASE_JUDGE_PARAMS, "mode": "pre_call", "on_failure": "log", "overall_threshold": 80},
        )
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": WORK_MODEL, "messages": [{"role": "user", "content": "synthetic off topic"}]},
                headers={CALL_ID_HEADER: "lit8332-logonly-" + RUN_TOKEN},
            )
            assert response.status_code == 200, response.text
            assert len(judge.drain()) == 1
            rows: Final = eventually(
                lambda: _spend_rows(("lit8332-logonly-" + RUN_TOKEN,)), lambda values: len(values) == 1, seconds=120
            )
            assert _statuses(rows[0], gname) == ["guardrail_intervened"], _guardrail_entries(_meta(rows[0]), gname)


def test_judge_upstream_error_fails_open(gateway: Gateway, tmp_path: Path) -> None:
    def broken(request: Request) -> Reply:
        return Reply(status=500, body=b'{"error": {"message": "judge down"}}')

    with wire_server(broken) as judge, wire_server(_work_handler) as work:
        config, gname = _write_config(
            tmp_path, work.url, judge.url, {**BASE_JUDGE_PARAMS, "mode": "pre_call", "on_failure": "block"}
        )
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": WORK_MODEL, "messages": [{"role": "user", "content": "synthetic request"}]},
                headers={CALL_ID_HEADER: "lit8332-chaos-500-" + RUN_TOKEN},
            )
            assert response.status_code == 200, response.text
            rows: Final = eventually(
                lambda: _spend_rows(("lit8332-chaos-500-" + RUN_TOKEN,)), lambda values: len(values) == 1, seconds=120
            )
            meta: Final = _meta(rows[0])
            assert _statuses(rows[0], gname) == ["guardrail_failed_to_respond"], _guardrail_entries(meta, gname)
            assert not any("passed" in entry for entry in _eval_entries(meta)), meta


def test_judge_non_json_verdict_fails_open(gateway: Gateway, tmp_path: Path) -> None:
    def garbled(request: Request) -> Reply:
        return _chat_completion("not a json verdict at all", model=JUDGE_MODEL)

    with wire_server(garbled) as judge, wire_server(_work_handler) as work:
        config, gname = _write_config(
            tmp_path, work.url, judge.url, {**BASE_JUDGE_PARAMS, "mode": "pre_call", "on_failure": "block"}
        )
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": WORK_MODEL, "messages": [{"role": "user", "content": "synthetic request"}]},
                headers={CALL_ID_HEADER: "lit8332-chaos-json-" + RUN_TOKEN},
            )
            assert response.status_code == 200, response.text
            rows: Final = eventually(
                lambda: _spend_rows(("lit8332-chaos-json-" + RUN_TOKEN,)), lambda values: len(values) == 1, seconds=120
            )
            meta: Final = _meta(rows[0])
            assert _statuses(rows[0], gname) == ["guardrail_failed_to_respond"], _guardrail_entries(meta, gname)
            assert not any("passed" in entry for entry in _eval_entries(meta)), meta


def _guardrail_table_rows(name: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT guardrail_name FROM "LiteLLM_GuardrailsTable" WHERE guardrail_name = %s', (name,))


def _drop_guardrail_row(name: str) -> None:
    write_rows('DELETE FROM "LiteLLM_GuardrailsTable" WHERE guardrail_name = %s', (name,))


def _create_guardrail(candidate: Gateway, name: str, params: Mapping[str, JsonValue]) -> httpx.Response:
    return candidate.request(
        "POST",
        "/guardrails",
        {"guardrail": {"guardrail_name": name, "litellm_params": dict(params)}},
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("sample_rate", -0.25),
        ("sample_rate", 1.5),
        ("sample_rate", "quarter"),
        ("sample_rate", True),
        ("max_input_chars", 0),
        ("max_input_chars", -10),
        ("max_input_chars", "many"),
        ("max_input_chars", True),
        ("max_input_chars", 2.5),
    ],
)
def test_invalid_sampling_params_rejected_on_create(
    gateway: Gateway, tmp_path: Path, field: str, value: JsonValue
) -> None:
    with wire_server(_judge_scoring(95)) as judge, wire_server(_work_handler) as work:
        config, gname = _write_config(tmp_path, work.url, judge.url, {**BASE_JUDGE_PARAMS, "mode": "pre_call"})
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            name: Final = f"lit8332-invalid-{field}-{uuid.uuid4().hex[:12]}"
            try:
                response: Final = _create_guardrail(
                    candidate,
                    name,
                    {
                        **BASE_JUDGE_PARAMS,
                        "default_on": False,
                        "mode": "pre_call",
                        "judge_model": "lit8332-zombie-judge",
                        field: value,
                    },
                )
                assert 400 <= response.status_code < 500, (
                    f"POST /guardrails {field}={value!r} returned {response.status_code}: {response.text}"
                )
                assert _guardrail_table_rows(name) == []
            finally:
                _drop_guardrail_row(name)


def test_valid_sampling_params_accepted_on_create(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_judge_scoring(95)) as judge, wire_server(_work_handler) as work:
        config, gname = _write_config(tmp_path, work.url, judge.url, {**BASE_JUDGE_PARAMS, "mode": "pre_call"})
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            name: Final = f"lit8332-valid-{uuid.uuid4().hex[:12]}"
            try:
                response: Final = _create_guardrail(
                    candidate,
                    name,
                    {
                        **BASE_JUDGE_PARAMS,
                        "default_on": False,
                        "mode": "pre_call",
                        "judge_model": "lit8332-zombie-judge",
                        "sample_rate": 0.05,
                        "max_input_chars": 5000,
                    },
                )
                assert response.status_code == 200, (
                    f"POST /guardrails valid params returned {response.status_code}: {response.text}"
                )
                assert _guardrail_table_rows(name) != []
            finally:
                _drop_guardrail_row(name)


def test_not_run_requests_excluded_from_guardrail_usage(gateway: Gateway, tmp_path: Path) -> None:
    total: Final = 40
    guardrail_name: Final = f"lit8332-usage-{uuid.uuid4().hex[:12]}"
    with wire_server(_judge_scoring(95)) as judge, wire_server(_work_handler) as work:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["model_list"] = [
            {
                "model_name": WORK_MODEL,
                "litellm_params": {
                    "model": f"openai/{WORK_MODEL}",
                    "api_base": f"{work.url}/v1",
                    "api_key": "synthetic-work-key",
                },
            },
            {
                "model_name": JUDGE_MODEL,
                "litellm_params": {
                    "model": f"openai/{JUDGE_MODEL}",
                    "api_base": f"{judge.url}/v1",
                    "api_key": "synthetic-judge-key",
                },
            },
        ]
        config.pop("vector_store_registry", None)
        config["litellm_settings"]["cache"] = False
        config["litellm_settings"].pop("cache_params", None)
        config["guardrails"] = [
            {
                "guardrail_name": guardrail_name,
                "litellm_params": {**BASE_JUDGE_PARAMS, "mode": "pre_call", "sample_rate": 0.5},
            }
        ]
        path: Final = tmp_path / "usage.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate:
            call_ids: Final = tuple(f"lit8332-usage-{RUN_TOKEN}-{i}" for i in range(total))
            for call_id in call_ids:
                _send(candidate, call_id)
            rows: Final = eventually(lambda: _spend_rows(call_ids), lambda values: len(values) == total, seconds=180)
            not_run: Final = [row for row in rows if _statuses(row, guardrail_name) == ["not_run"]]
            assert not_run, f"no not_run guardrail rows among {total} requests at sample_rate 0.5"
            judged_rows: Final = [row for row in rows if _eval_entries(_meta(row))]

            def overview_row() -> dict[str, JsonValue]:
                overview: Final = candidate.get("/guardrails/usage/overview")
                matches: Final = [
                    object_value(row)
                    for row in list_value(overview["rows"])
                    if object_value(row)["name"] == guardrail_name
                ]
                return matches[0] if matches else {}

            row: Final = eventually(
                overview_row,
                lambda value: int(value.get("requestsEvaluated", 0) or 0) == len(judged_rows),
                seconds=120,
            )
            assert int(row["requestsEvaluated"]) == len(judged_rows) < total, row


def test_logging_only_judge_records_eval_information(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_judge_scoring(95)) as judge, wire_server(_work_handler) as work:
        config, gname = _write_config(
            tmp_path, work.url, judge.url, {**BASE_JUDGE_PARAMS, "mode": "logging_only", "on_failure": "log"}
        )
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            call_id: Final = "lit8332-logscan-" + RUN_TOKEN
            _send(candidate, call_id)
            rows: Final = eventually(lambda: _spend_rows((call_id,)), lambda values: len(values) == 1, seconds=120)
            meta: Final = _meta(rows[0])
            entries: Final = _eval_entries(meta)
            assert entries, f"logging_only judge left no eval_information in spend row: {meta}"
            assert all(entry.get("overall_score") == 95 for entry in entries), entries
            assert all(entry.get("eval_name") == gname for entry in entries), entries


def test_eval_information_reasoning_redacted_when_store_prompts_disabled(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_judge_scoring(95)) as judge, wire_server(_work_handler) as work:
        config, gname = _write_config(
            tmp_path, work.url, judge.url, {**BASE_JUDGE_PARAMS, "mode": "pre_call", "on_failure": "log"}
        )
        loaded: Final = yaml.safe_load(config.read_text())
        loaded["general_settings"]["store_prompts_in_spend_logs"] = False
        config.write_text(yaml.safe_dump(loaded))
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            call_id: Final = "lit8332-redact-" + RUN_TOKEN
            _send(candidate, call_id)
            rows: Final = eventually(lambda: _spend_rows((call_id,)), lambda values: len(values) == 1, seconds=120)
            meta: Final = _meta(rows[0])
            entries: Final = _eval_entries(meta)
            assert entries, f"judged spend row has no eval_information: {meta}"
            for entry in entries:
                assert entry.get("passed") is True
                assert entry.get("overall_score") == 95
                for verdict in list_value(entry.get("verdicts")):
                    assert object_value(verdict)["reasoning"] == "REDACTED_BY_LITELM"
                    assert object_value(verdict)["score"] == 95


_BLOCKING_JUDGE_PARAMS: Final = [
    ({"mode": "pre_call", "on_failure": "block", "sample_rate": 0.5}, "sample_rate"),
    ({"mode": "pre_call", "sample_rate": 0.5}, "sample_rate"),
    ({"mode": "pre_call", "on_failure": "block", "max_input_chars": 5000}, "max_input_chars"),
    ({"mode": "during_call", "on_failure": "block", "max_input_chars": 5000}, "max_input_chars"),
]


@pytest.mark.parametrize("blocking_params,field", _BLOCKING_JUDGE_PARAMS)
def test_sampling_params_rejected_on_blocking_judge(
    gateway: Gateway, tmp_path: Path, blocking_params: Mapping[str, JsonValue], field: str
) -> None:
    with wire_server(_judge_scoring(95)) as judge, wire_server(_work_handler) as work:
        config, gname = _write_config(tmp_path, work.url, judge.url, {**BASE_JUDGE_PARAMS, "mode": "pre_call"})
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            name: Final = f"lit8332-blocking-{field}-{uuid.uuid4().hex[:12]}"
            try:
                litellm_params: Final = {
                    **BASE_JUDGE_PARAMS,
                    "default_on": False,
                    "judge_model": "lit8332-zombie-judge",
                    **blocking_params,
                }
                if "on_failure" not in blocking_params:
                    litellm_params.pop("on_failure", None)
                response: Final = _create_guardrail(candidate, name, litellm_params)
                assert 400 <= response.status_code < 500, (
                    f"POST /guardrails {blocking_params} returned {response.status_code}: {response.text}"
                )
                assert field in response.text, f"rejection did not name {field}: {response.text}"
                assert _guardrail_table_rows(name) == []
            finally:
                _drop_guardrail_row(name)


@pytest.mark.parametrize(
    "accepted_params",
    [
        {"mode": "pre_call", "on_failure": "block", "sample_rate": 1},
        {"mode": "logging_only", "on_failure": "block", "sample_rate": 0.5},
    ],
)
def test_sampling_params_accepted_when_judge_cannot_block(
    gateway: Gateway, tmp_path: Path, accepted_params: Mapping[str, JsonValue]
) -> None:
    with wire_server(_judge_scoring(95)) as judge, wire_server(_work_handler) as work:
        config, gname = _write_config(tmp_path, work.url, judge.url, {**BASE_JUDGE_PARAMS, "mode": "pre_call"})
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            name: Final = f"lit8332-nonblocking-{uuid.uuid4().hex[:12]}"
            try:
                response: Final = _create_guardrail(
                    candidate,
                    name,
                    {
                        **BASE_JUDGE_PARAMS,
                        "default_on": False,
                        "judge_model": "lit8332-zombie-judge",
                        **accepted_params,
                    },
                )
                assert response.status_code == 200, (
                    f"POST /guardrails {accepted_params} returned {response.status_code}: {response.text}"
                )
                assert _guardrail_table_rows(name) != []
            finally:
                _drop_guardrail_row(name)


def test_blocking_update_to_sampled_judge_rejected(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_judge_scoring(95)) as judge, wire_server(_work_handler) as work:
        config, gname = _write_config(tmp_path, work.url, judge.url, {**BASE_JUDGE_PARAMS, "mode": "pre_call"})
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            name: Final = f"lit8332-update-{uuid.uuid4().hex[:12]}"
            try:
                create: Final = _create_guardrail(
                    candidate,
                    name,
                    {
                        **BASE_JUDGE_PARAMS,
                        "default_on": False,
                        "mode": "pre_call",
                        "on_failure": "log",
                        "sample_rate": 0.5,
                        "judge_model": "lit8332-zombie-judge",
                    },
                )
                assert create.status_code == 200, create.text
                guardrail_id: Final = str(object_value(create.json())["guardrail_id"])

                put_response: Final = candidate.request(
                    "PUT",
                    f"/guardrails/{guardrail_id}",
                    {
                        "guardrail": {
                            "guardrail_name": name,
                            "litellm_params": {
                                **BASE_JUDGE_PARAMS,
                                "default_on": False,
                                "mode": "pre_call",
                                "on_failure": "block",
                                "sample_rate": 0.5,
                                "judge_model": "lit8332-zombie-judge",
                            },
                        }
                    },
                )
                assert 400 <= put_response.status_code < 500, (
                    f"PUT on_failure=block onto sampled judge returned {put_response.status_code}: {put_response.text}"
                )

                patch_response: Final = candidate.request(
                    "PATCH",
                    f"/guardrails/{guardrail_id}",
                    {"litellm_params": {"on_failure": "block"}},
                )
                assert 400 <= patch_response.status_code < 500, (
                    f"PATCH on_failure=block onto sampled judge returned {patch_response.status_code}: {patch_response.text}"
                )

                stored: Final = read_rows(
                    'SELECT litellm_params FROM "LiteLLM_GuardrailsTable" WHERE guardrail_name = %s', (name,)
                )
                assert stored, f"guardrail row for {name} missing after rejected update"
                assert object_value(stored[0]["litellm_params"]).get("on_failure") == "log", (
                    f"rejected update still changed the stored row: {stored[0]}"
                )
            finally:
                _drop_guardrail_row(name)


def test_config_does_not_load_sampled_blocking_judge(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_judge_scoring(95)) as judge, wire_server(_work_handler) as work:
        config, gname = _write_config(
            tmp_path,
            work.url,
            judge.url,
            {**BASE_JUDGE_PARAMS, "mode": "pre_call", "on_failure": "block", "sample_rate": 0.5},
        )
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            call_ids: Final = tuple(f"lit8332-cfgblock-{RUN_TOKEN}-{i}" for i in range(5))
            for call_id in call_ids:
                _send(candidate, call_id)
            rows: Final = eventually(
                lambda: _spend_rows(call_ids), lambda values: len(values) == len(call_ids), seconds=120
            )
            assert len(judge.drain()) == 0, "a blocking judge was sampled despite on_failure=block + sample_rate"
            for row in rows:
                assert _statuses(row, gname) == [], (
                    f"sampled blocking judge left guardrail_information: {_guardrail_entries(_meta(row), gname)}"
                )


@pytest.mark.parametrize(
    "extra",
    [
        {"max_input_chars": True},
        {"max_input_chars": 5},
        {"sample_rate": 1.5},
        {"sample_rate": "0.5"},
    ],
)
def test_non_judge_guardrail_validation_unchanged(
    gateway: Gateway, tmp_path: Path, extra: Mapping[str, JsonValue]
) -> None:
    with wire_server(_judge_scoring(95)) as judge, wire_server(_work_handler) as work:
        config, gname = _write_config(tmp_path, work.url, judge.url, {**BASE_JUDGE_PARAMS, "mode": "pre_call"})
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            name: Final = f"lit8332-decision-{uuid.uuid4().hex[:12]}"
            try:
                response: Final = _create_guardrail(
                    candidate,
                    name,
                    {
                        "guardrail": "decision_model",
                        "mode": "pre_call",
                        "decision_model": "lit8332-live",
                        "checks": [{"name": "c", "instructions": "work?"}],
                        "default_on": False,
                        **extra,
                    },
                )
                assert response.status_code == 200, (
                    f"POST /guardrails decision_model {extra} returned {response.status_code}: {response.text}"
                )
            finally:
                _drop_guardrail_row(name)
