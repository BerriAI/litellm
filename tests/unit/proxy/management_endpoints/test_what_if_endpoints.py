from collections.abc import Mapping
from typing import Final

import pytest
from fastapi import HTTPException

from litellm.proxy.management_endpoints.what_if_endpoints import (
    Exchange,
    Tape,
    _provider_request,
    answer_changes,
    compare_tape,
    field_fates,
    fingerprint,
    gateway_base,
    key_split,
    lost_fields,
    replayable_exchanges,
    summarize_answer,
)

MESSAGES: Final = [{"role": "user", "content": "hi"}]


def _provider_that_ignores_seed_and_renames_stop(body: Mapping[str, object]) -> Mapping[str, object]:
    renamed: Final = {"stop_sequences": body["stop"]} if "stop" in body else {}
    return {"model": body["model"], "messages": body["messages"], **renamed}


def _logged(
    request_id: str, key: str, system: str, tools: tuple[str, ...] = (), seed: int | None = None
) -> Mapping[str, object]:
    return {
        "request_id": request_id,
        "api_key": key,
        "key_alias": "shared-service-account",
        "spend": 0.5,
        "status": "success",
        "proxy_server_request": {
            "model": "public-name",
            "messages": [{"role": "system", "content": system}, *MESSAGES],
            **({"tools": [{"type": "function", "function": {"name": name}} for name in tools]} if tools else {}),
            "metadata": {"deployment": "provider/model", "headers": {"user-agent": "python-httpx/0.28"}},
            **({} if seed is None else {"seed": seed}),
        },
    }


def test_a_field_that_changes_nothing_in_the_provider_request_is_dropped_and_one_that_changes_it_is_rewritten():
    body: Final = {"model": "provider/model", "messages": MESSAGES, "seed": 7, "stop": ["END"]}

    fates: Final = {fate.field: fate for fate in field_fates(body, _provider_that_ignores_seed_and_renames_stop)}

    assert (fates["seed"].fate, fates["seed"].became) == ("dropped", ())
    assert (fates["stop"].fate, fates["stop"].became) == ("rewritten", ("stop_sequences",))


def test_a_field_the_provider_receives_under_its_own_name_is_not_reported():
    def passthrough(body: Mapping[str, object]) -> Mapping[str, object]:
        return body

    assert field_fates({"model": "m", "messages": MESSAGES, "temperature": 0.2}, passthrough) == ()


def test_the_real_anthropic_translation_drops_n_and_rewrites_stop():
    body: Final = {
        "model": "anthropic/claude-haiku-4-5",
        "messages": MESSAGES,
        "n": 3,
        "stop": ["END"],
        "drop_params": True,
    }

    fates: Final = {fate.field: fate.fate for fate in field_fates(body, _provider_request)}

    assert fates["n"] == "dropped"
    assert fates["stop"] == "rewritten"


def test_lost_fields_counts_every_request_that_lost_the_field_and_points_at_one_of_them():
    rows: Final = (
        _logged("req-1", "key", "one", seed=7),
        _logged("req-2", "key", "two", seed=7),
        _logged("req-3", "key", "three"),
    )

    report: Final = lost_fields(rows, 24, _provider_that_ignores_seed_and_renames_stop)

    assert (report.requests_scanned, report.requests_with_dropped_fields) == (3, 2)
    (seed,) = (field for field in report.fields if field.field == "seed")
    assert (seed.model, seed.fate, seed.requests) == ("provider/model", "dropped", 2)
    assert seed.example_request_id in {"req-1", "req-2"}


def test_requests_with_the_same_system_prompt_and_tools_are_one_agent_whatever_the_user_says():
    first: Final = {
        "messages": [{"role": "system", "content": "You are the refund agent."}, {"role": "user", "content": "a"}]
    }
    second: Final = {
        "messages": [{"role": "system", "content": "You are the refund agent."}, {"role": "user", "content": "b"}]
    }
    other_tools: Final = {**first, "tools": [{"type": "function", "function": {"name": "issue_refund"}}]}

    assert fingerprint(first) == fingerprint(second)
    assert fingerprint(first) != fingerprint(other_tools)


def test_key_split_reports_only_keys_carrying_more_than_one_agent_with_spend_per_agent():
    rows: Final = (
        _logged("req-1", "shared", "You are the invoice triage agent.", ("lookup_invoice",)),
        _logged("req-2", "shared", "You are the invoice triage agent.", ("lookup_invoice",)),
        _logged("req-3", "shared", "You are the refund agent.", ("issue_refund",)),
        _logged("req-4", "solo", "You are the refund agent.", ("issue_refund",)),
    )

    report: Final = key_split(rows, 24)

    assert report.keys_scanned == 2
    (shared,) = report.shared_keys
    assert (shared.key_hash, shared.requests, shared.spend) == ("shared", 3, 1.5)
    assert [(agent.suggested_name, agent.requests, agent.spend) for agent in shared.agents] == [
        ("invoice-triage-agent", 2, 1.0),
        ("refund-agent", 1, 0.5),
    ]
    assert shared.agents[0].tools == ("lookup_invoice",)


def _listing(*names: str, error: str | None = None) -> Mapping[str, object]:
    tools: Final = [
        {"name": name, "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}}} for name in names
    ]
    return summarize_answer(200, {"tools": tools, **({"error": error} if error else {})})


def test_a_tool_list_that_still_answers_200_but_lost_its_tools_is_a_change():
    changes: Final = answer_changes(_listing("search", "get"), _listing(error="partial_failure"))

    assert changes == ("error: None -> partial_failure", "tool missing: get", "tool missing: search")


def test_a_tool_whose_input_schema_changed_is_a_change_and_an_identical_list_is_not():
    renamed_argument: Final = summarize_answer(
        200, {"tools": [{"name": "search", "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}}}]}
    )

    assert answer_changes(_listing("search"), renamed_argument) == ("schema changed: search",)
    assert answer_changes(_listing("search"), _listing("search")) == ()


def test_replaying_a_tape_marks_only_the_exchanges_whose_answer_changed():
    def exchange(name: str, answer: Mapping[str, object]) -> Exchange:
        return Exchange(
            name=name, method="GET", path="/mcp-rest/tools/list", upstream_credential=False, status=200, answer=answer
        )

    tape: Final = Tape(
        recorded_from="http://current",
        version="1.0.0",
        exchanges=(exchange("no credential", _listing("search")), exchange("with credential", _listing("search"))),
    )
    replayed: Final = (exchange("no credential", _listing()), exchange("with credential", _listing("search")))

    check: Final = compare_tape(tape, "http://candidate", "2.0.0", replayed)

    assert check.changed == 1
    assert [(e.name, e.verdict) for e in check.exchanges] == [("no credential", "changed"), ("with credential", "same")]
    assert check.exchanges[0].changes == ("tool missing: search",)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://user:pw@gateway.internal", "gateway.internal:4000"])
def test_tape_check_refuses_a_gateway_url_that_is_not_plain_http(url: str) -> None:
    with pytest.raises(HTTPException) as refused:
        gateway_base(url)
    assert refused.value.status_code == 400


def test_tape_check_accepts_a_plain_gateway_url() -> None:
    assert gateway_base("http://localhost:4000/") == "http://localhost:4000"


def test_tape_check_refuses_to_replay_a_route_it_does_not_record() -> None:
    forged: Final = Tape(
        recorded_from="http://old",
        version="1",
        exchanges=(
            Exchange(name="x", method="GET", path="/key/list", upstream_credential=False, status=200, answer={}),
        ),
    )
    with pytest.raises(HTTPException) as refused:
        replayable_exchanges(forged)
    assert refused.value.status_code == 400
