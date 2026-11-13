from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

import pytest
from cases.small_bodies_5rps import CASE as BASELINE
from cases.small_bodies_customer_5rps import CASE as CUSTOMER
from setup.plan import build_plan
from setup.provision import CaseSetup
from setup.run import run_until, wire_request
from setup.wire import CustomerHeaders, TokenSource, WireRecord, WireRequest


def test_customer_case_preserves_baseline_ceilings_and_transmits_all_headers_on_every_endpoint() -> None:
    customer: Final = CustomerHeaders(organization="org", team="team", member="member")
    setup: Final = CaseSetup(
        marker="run",
        team_ids={},
        key_bearers={key.name: "key" for key in CUSTOMER.virtual_keys},
        token_sources={
            user.name: TokenSource(
                token_url="http://idp.invalid/token", client_id="perf", username=user.name, password="unused"
            )
            for user in CUSTOMER.users
        },
        user_ids={},
        customer_headers=customer,
    )
    assert CUSTOMER.limits == BASELINE.limits
    assert CUSTOMER.traffic == BASELINE.traffic
    assert CUSTOMER.customer_hierarchy
    plan: Final = build_plan(CUSTOMER)
    requests: Final = tuple(wire_request(item, setup) for item in plan)
    assert {item.endpoint for item in requests} == {"chat_completions", "responses", "messages"}
    assert all(
        WireRequest.model_validate_json(item.model_dump_json()).customer_headers == customer for item in requests
    )
    assert dict(customer.headers()) == {
        "x-litellm-customer-org-alias": "org",
        "x-litellm-customer-team-id": "team",
        "x-litellm-customer-member-id": "member",
    }


def record(seq: int = 0, *, started_at: float = 100) -> WireRecord:
    return WireRecord(
        seq=seq,
        marker=f"request-{seq}",
        caller="key",
        endpoint="chat_completions",
        stream=False,
        blocked=False,
        started_at=started_at,
        status=200,
        header_s=1,
        first_byte_s=1,
        first_token_s=None,
        total_s=1,
        text="answer",
        ok=True,
        mock_delay_s=1,
        call_id=f"call-{seq}",
        error=None,
        response_id=f"response-{seq}",
    )


class StopAfterRecord:
    def __init__(self, *, partial: Path, resume: Path, fail: bool = False) -> None:
        self.partial: Final = partial
        self.resume: Final = resume
        self.fail: Final = fail
        self.received: tuple[WireRecord, ...] = ()

    def __call__(self, records: tuple[WireRecord, ...]) -> bool:
        self.received += records
        if self.partial.exists():
            self.resume.touch()
        assert not (self.fail and records), "Database failed"
        return bool(records)


@pytest.mark.parametrize("fail", [False, True])
def test_controlled_process_preserves_partial_lines_and_drains_on_stop_or_failure(tmp_path: Path, fail: bool) -> None:
    records: Final = tmp_path / "records.jsonl"
    stop: Final = tmp_path / "stop"
    partial: Final = tmp_path / "partial"
    resume: Final = tmp_path / "resume"
    records.touch()
    encoded: Final = record().model_dump_json() + "\n"
    program: Final = f"""
import time
from pathlib import Path
with Path({str(records)!r}).open("w") as f:
    text = {encoded!r}
    f.write(text[:10])
    f.flush()
    Path({str(partial)!r}).touch()
    while not Path({str(resume)!r}).exists():
        time.sleep(0.01)
    f.write(text[10:])
    f.flush()
    while not Path({str(stop)!r}).exists():
        time.sleep(0.01)
    f.write({record(1).model_dump_json()!r} + "\\n")
    f.flush()
"""
    callback: Final = StopAfterRecord(partial=partial, resume=resume, fail=fail)
    command: Final = [sys.executable, "-c", program]
    if fail:
        with pytest.raises(AssertionError, match="Database failed"):
            _ = run_until(command, dict(os.environ), records, stop, callback, timeout=10)
    else:
        result: Final = run_until(command, dict(os.environ), records, stop, callback, timeout=10)
        assert result.returncode == 0, result.stdout
    assert callback.received == (record(),)
    assert tuple(WireRecord.model_validate_json(line) for line in records.read_text().splitlines()) == (
        record(),
        record(1),
    )
