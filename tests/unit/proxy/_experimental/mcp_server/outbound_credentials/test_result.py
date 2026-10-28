"""Smoke test for the outbound_credentials Result union.

Result is trivial frozen dataclasses; its load-bearing guarantee (no `.ok` access before
the Error arm is eliminated) is a type-checker property, not a runtime one. This pins only
the runtime contract consumers rely on: each arm carries its payload and discriminates by
type. The union is exercised for real where it is used (see PR2's parse_auth_spec_kind).
"""

from litellm.proxy._experimental.mcp_server.outbound_credentials import (
    Error,
    Ok,
    Result,
)


def test_ok_and_error_carry_payload_and_discriminate():
    ok: Result[int, str] = Ok(5)
    err: Result[int, str] = Error("boom")
    assert isinstance(ok, Ok) and ok.ok == 5
    assert isinstance(err, Error) and err.error == "boom"
