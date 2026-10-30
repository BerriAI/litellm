from __future__ import annotations

import math

import pytest

from litellm.proxy.mcp_server.code_worker import Reply, encode_message, load_json


def test_code_mode_rejects_archived_quickjs_package(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.mcp_server import code_runtime

    monkeypatch.setattr(code_runtime.metadata, "distribution", lambda name: None)
    with pytest.raises(ValueError, match="conflicting quickjs package"):
        code_runtime.check_runtime()


def test_code_mode_reports_missing_runtime_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.mcp_server import code_runtime

    def missing(name: str) -> None:
        raise code_runtime.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(code_runtime.metadata, "distribution", missing)
    with pytest.raises(ValueError, match=r"requires quickjs-ng\. Install 'litellm\[proxy,mcp-code-mode\]'"):
        code_runtime.check_runtime()


@pytest.mark.parametrize("value", [2**53, -(2**53), float(2**53), math.inf, math.nan])
def test_tool_result_numbers_are_never_silently_changed(value: float) -> None:
    with pytest.raises(ValueError, match="JSON"):
        encode_message(Reply(id=0, value={"id": value}), 4096)


def test_large_ids_as_strings_and_exact_safe_integers_survive() -> None:
    value = {"large": str(2**63), "safe": 2**53 - 1}
    result = Reply.model_validate(load_json(encode_message(Reply(id=7, value=value), 4096).decode()))
    assert result.value == value
    assert result.id == 7


def test_pipe_limit_counts_unicode_bytes() -> None:
    with pytest.raises(ValueError, match="byte limit"):
        encode_message(Reply(id=0, value={"value": "猫" * 100}), 200)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["\ud800", " " * (128 * 1024 + 1)])
async def test_invalid_script_fails_before_starting_a_worker(code: str) -> None:
    from pydantic import JsonValue

    from litellm.proxy.mcp_server.code_runtime import execute_code

    async def unused(tool_id: str, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        pytest.fail("Invalid code must never execute tools")

    result = await execute_code(code, unused)
    assert result.error is not None
    assert result.calls_started == 0
