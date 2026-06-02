"""
Unit tests for ``LocalSandbox``.

Covers the tool surface (Bash / Read / Write / Edit / ls) plus the
sandbox lifecycle (setup creates a tmpdir, teardown removes it).
"""

import os
import tempfile
from pathlib import Path

import pytest

from litellm.managed_agents.sandbox.base import ToolResult
from litellm.managed_agents.sandbox.local import LocalSandbox


@pytest.mark.asyncio
async def test_setup_creates_tmpdir_and_teardown_removes_it():
    sb = LocalSandbox()
    await sb.setup()
    assert sb.cwd is not None and os.path.isdir(sb.cwd)
    cwd = sb.cwd
    await sb.teardown()
    assert not os.path.exists(cwd)
    assert sb.cwd is None


@pytest.mark.asyncio
async def test_setup_respects_explicit_working_dir():
    with tempfile.TemporaryDirectory() as tmp:
        sb = LocalSandbox(working_dir=tmp)
        await sb.setup()
        assert sb.cwd == tmp
        await sb.teardown()
        # Explicit dir is left intact (we don't own it).
        assert os.path.isdir(tmp)


@pytest.mark.asyncio
async def test_bash_runs_command_and_returns_stdout():
    with tempfile.TemporaryDirectory() as tmp:
        sb = LocalSandbox(working_dir=tmp)
        result = await sb.execute_tool("Bash", {"command": "echo hello"})
        assert isinstance(result, ToolResult)
        assert not result.is_error
        assert result.output.strip() == "hello"


@pytest.mark.asyncio
async def test_bash_nonzero_exit_marks_is_error():
    with tempfile.TemporaryDirectory() as tmp:
        sb = LocalSandbox(working_dir=tmp)
        result = await sb.execute_tool("Bash", {"command": "exit 7"})
        assert result.is_error
        assert result.metadata.get("exit_code") == 7


@pytest.mark.asyncio
async def test_bash_empty_command_returns_error():
    sb = LocalSandbox()
    result = await sb.execute_tool("Bash", {})
    assert result.is_error
    assert "command" in result.output


@pytest.mark.asyncio
async def test_write_then_read_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        sb = LocalSandbox(working_dir=tmp)
        write = await sb.execute_tool("Write", {"path": "a.txt", "content": "hi"})
        assert not write.is_error
        assert (Path(tmp) / "a.txt").read_text() == "hi"

        read = await sb.execute_tool("Read", {"path": "a.txt"})
        assert not read.is_error
        assert read.output == "hi"


@pytest.mark.asyncio
async def test_edit_replaces_substring():
    with tempfile.TemporaryDirectory() as tmp:
        sb = LocalSandbox(working_dir=tmp)
        await sb.execute_tool("Write", {"path": "a.txt", "content": "hello world"})
        edit = await sb.execute_tool(
            "Edit",
            {"path": "a.txt", "old_string": "world", "new_string": "moon"},
        )
        assert not edit.is_error
        assert (Path(tmp) / "a.txt").read_text() == "hello moon"


@pytest.mark.asyncio
async def test_edit_missing_old_string_is_error():
    with tempfile.TemporaryDirectory() as tmp:
        sb = LocalSandbox(working_dir=tmp)
        await sb.execute_tool("Write", {"path": "a.txt", "content": "hello"})
        edit = await sb.execute_tool(
            "Edit",
            {"path": "a.txt", "old_string": "nope", "new_string": "_"},
        )
        assert edit.is_error
        assert "not found" in edit.output


@pytest.mark.asyncio
async def test_ls_lists_dir_entries():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "a.txt").write_text("")
        (Path(tmp) / "b.txt").write_text("")
        sb = LocalSandbox(working_dir=tmp)
        result = await sb.execute_tool("ls", {"path": tmp})
        assert not result.is_error
        names = sorted(result.output.split("\n"))
        assert names == ["a.txt", "b.txt"]


@pytest.mark.asyncio
async def test_unknown_tool_returns_error():
    sb = LocalSandbox()
    result = await sb.execute_tool("Magic", {})
    assert result.is_error
    assert "unknown tool" in result.output


@pytest.mark.asyncio
async def test_bash_timeout_kills_command():
    with tempfile.TemporaryDirectory() as tmp:
        sb = LocalSandbox(working_dir=tmp, shell_timeout_seconds=0.5)
        result = await sb.execute_tool("Bash", {"command": "sleep 5"})
        assert result.is_error
        assert "timed out" in result.output
