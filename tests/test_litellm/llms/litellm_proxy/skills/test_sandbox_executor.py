import json
from unittest.mock import patch

from litellm.llms.litellm_proxy.skills.sandbox_executor import SkillsSandboxExecutor


def test_execute_shell_command_serializes_basic_args():
    executor = SkillsSandboxExecutor()
    mock_result = {"success": True, "output": "", "error": "", "files": []}

    with patch.object(executor, "execute", return_value=mock_result) as mock_execute:
        result = executor.execute_shell_command(["ls", "-la"])

    assert result == mock_result
    mock_execute.assert_called_once()
    code = mock_execute.call_args.kwargs["code"]
    assert "subprocess.run([\"ls\", \"-la\"], capture_output=True, text=True)" in code
    assert mock_execute.call_args.kwargs["skill_files"] == {}


def test_execute_shell_command_serializes_spaces_and_quotes():
    executor = SkillsSandboxExecutor()
    command = ["echo", "hello world", 'a "quoted" value', "it's ok"]

    with patch.object(executor, "execute", return_value={}) as mock_execute:
        executor.execute_shell_command(command)

    code = mock_execute.call_args.kwargs["code"]
    assert f"subprocess.run({json.dumps(command)}, capture_output=True, text=True)" in code
    assert "capture_output=True, text=True" in code


def test_execute_shell_command_empty_command_returns_error():
    executor = SkillsSandboxExecutor()

    with patch.object(executor, "execute") as mock_execute:
        result = executor.execute_shell_command([])

    assert result["success"] is False
    assert "No command provided" in result["error"]
    mock_execute.assert_not_called()
