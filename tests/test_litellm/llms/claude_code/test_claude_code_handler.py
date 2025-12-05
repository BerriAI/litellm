"""
Unit tests for Claude Code Chat Completion Handler.

Tests CLI argument building, response parsing, and chunk processing.
All tests use mocking - no real API calls.
"""
import json
import os
import sys
from unittest.mock import MagicMock, patch, mock_open

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.claude_code.chat.handler import (
    ClaudeCodeChatCompletion,
    ClaudeCodeError,
    MAX_SYSTEM_PROMPT_LENGTH,
    CLAUDE_CODE_TIMEOUT,
    CLAUDE_CODE_MAX_OUTPUT_TOKENS,
)
from litellm.llms.claude_code.chat.transformation import ClaudeCodeChatConfig
from litellm.types.utils import ModelResponse


class TestClaudeCodeChatCompletion:
    """Tests for ClaudeCodeChatCompletion class."""

    def test_init(self):
        """Test that ClaudeCodeChatCompletion can be instantiated."""
        handler = ClaudeCodeChatCompletion()
        assert handler is not None

    def test_build_cli_args_basic(self):
        """Test building CLI arguments with basic parameters."""
        handler = ClaudeCodeChatCompletion()
        config = ClaudeCodeChatConfig()

        args, temp_file = handler._build_cli_args(
            model="claude-sonnet-4-5-20250929",
            system_prompt="You are helpful.",
            config=config,
            thinking_budget_tokens=0,
        )

        # On Windows, it should use temp file; on Unix it depends on prompt length
        if os.name == "nt":
            assert temp_file is not None
            assert "--system-prompt-file" in args
        else:
            assert "--system-prompt" in args

        assert "claude" in args[0] or args[0] == "claude"
        assert "--verbose" in args
        assert "--output-format" in args
        assert "stream-json" in args
        assert "--disallowedTools" in args
        assert "--max-turns" in args
        assert "--model" in args
        assert "claude-sonnet-4-5-20250929" in args
        assert "-p" in args

    @patch("tempfile.gettempdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_build_cli_args_long_system_prompt(self, mock_file, mock_tempdir):
        """Test building CLI arguments with long system prompt uses temp file."""
        mock_tempdir.return_value = "/tmp"
        handler = ClaudeCodeChatCompletion()
        config = ClaudeCodeChatConfig()

        # Create a system prompt longer than MAX_SYSTEM_PROMPT_LENGTH
        long_prompt = "x" * (MAX_SYSTEM_PROMPT_LENGTH + 100)

        args, temp_file = handler._build_cli_args(
            model="claude-sonnet-4-5-20250929",
            system_prompt=long_prompt,
            config=config,
            thinking_budget_tokens=0,
        )

        assert temp_file is not None
        assert "--system-prompt-file" in args
        mock_file.assert_called()

    def test_get_env_basic(self):
        """Test getting environment variables for subprocess."""
        handler = ClaudeCodeChatCompletion()

        env = handler._get_env(thinking_budget_tokens=1000)

        assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS" in env
        assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == CLAUDE_CODE_MAX_OUTPUT_TOKENS
        assert "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC" in env
        assert "DISABLE_NON_ESSENTIAL_MODEL_CALLS" in env
        assert env["MAX_THINKING_TOKENS"] == "1000"

    def test_get_env_removes_anthropic_key(self):
        """Test that ANTHROPIC_API_KEY is removed from env."""
        handler = ClaudeCodeChatCompletion()

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            env = handler._get_env(thinking_budget_tokens=0)
            assert "ANTHROPIC_API_KEY" not in env


class TestResponseParsing:
    """Tests for response parsing methods."""

    def test_parse_response_basic(self):
        """Test parsing basic Claude Code output."""
        handler = ClaudeCodeChatCompletion()
        model_response = ModelResponse()

        output = json.dumps({
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "Hello, world!"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        })

        result = handler._parse_response(output, model_response, "claude-sonnet-4-5-20250929")

        assert result.choices[0].message.content == "Hello, world!"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15

    def test_parse_response_multiple_text_blocks(self):
        """Test parsing output with multiple text blocks."""
        handler = ClaudeCodeChatCompletion()
        model_response = ModelResponse()

        output = json.dumps({
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Hello, "},
                    {"type": "text", "text": "world!"},
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        })

        result = handler._parse_response(output, model_response, "claude-sonnet-4-5-20250929")

        assert result.choices[0].message.content == "Hello, world!"

    def test_parse_response_with_result_chunk(self):
        """Test parsing output with result chunk containing cost."""
        handler = ClaudeCodeChatCompletion()
        model_response = ModelResponse()
        model_response._hidden_params = {}

        output = (
            json.dumps({
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Hello"}],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
            }) + "\n" +
            json.dumps({
                "type": "result",
                "total_cost_usd": 0.005,
            })
        )

        result = handler._parse_response(output, model_response, "claude-sonnet-4-5-20250929")

        assert result.choices[0].message.content == "Hello"
        assert result._hidden_params.get("response_cost") == 0.005

    def test_parse_response_skips_invalid_json(self):
        """Test that invalid JSON lines are skipped."""
        handler = ClaudeCodeChatCompletion()
        model_response = ModelResponse()

        output = (
            "invalid json\n" +
            json.dumps({
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Hello"}],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
            })
        )

        result = handler._parse_response(output, model_response, "claude-sonnet-4-5-20250929")

        assert result.choices[0].message.content == "Hello"


class TestChunkProcessing:
    """Tests for streaming chunk processing."""

    def test_process_chunk_text_content(self):
        """Test processing chunk with text content."""
        handler = ClaudeCodeChatCompletion()

        chunk = {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "Hello"}],
            }
        }

        chunks = list(handler._process_chunk(chunk, "claude-sonnet-4-5-20250929", "test-id"))

        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.content == "Hello"
        assert chunks[0].choices[0].delta.role == "assistant"

    def test_process_chunk_with_stop_reason(self):
        """Test processing chunk with stop reason."""
        handler = ClaudeCodeChatCompletion()

        chunk = {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "Done"}],
                "stop_reason": "end_turn",
            }
        }

        chunks = list(handler._process_chunk(chunk, "claude-sonnet-4-5-20250929", "test-id"))

        # Should have content chunk + stop chunk
        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "Done"
        assert chunks[1].choices[0].finish_reason == "stop"

    def test_process_chunk_thinking_content(self):
        """Test processing chunk with thinking content."""
        handler = ClaudeCodeChatCompletion()

        chunk = {
            "type": "assistant",
            "message": {
                "content": [{"type": "thinking", "thinking": "Let me think..."}],
            }
        }

        chunks = list(handler._process_chunk(chunk, "claude-sonnet-4-5-20250929", "test-id"))

        # Thinking content should yield a chunk
        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.role == "assistant"

    def test_process_chunk_result_type(self):
        """Test that result type chunks are handled gracefully."""
        handler = ClaudeCodeChatCompletion()

        chunk = {
            "type": "result",
            "total_cost_usd": 0.01,
        }

        chunks = list(handler._process_chunk(chunk, "claude-sonnet-4-5-20250929", "test-id"))

        # Result chunks don't yield any response chunks
        assert len(chunks) == 0

    def test_process_chunk_empty_text(self):
        """Test processing chunk with empty text content."""
        handler = ClaudeCodeChatCompletion()

        chunk = {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": ""}],
            }
        }

        chunks = list(handler._process_chunk(chunk, "claude-sonnet-4-5-20250929", "test-id"))

        # Empty text should not yield a chunk
        assert len(chunks) == 0


class TestClaudeCodeError:
    """Tests for ClaudeCodeError exception."""

    def test_error_creation(self):
        """Test creating ClaudeCodeError."""
        error = ClaudeCodeError(status_code=500, message="Test error")

        assert error.status_code == 500
        assert error.message == "Test error"
        assert str(error) == "Test error"

    def test_error_inheritance(self):
        """Test that ClaudeCodeError is an Exception."""
        error = ClaudeCodeError(status_code=400, message="Bad request")

        assert isinstance(error, Exception)


class TestCompletionMocked:
    """Tests for completion method with mocked subprocess."""

    @patch("subprocess.Popen")
    def test_sync_completion_success(self, mock_popen):
        """Test successful sync completion with mocked subprocess."""
        handler = ClaudeCodeChatCompletion()

        # Mock process
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (
            json.dumps({
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Hello!"}],
                    "usage": {"input_tokens": 5, "output_tokens": 3},
                }
            }),
            ""
        )
        mock_popen.return_value = mock_process

        model_response = ModelResponse()
        config = ClaudeCodeChatConfig()

        result = handler._sync_completion(
            model="claude-sonnet-4-5-20250929",
            system_prompt="Be helpful",
            messages=[{"role": "user", "content": "Hi"}],
            model_response=model_response,
            config=config,
            thinking_budget_tokens=0,
            logging_obj=None,
            timeout=CLAUDE_CODE_TIMEOUT,
        )

        assert result.choices[0].message.content == "Hello!"
        mock_popen.assert_called_once()

    @patch("subprocess.Popen")
    def test_sync_completion_process_error(self, mock_popen):
        """Test sync completion handles process error."""
        handler = ClaudeCodeChatCompletion()

        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = ("", "Error occurred")
        mock_popen.return_value = mock_process

        model_response = ModelResponse()
        config = ClaudeCodeChatConfig()

        with pytest.raises(ClaudeCodeError) as exc_info:
            handler._sync_completion(
                model="claude-sonnet-4-5-20250929",
                system_prompt="Be helpful",
                messages=[{"role": "user", "content": "Hi"}],
                model_response=model_response,
                config=config,
                thinking_budget_tokens=0,
                logging_obj=None,
                timeout=CLAUDE_CODE_TIMEOUT,
            )

        assert exc_info.value.status_code == 1
        assert "failed" in exc_info.value.message.lower()

    @patch("subprocess.Popen")
    def test_sync_completion_cli_not_found(self, mock_popen):
        """Test sync completion handles CLI not found."""
        handler = ClaudeCodeChatCompletion()

        mock_popen.side_effect = FileNotFoundError()

        model_response = ModelResponse()
        config = ClaudeCodeChatConfig()

        with pytest.raises(ClaudeCodeError) as exc_info:
            handler._sync_completion(
                model="claude-sonnet-4-5-20250929",
                system_prompt="Be helpful",
                messages=[{"role": "user", "content": "Hi"}],
                model_response=model_response,
                config=config,
                thinking_budget_tokens=0,
                logging_obj=None,
                timeout=CLAUDE_CODE_TIMEOUT,
            )

        assert exc_info.value.status_code == 500
        assert "not found" in exc_info.value.message.lower()


class TestStreamingMocked:
    """Tests for streaming completion with mocked subprocess."""

    @patch("subprocess.Popen")
    def test_stream_completion_yields_chunks(self, mock_popen):
        """Test that streaming completion yields response chunks."""
        handler = ClaudeCodeChatCompletion()

        # Mock process with streaming output
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdin = MagicMock()
        mock_process.stderr = MagicMock()
        mock_process.stderr.read.return_value = ""

        # Mock stdout to return lines
        chunk_line = json.dumps({
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "Hello"}],
            }
        })
        mock_process.stdout = iter([chunk_line])
        mock_popen.return_value = mock_process

        config = ClaudeCodeChatConfig()

        chunks = list(handler._stream_completion(
            model="claude-sonnet-4-5-20250929",
            system_prompt="Be helpful",
            messages=[{"role": "user", "content": "Hi"}],
            model_response=ModelResponse(),
            config=config,
            thinking_budget_tokens=0,
            logging_obj=None,
            timeout=CLAUDE_CODE_TIMEOUT,
        ))

        assert len(chunks) >= 1
        assert chunks[0].choices[0].delta.content == "Hello"


class TestConstants:
    """Tests for module constants."""

    def test_max_system_prompt_length(self):
        """Test MAX_SYSTEM_PROMPT_LENGTH constant."""
        assert MAX_SYSTEM_PROMPT_LENGTH == 65536

    def test_claude_code_timeout(self):
        """Test CLAUDE_CODE_TIMEOUT constant."""
        assert CLAUDE_CODE_TIMEOUT == 600

    def test_claude_code_max_output_tokens(self):
        """Test CLAUDE_CODE_MAX_OUTPUT_TOKENS constant."""
        assert CLAUDE_CODE_MAX_OUTPUT_TOKENS == "32000"
