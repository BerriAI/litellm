"""
Claude Code Chat Completion Handler

Executes Claude Code CLI as a subprocess and streams responses back.

CLI Usage:
claude --system-prompt <prompt> --verbose --output-format stream-json
       --disallowedTools <tools> --max-turns 1 --model <model> -p
"""

import asyncio
import json
import os
import subprocess
import tempfile
import uuid
from typing import (
    IO,
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

from litellm._logging import verbose_logger
from litellm.types.utils import (
    Choices,
    Delta,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    Usage,
    _generate_id,
)

from ...base import BaseLLM
from .transformation import ClaudeCodeChatConfig


class ClaudeCodeError(Exception):
    """Custom exception for Claude Code errors."""

    def __init__(
        self,
        status_code: int,
        message: str,
    ):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)


# Constants
MAX_SYSTEM_PROMPT_LENGTH = 65536
CLAUDE_CODE_TIMEOUT = 600  # 10 minutes in seconds
CLAUDE_CODE_MAX_OUTPUT_TOKENS = "32000"


class ClaudeCodeChatCompletion(BaseLLM):
    """
    Handler for Claude Code CLI completions.

    This provider spawns the Claude Code CLI as a subprocess and streams
    the JSON output back as OpenAI-compatible responses.
    """

    def __init__(self) -> None:
        super().__init__()

    def completion(
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding: Any,
        logging_obj: Any,
        optional_params: dict,
        custom_prompt_dict: dict = {},
        acompletion: bool = False,
        litellm_params: dict = {},
        logger_fn: Optional[Callable] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Optional[Union[float, int]] = None,
        headers: Optional[dict] = None,
    ) -> Union[
        ModelResponse,
        Iterator[ModelResponseStream],
        Coroutine[Any, Any, Union[ModelResponse, AsyncIterator[ModelResponseStream]]],
    ]:
        """
        Execute a chat completion using Claude Code CLI.
        """
        config = ClaudeCodeChatConfig()
        stream = optional_params.get("stream", False)

        # Transform messages to Claude Code format
        system_prompt, transformed_messages = config.transform_messages_to_claude_code_format(
            messages=messages
        )

        # Get thinking budget if set
        thinking_budget_tokens = optional_params.get("thinking_budget_tokens", 0)

        # Convert timeout to float
        effective_timeout: float = float(timeout) if timeout is not None else CLAUDE_CODE_TIMEOUT

        if acompletion:
            return self.acompletion(
                model=model,
                system_prompt=system_prompt,
                messages=transformed_messages,
                model_response=model_response,
                stream=stream,
                config=config,
                thinking_budget_tokens=thinking_budget_tokens,
                logging_obj=logging_obj,
                timeout=effective_timeout,
            )

        if stream:
            return self._stream_completion(
                model=model,
                system_prompt=system_prompt,
                messages=transformed_messages,
                model_response=model_response,
                config=config,
                thinking_budget_tokens=thinking_budget_tokens,
                logging_obj=logging_obj,
                timeout=effective_timeout,
            )
        else:
            return self._sync_completion(
                model=model,
                system_prompt=system_prompt,
                messages=transformed_messages,
                model_response=model_response,
                config=config,
                thinking_budget_tokens=thinking_budget_tokens,
                logging_obj=logging_obj,
                timeout=effective_timeout,
            )

    async def acompletion(
        self,
        model: str,
        system_prompt: str,
        messages: list,
        model_response: ModelResponse,
        stream: bool,
        config: ClaudeCodeChatConfig,
        thinking_budget_tokens: int,
        logging_obj: Any,
        timeout: float,
    ) -> Union[ModelResponse, AsyncIterator[ModelResponseStream]]:
        """
        Async completion using Claude Code CLI.
        """
        if stream:
            return self._async_stream_completion(
                model=model,
                system_prompt=system_prompt,
                messages=messages,
                model_response=model_response,
                config=config,
                thinking_budget_tokens=thinking_budget_tokens,
                logging_obj=logging_obj,
                timeout=timeout,
            )
        else:
            return await self._async_completion(
                model=model,
                system_prompt=system_prompt,
                messages=messages,
                model_response=model_response,
                config=config,
                thinking_budget_tokens=thinking_budget_tokens,
                logging_obj=logging_obj,
                timeout=timeout,
            )

    def _build_cli_args(
        self,
        model: str,
        system_prompt: str,
        config: ClaudeCodeChatConfig,
        thinking_budget_tokens: int,
        use_file: bool = False,
    ) -> Tuple[List[str], Optional[str]]:
        """
        Build CLI arguments for Claude Code.

        Returns:
            Tuple of (args list, temp_file_path or None)
        """
        claude_path = config.get_claude_code_path()
        temp_file_path: Optional[str] = None

        # Handle long system prompts
        if len(system_prompt) > MAX_SYSTEM_PROMPT_LENGTH or os.name == "nt":
            # Use a temporary file for the system prompt
            temp_file_path = os.path.join(
                tempfile.gettempdir(),
                f"litellm-claude-code-{uuid.uuid4()}.txt"
            )
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(system_prompt)
            system_prompt_arg = ["--system-prompt-file", temp_file_path]
        else:
            system_prompt_arg = ["--system-prompt", system_prompt]

        args = [
            claude_path,
            *system_prompt_arg,
            "--verbose",
            "--output-format", "stream-json",
            "--disallowedTools", config.get_disabled_tools_string(),
            "--max-turns", str(config.max_turns),
            "--model", model,
            "-p",
        ]

        return args, temp_file_path

    def _get_env(self, thinking_budget_tokens: int) -> dict:
        """Get environment variables for Claude Code process."""
        env = os.environ.copy()
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = env.get(
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS", CLAUDE_CODE_MAX_OUTPUT_TOKENS
        )
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = env.get(
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1"
        )
        env["DISABLE_NON_ESSENTIAL_MODEL_CALLS"] = env.get(
            "DISABLE_NON_ESSENTIAL_MODEL_CALLS", "1"
        )
        env["MAX_THINKING_TOKENS"] = str(thinking_budget_tokens)

        # Don't use user's ANTHROPIC_API_KEY - let Claude Code handle auth
        env.pop("ANTHROPIC_API_KEY", None)

        return env

    def _sync_completion(
        self,
        model: str,
        system_prompt: str,
        messages: list,
        model_response: ModelResponse,
        config: ClaudeCodeChatConfig,
        thinking_budget_tokens: int,
        logging_obj: Any,
        timeout: float,
    ) -> ModelResponse:
        """
        Synchronous (non-streaming) completion.
        """
        args, temp_file_path = self._build_cli_args(
            model=model,
            system_prompt=system_prompt,
            config=config,
            thinking_budget_tokens=thinking_budget_tokens,
        )

        process: Optional[subprocess.Popen[str]] = None
        try:
            process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._get_env(thinking_budget_tokens),
                text=True,
            )

            # Send messages as JSON to stdin
            messages_json = json.dumps(messages)
            stdout, stderr = process.communicate(
                input=messages_json,
                timeout=timeout,
            )

            if process.returncode != 0:
                raise ClaudeCodeError(
                    status_code=process.returncode or 1,
                    message=f"Claude Code process failed: {stderr}",
                )

            # Parse the output
            return self._parse_response(stdout, model_response, model)

        except subprocess.TimeoutExpired:
            if process is not None:
                process.kill()
            raise ClaudeCodeError(
                status_code=408,
                message=f"Claude Code process timed out after {timeout} seconds",
            )
        except FileNotFoundError:
            raise ClaudeCodeError(
                status_code=500,
                message="Claude Code CLI not found. Make sure it's installed and in PATH.",
            )
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def _stream_completion(
        self,
        model: str,
        system_prompt: str,
        messages: list,
        model_response: ModelResponse,
        config: ClaudeCodeChatConfig,
        thinking_budget_tokens: int,
        logging_obj: Any,
        timeout: float,
    ) -> Iterator[ModelResponseStream]:
        """
        Streaming completion - yields chunks as they arrive.
        """
        args, temp_file_path = self._build_cli_args(
            model=model,
            system_prompt=system_prompt,
            config=config,
            thinking_budget_tokens=thinking_budget_tokens,
        )

        process: Optional[subprocess.Popen[str]] = None
        try:
            process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._get_env(thinking_budget_tokens),
                text=True,
                bufsize=1,  # Line buffered
            )

            # Send messages as JSON to stdin
            messages_json = json.dumps(messages)
            if process.stdin is not None:
                process.stdin.write(messages_json)
                process.stdin.close()

            response_id = f"chatcmpl-{_generate_id()}"

            # Read and yield chunks
            if process.stdout is not None:
                for chunk in self._parse_streaming_response(
                    process.stdout, model, response_id
                ):
                    yield chunk

            process.wait(timeout=timeout)

            if process.returncode != 0:
                if process.stderr is not None:
                    stderr = process.stderr.read()
                    verbose_logger.warning(f"Claude Code stderr: {stderr}")

        except subprocess.TimeoutExpired:
            if process is not None:
                process.kill()
            raise ClaudeCodeError(
                status_code=408,
                message=f"Claude Code process timed out after {timeout} seconds",
            )
        except FileNotFoundError:
            raise ClaudeCodeError(
                status_code=500,
                message="Claude Code CLI not found. Make sure it's installed and in PATH.",
            )
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    async def _async_completion(
        self,
        model: str,
        system_prompt: str,
        messages: list,
        model_response: ModelResponse,
        config: ClaudeCodeChatConfig,
        thinking_budget_tokens: int,
        logging_obj: Any,
        timeout: float,
    ) -> ModelResponse:
        """
        Async non-streaming completion.
        """
        args, temp_file_path = self._build_cli_args(
            model=model,
            system_prompt=system_prompt,
            config=config,
            thinking_budget_tokens=thinking_budget_tokens,
        )

        process: Optional[asyncio.subprocess.Process] = None
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._get_env(thinking_budget_tokens),
            )

            messages_json = json.dumps(messages).encode()
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=messages_json),
                timeout=timeout,
            )

            if process.returncode != 0:
                raise ClaudeCodeError(
                    status_code=process.returncode or 1,
                    message=f"Claude Code process failed: {stderr.decode()}",
                )

            return self._parse_response(stdout.decode(), model_response, model)

        except asyncio.TimeoutError:
            if process is not None:
                process.kill()
            raise ClaudeCodeError(
                status_code=408,
                message=f"Claude Code process timed out after {timeout} seconds",
            )
        except FileNotFoundError:
            raise ClaudeCodeError(
                status_code=500,
                message="Claude Code CLI not found. Make sure it's installed and in PATH.",
            )
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    async def _async_stream_completion(
        self,
        model: str,
        system_prompt: str,
        messages: list,
        model_response: ModelResponse,
        config: ClaudeCodeChatConfig,
        thinking_budget_tokens: int,
        logging_obj: Any,
        timeout: float,
    ) -> AsyncIterator[ModelResponseStream]:
        """
        Async streaming completion.
        """
        args, temp_file_path = self._build_cli_args(
            model=model,
            system_prompt=system_prompt,
            config=config,
            thinking_budget_tokens=thinking_budget_tokens,
        )

        process: Optional[asyncio.subprocess.Process] = None
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._get_env(thinking_budget_tokens),
            )

            messages_json = json.dumps(messages).encode()
            if process.stdin is not None:
                process.stdin.write(messages_json)
                await process.stdin.drain()
                process.stdin.close()

            response_id = f"chatcmpl-{_generate_id()}"

            if process.stdout is not None:
                async for chunk in self._parse_async_streaming_response(
                    process.stdout, model, response_id
                ):
                    yield chunk

            await asyncio.wait_for(process.wait(), timeout=timeout)

            if process.returncode != 0:
                if process.stderr is not None:
                    stderr = await process.stderr.read()
                    verbose_logger.warning(f"Claude Code stderr: {stderr.decode()}")

        except asyncio.TimeoutError:
            if process is not None:
                process.kill()
            raise ClaudeCodeError(
                status_code=408,
                message=f"Claude Code process timed out after {timeout} seconds",
            )
        except FileNotFoundError:
            raise ClaudeCodeError(
                status_code=500,
                message="Claude Code CLI not found. Make sure it's installed and in PATH.",
            )
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def _parse_response(
        self,
        output: str,
        model_response: ModelResponse,
        model: str,
    ) -> ModelResponse:
        """
        Parse the complete Claude Code output into a ModelResponse.
        """
        full_text = ""
        usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        total_cost = 0.0

        for line in output.strip().split("\n"):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            if chunk.get("type") == "assistant" and "message" in chunk:
                message = chunk["message"]

                # Extract text content
                for content_block in message.get("content", []):
                    if content_block.get("type") == "text":
                        full_text += content_block.get("text", "")

                # Extract usage
                msg_usage = message.get("usage", {})
                usage.prompt_tokens = msg_usage.get("input_tokens", 0)
                usage.completion_tokens = msg_usage.get("output_tokens", 0)
                usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

            elif chunk.get("type") == "result":
                total_cost = chunk.get("total_cost_usd", 0.0)

        # Build response
        model_response.id = f"chatcmpl-{_generate_id()}"
        model_response.model = model
        model_response.choices = [
            Choices(
                index=0,
                message=Message(role="assistant", content=full_text),
                finish_reason="stop",
            )
        ]
        setattr(model_response, "usage", usage)

        # Add cost to response metadata
        if hasattr(model_response, "_hidden_params"):
            model_response._hidden_params["response_cost"] = total_cost

        return model_response

    def _parse_streaming_response(
        self,
        stdout: IO[str],
        model: str,
        response_id: str,
    ) -> Iterator[ModelResponseStream]:
        """
        Parse streaming output from Claude Code CLI.
        """
        for line in stdout:
            line = line.strip()
            if not line:
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            for response_chunk in self._process_chunk(chunk, model, response_id):
                yield response_chunk

    async def _parse_async_streaming_response(
        self,
        stdout: asyncio.StreamReader,
        model: str,
        response_id: str,
    ) -> AsyncIterator[ModelResponseStream]:
        """
        Parse async streaming output from Claude Code CLI.
        """
        async for line in stdout:
            line_str = line.decode().strip()
            if not line_str:
                continue

            try:
                chunk = json.loads(line_str)
            except json.JSONDecodeError:
                continue

            for response_chunk in self._process_chunk(chunk, model, response_id):
                yield response_chunk

    def _process_chunk(
        self,
        chunk: dict,
        model: str,
        response_id: str,
    ) -> Iterator[ModelResponseStream]:
        """
        Process a single Claude Code chunk and yield ModelResponseStream objects.
        """
        if chunk.get("type") == "assistant" and "message" in chunk:
            message = chunk["message"]

            for content_block in message.get("content", []):
                content_type = content_block.get("type")

                if content_type == "text":
                    text = content_block.get("text", "")
                    if text:
                        yield ModelResponseStream(
                            id=response_id,
                            model=model,
                            choices=[
                                StreamingChoices(
                                    index=0,
                                    delta=Delta(content=text, role="assistant"),
                                    finish_reason=None,
                                )
                            ],
                        )

                elif content_type == "thinking":
                    thinking = content_block.get("thinking", "")
                    if thinking:
                        # Yield thinking as a separate chunk with metadata
                        yield ModelResponseStream(
                            id=response_id,
                            model=model,
                            choices=[
                                StreamingChoices(
                                    index=0,
                                    delta=Delta(
                                        content=None,
                                        role="assistant",
                                    ),
                                    finish_reason=None,
                                )
                            ],
                        )

            # Check for stop reason
            if message.get("stop_reason") is not None:
                yield ModelResponseStream(
                    id=response_id,
                    model=model,
                    choices=[
                        StreamingChoices(
                            index=0,
                            delta=Delta(),
                            finish_reason="stop",
                        )
                    ],
                )

        elif chunk.get("type") == "result":
            # Final chunk with usage/cost info
            pass  # Usage is handled separately
