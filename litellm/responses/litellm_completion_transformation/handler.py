"""
Handler for transforming responses api requests to litellm.completion requests.

When the Responses API ``shell`` tool is used with a provider that lacks native
shell / code-execution support, this handler transparently:

1. Converts the shell tool to a synthetic ``_litellm_shell`` function tool
   (done in the transformation layer).
2. Detects when the model calls ``_litellm_shell`` in the response.
3. Executes the command in a sandboxed Docker container via
   ``SkillsSandboxExecutor``.
4. Feeds stdout / stderr back as a tool result and re-invokes the model.
5. Repeats until the model produces a final (non-tool-call) response.
"""

import asyncio
import json
from typing import Any, Coroutine, Dict, List, Optional, Union

import litellm
from litellm._logging import verbose_logger
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.responses.shell_tool_handler import (
    MAX_SHELL_ITERATIONS,
    execute_shell_calls_for_completion,
    format_shell_result,
)
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
)
from litellm.types.utils import ModelResponse

_SHELL_TOOL_NAME = LiteLLMCompletionResponsesConfig.LITELLM_SHELL_TOOL_NAME


class LiteLLMCompletionTransformationHandler:

    def response_api_handler(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
        custom_llm_provider: Optional[str] = None,
        _is_async: bool = False,
        stream: Optional[bool] = None,
        extra_headers: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[
        ResponsesAPIResponse,
        BaseResponsesAPIStreamingIterator,
        Coroutine[
            Any, Any, Union[ResponsesAPIResponse, BaseResponsesAPIStreamingIterator]
        ],
    ]:
        litellm_completion_request: dict = (
            LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
                model=model,
                input=input,
                responses_api_request=responses_api_request,
                custom_llm_provider=custom_llm_provider,
                stream=stream,
                extra_headers=extra_headers,
                **kwargs,
            )
        )

        if _is_async:
            return self.async_response_api_handler(
                litellm_completion_request=litellm_completion_request,
                request_input=input,
                responses_api_request=responses_api_request,
                **kwargs,
            )

        completion_args = {}
        completion_args.update(kwargs)
        completion_args.update(litellm_completion_request)

        litellm_completion_response: Union[
            ModelResponse, litellm.CustomStreamWrapper
        ] = litellm.completion(
            **litellm_completion_request,
            **kwargs,
        )

        needs_shell = LiteLLMCompletionResponsesConfig.request_has_litellm_shell_tool(
            litellm_completion_request.get("tools")
        )
        if needs_shell and isinstance(litellm_completion_response, ModelResponse):
            litellm_completion_response = self._run_shell_execution_loop_sync(
                initial_response=litellm_completion_response,
                completion_args=completion_args,
            )

        if isinstance(litellm_completion_response, ModelResponse):
            responses_api_response: ResponsesAPIResponse = (
                LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                    chat_completion_response=litellm_completion_response,
                    request_input=input,
                    responses_api_request=responses_api_request,
                )
            )

            return responses_api_response

        elif isinstance(litellm_completion_response, litellm.CustomStreamWrapper):
            return LiteLLMCompletionStreamingIterator(
                model=model,
                litellm_custom_stream_wrapper=litellm_completion_response,
                request_input=input,
                responses_api_request=responses_api_request,
                custom_llm_provider=custom_llm_provider,
                litellm_metadata=kwargs.get("litellm_metadata", {}),
            )
        raise ValueError(f"Unexpected response type: {type(litellm_completion_response)}")

    async def async_response_api_handler(
        self,
        litellm_completion_request: dict,
        request_input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
        **kwargs,
    ) -> Union[ResponsesAPIResponse, BaseResponsesAPIStreamingIterator]:

        previous_response_id: Optional[str] = responses_api_request.get(
            "previous_response_id"
        )
        if previous_response_id:
            litellm_completion_request = await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
                previous_response_id=previous_response_id,
                litellm_completion_request=litellm_completion_request,
            )

        acompletion_args = {}
        acompletion_args.update(kwargs)
        acompletion_args.update(litellm_completion_request)

        litellm_completion_response: Union[
            ModelResponse, litellm.CustomStreamWrapper
        ] = await litellm.acompletion(
            **acompletion_args,
        )

        # --- sandbox shell execution loop ---
        needs_shell = LiteLLMCompletionResponsesConfig.request_has_litellm_shell_tool(
            litellm_completion_request.get("tools")
        )
        if needs_shell and isinstance(litellm_completion_response, ModelResponse):
            litellm_completion_response = await self._run_shell_execution_loop(
                initial_response=litellm_completion_response,
                completion_args=acompletion_args,
            )

        if isinstance(litellm_completion_response, ModelResponse):
            responses_api_response: ResponsesAPIResponse = (
                LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                    chat_completion_response=litellm_completion_response,
                    request_input=request_input,
                    responses_api_request=responses_api_request,
                )
            )

            return responses_api_response

        elif isinstance(litellm_completion_response, litellm.CustomStreamWrapper):
            return LiteLLMCompletionStreamingIterator(
                model=litellm_completion_request.get("model") or "",
                litellm_custom_stream_wrapper=litellm_completion_response,
                request_input=request_input,
                responses_api_request=responses_api_request,
                custom_llm_provider=litellm_completion_request.get(
                    "custom_llm_provider"
                ),
                litellm_metadata=kwargs.get("litellm_metadata", {}),
            )
        raise ValueError(f"Unexpected response type: {type(litellm_completion_response)}")

    # ------------------------------------------------------------------
    # Sandbox shell execution loop
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_shell_tool_calls(
        response: ModelResponse,
    ) -> List[Dict[str, Any]]:
        """
        Return a list of ``_litellm_shell`` tool calls from a ModelResponse.

        Each entry is ``{"id": str, "command": List[str]}``.
        """
        shell_calls: List[Dict[str, Any]] = []
        for choice in getattr(response, "choices", []):
            msg = getattr(choice, "message", None)
            if not msg or not getattr(msg, "tool_calls", None):
                continue
            for tc in msg.tool_calls:
                if getattr(tc, "type", None) != "function":
                    continue
                fn = getattr(tc, "function", None)
                if not fn or getattr(fn, "name", None) != _SHELL_TOOL_NAME:
                    continue
                try:
                    args = json.loads(fn.arguments or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                command = args.get("command")
                if isinstance(command, list):
                    shell_calls.append({"id": tc.id, "command": command})
                elif isinstance(command, str):
                    shell_calls.append({"id": tc.id, "command": [command]})
        return shell_calls

    @staticmethod
    def _build_assistant_message(response: ModelResponse) -> Dict[str, Any]:
        """
        Build a Chat Completion assistant message dict from a ModelResponse.
        """
        msg = response.choices[0].message  # type: ignore[union-attr]
        assistant_dict: Dict[str, Any] = {
            "role": "assistant",
            "content": getattr(msg, "content", None),
        }
        if getattr(msg, "tool_calls", None):
            assistant_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in (msg.tool_calls or [])
            ]
        return assistant_dict

    def _prepare_next_completion(
        self,
        current_response: ModelResponse,
        shell_calls: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
        completion_args: Dict[str, Any],
        executor: Any,
        iteration: int,
        log_prefix: str,
    ) -> Dict[str, Any]:
        """
        Execute pending shell calls (sync), append results to *messages*,
        and return the kwargs for the next completion call.

        Shared by both the sync and async shell execution loops.
        The async variant (:meth:`_async_prepare_next_completion`) offloads
        sandbox calls to a thread pool.
        """
        messages.append(self._build_assistant_message(current_response))
        messages.extend(
            execute_shell_calls_for_completion(executor, shell_calls)
        )

        verbose_logger.debug(
            "LiteLLMCompletionTransformationHandler: %sshell loop iteration %d, "
            "executed %d command(s)",
            log_prefix,
            iteration + 1,
            len(shell_calls),
        )

        next_args = dict(completion_args)
        next_args["messages"] = messages
        next_args.pop("stream", None)
        return next_args

    async def _async_prepare_next_completion(
        self,
        current_response: ModelResponse,
        shell_calls: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
        completion_args: Dict[str, Any],
        executor: Any,
        iteration: int,
    ) -> Dict[str, Any]:
        """
        Async variant of :meth:`_prepare_next_completion`.

        Runs each ``execute_shell_command`` call in a thread pool via
        ``run_in_executor`` to avoid blocking the event loop.
        """
        messages.append(self._build_assistant_message(current_response))

        loop = asyncio.get_running_loop()
        for sc in shell_calls:
            result = await loop.run_in_executor(
                None, executor.execute_shell_command, sc["command"]
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": sc["id"],
                    "content": format_shell_result(result),
                }
            )

        verbose_logger.debug(
            "LiteLLMCompletionTransformationHandler: shell loop iteration %d, "
            "executed %d command(s)",
            iteration + 1,
            len(shell_calls),
        )

        next_args = dict(completion_args)
        next_args["messages"] = messages
        next_args.pop("stream", None)
        return next_args

    def _run_shell_execution_loop_sync(
        self,
        initial_response: ModelResponse,
        completion_args: Dict[str, Any],
    ) -> ModelResponse:
        """Synchronous version of :meth:`_run_shell_execution_loop`."""
        from litellm.llms.litellm_proxy.skills.sandbox_executor import (
            SkillsSandboxExecutor,
        )

        current_response = initial_response
        shell_calls = self._extract_shell_tool_calls(current_response)
        if not shell_calls:
            return current_response

        executor = SkillsSandboxExecutor()
        messages: List[Dict[str, Any]] = list(
            completion_args.get("messages") or []
        )

        for _iteration in range(MAX_SHELL_ITERATIONS):
            if not shell_calls:
                break
            next_args = self._prepare_next_completion(
                current_response, shell_calls, messages,
                completion_args, executor, _iteration, "sync ",
            )
            current_response = litellm.completion(**next_args)  # type: ignore[assignment]
            shell_calls = self._extract_shell_tool_calls(current_response)

        if shell_calls:
            verbose_logger.warning(
                "LiteLLMCompletionTransformationHandler: max shell iterations "
                "(%d) reached, returning last response",
                MAX_SHELL_ITERATIONS,
            )

        return current_response

    async def _run_shell_execution_loop(
        self,
        initial_response: ModelResponse,
        completion_args: Dict[str, Any],
    ) -> ModelResponse:
        """
        If the model called ``_litellm_shell``, execute the command in a
        sandboxed container, feed the result back, and repeat until the
        model produces a final non-tool-call response.

        Sandbox calls are offloaded to a thread pool via ``run_in_executor``
        to avoid blocking the event loop.

        Returns the final ``ModelResponse``.
        """
        from litellm.llms.litellm_proxy.skills.sandbox_executor import (
            SkillsSandboxExecutor,
        )

        current_response = initial_response
        shell_calls = self._extract_shell_tool_calls(current_response)
        if not shell_calls:
            return current_response

        executor = SkillsSandboxExecutor()
        messages: List[Dict[str, Any]] = list(
            completion_args.get("messages") or []
        )

        for _iteration in range(MAX_SHELL_ITERATIONS):
            if not shell_calls:
                break
            next_args = await self._async_prepare_next_completion(
                current_response, shell_calls, messages,
                completion_args, executor, _iteration,
            )
            current_response = await litellm.acompletion(**next_args)  # type: ignore[assignment]
            shell_calls = self._extract_shell_tool_calls(current_response)

        if shell_calls:
            verbose_logger.warning(
                "LiteLLMCompletionTransformationHandler: max shell iterations "
                "(%d) reached, returning last response",
                MAX_SHELL_ITERATIONS,
            )

        return current_response
