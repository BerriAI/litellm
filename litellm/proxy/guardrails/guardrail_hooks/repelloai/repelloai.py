from __future__ import annotations

from datetime import datetime
from typing import AsyncGenerator, Literal

from pydantic import TypeAdapter, ValidationError
from pydantic import BaseModel
from typing_extensions import TypeGuard

from fastapi import HTTPException
from httpx import HTTPError, Response as HttpxResponse

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,  # pyright: ignore[reportUnknownVariableType]
)
from litellm.proxy.guardrails._content_utils import build_inspection_messages
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
from litellm.types.proxy.guardrails.guardrail_hooks.repelloai import (
    RepelloAIAnalyzeResponse,
)
from litellm.types.utils import (
    CallTypesLiteral,
    GuardrailStatus,
    LLMResponseTypes,
    ModelResponse,
    ModelResponseStream,
)

DEFAULT_REPELLOAI_API_BASE = "https://argusapi.repello.ai/sdk/v1"
DEFAULT_REPELLOAI_TIMEOUT = 30.0
BLOCKED_VERDICT = "blocked"
FLAGGED_VERDICT = "flagged"
PASSED_VERDICT = "passed"

# Argus returns these for a permanently broken guardrail (bad key, unknown
# asset_id, malformed payload), not a transient outage. They must always
# block, never honour fail_open.
CONFIG_ERROR_STATUS_CODES = frozenset({400, 401, 403, 404, 422})
_SCHEMA_SCALAR_KEYS = frozenset(("name", "description", "title", "const", "default"))
_SCHEMA_LIST_KEYS = frozenset(("enum", "examples"))
_SCHEMA_EXTRACTED_KEYS = _SCHEMA_SCALAR_KEYS | _SCHEMA_LIST_KEYS


class RepelloAIGuardrailMissingSecrets(Exception):
    pass


def _is_object_dict(value: object) -> TypeGuard[dict[str, object]]:  # guard-ok: isinstance narrows correctly; predicate is trivially correct  # fmt: skip
    return isinstance(value, dict)


def _is_object_list(value: object) -> TypeGuard[list[object]]:  # guard-ok: isinstance narrows correctly; predicate is trivially correct  # fmt: skip
    return isinstance(value, list)


class RepelloAIGuardrail(CustomGuardrail):
    @staticmethod
    def _get_field(obj: object, key: str) -> object:
        if _is_object_dict(obj):
            return obj.get(key)
        return getattr(obj, key, None)

    @classmethod
    def _extract_tool_call_args_from_message(cls, message: object) -> list[str]:
        args: list[str] = []

        tool_calls = cls._get_field(message, "tool_calls")
        if _is_object_list(tool_calls):
            for tool_call in tool_calls:
                function = cls._get_field(tool_call, "function")
                arguments = cls._get_field(function, "arguments")
                if isinstance(arguments, str) and arguments.strip():
                    args.append(arguments)

        function_call = cls._get_field(message, "function_call")
        arguments = cls._get_field(function_call, "arguments")
        if isinstance(arguments, str) and arguments.strip():
            args.append(arguments)

        return args

    @staticmethod
    def _iter_schema_text(node: object) -> list[str]:
        texts: list[str] = []
        stack: list[object] = [node]

        while stack:
            current = stack.pop()
            if _is_object_dict(current):
                for key in _SCHEMA_SCALAR_KEYS:
                    value = current.get(key)
                    if isinstance(value, str) and value:
                        texts.append(value)
                for key in _SCHEMA_LIST_KEYS:
                    items = current.get(key)
                    if _is_object_list(items):
                        for item in items:
                            if isinstance(item, str) and item:
                                texts.append(item)
                remaining: list[object] = [
                    v for k, v in current.items() if k not in _SCHEMA_EXTRACTED_KEYS
                ]
                stack.extend(reversed(remaining))
            elif _is_object_list(current):
                stack.extend(reversed(current))

        return texts

    @classmethod
    def _extract_tool_definition_text(cls, data: dict[str, object]) -> list[str]:
        texts: list[str] = []

        tools = data.get("tools")
        for tool in tools if _is_object_list(tools) else []:
            if not _is_object_dict(tool):
                continue
            function = tool.get("function")
            if _is_object_dict(function):
                texts.extend(cls._iter_schema_text(function))

        functions = data.get("functions")
        for function in functions if _is_object_list(functions) else []:
            if _is_object_dict(function):
                texts.extend(cls._iter_schema_text(function))

        return texts

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        asset_id: str | None = None,
        unreachable_fallback: Literal["fail_closed", "fail_open"] = "fail_closed",
        guardrail_name: str | None = None,
        event_hook: (
            GuardrailEventHooks | list[GuardrailEventHooks] | Mode | None
        ) = None,
        default_on: bool = False,
    ):
        self.repelloai_api_key = (
            api_key
            or get_secret_str("ARGUS_API_KEY")
            or get_secret_str("REPELLOAI_API_KEY")
            or ""
        )
        if not self.repelloai_api_key:
            raise RepelloAIGuardrailMissingSecrets(
                "Couldn't get Repello API key. Set `ARGUS_API_KEY` in the environment "
                "or pass `api_key` to the guardrail in the config file."
            )

        self.asset_id = asset_id
        if not self.asset_id:
            raise ValueError(
                "Repello guardrail requires an `asset_id`. Create an asset in the Repello "
                "dashboard and set `asset_id` on the guardrail in the config file."
            )

        self.api_base = (
            api_base
            or get_secret_str("REPELLOAI_API_BASE")
            or DEFAULT_REPELLOAI_API_BASE
        )
        self.unreachable_fallback: Literal["fail_closed", "fail_open"] = (
            "fail_open" if unreachable_fallback == "fail_open" else "fail_closed"
        )
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
            params={"timeout": DEFAULT_REPELLOAI_TIMEOUT},
        )
        super().__init__(  # pyright: ignore[reportUnknownMemberType]
            guardrail_name=guardrail_name,
            event_hook=event_hook,
            default_on=default_on,
        )

    async def _call_analyze(
        self,
        text: str,
        stage: Literal["prompt", "response"],
        request_data: dict[str, object],
        event_type: GuardrailEventHooks,
    ) -> RepelloAIAnalyzeResponse | None:
        endpoint = f"{self.api_base}/analyze/{stage}"
        request: dict[str, object] = {
            "asset_id": self.asset_id or "",
            "scan_data": {stage: text},
        }

        status: GuardrailStatus = "success"
        guardrail_json_response: str | dict[str, object] | list[dict[str, object]] = ""
        start_time: datetime = datetime.now()
        repelloai_response: RepelloAIAnalyzeResponse | None = None
        try:
            verbose_proxy_logger.debug("RepelloAI Argus request: %s", request)
            raw_response: HttpxResponse | None = await self.async_handler.post(  # pyright: ignore[reportUnknownMemberType]
                url=endpoint,
                headers={"X-API-Key": self.repelloai_api_key},
                json=request,
            )
            if raw_response is None:
                raise ValueError("RepelloAI Argus returned no response")
            response: HttpxResponse = raw_response
            self._raise_for_config_error(response)
            response.raise_for_status()
            try:
                repelloai_response = TypeAdapter(
                    RepelloAIAnalyzeResponse
                ).validate_json(response.text)
            except ValidationError as e:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "RepelloAI Argus guardrail returned invalid JSON",
                        "status_code": response.status_code,
                    },
                ) from e
            verbose_proxy_logger.debug(
                "RepelloAI Argus response: %s", repelloai_response
            )
            if self._verdict_blocks(repelloai_response):
                status = "guardrail_intervened"
            return repelloai_response
        except HTTPException as e:
            status = "guardrail_failed_to_respond"
            guardrail_json_response = (
                str(e.detail) if not isinstance(e.detail, (dict, list)) else e.detail
            )  # type: ignore[assignment]
            raise
        except HTTPError as e:
            status = "guardrail_failed_to_respond"
            guardrail_json_response = str(e)
            return self._handle_unreachable(e)
        except Exception as e:
            status = "guardrail_failed_to_respond"
            guardrail_json_response = str(e)
            raise HTTPException(
                status_code=500, detail={"error": "RepelloAI Argus guardrail failed"}
            ) from e
        finally:
            end_time = datetime.now()
            if repelloai_response is not None:
                guardrail_json_response = dict(repelloai_response)
            self.add_standard_logging_guardrail_information_to_request_data(  # pyright: ignore[reportUnknownMemberType]
                guardrail_json_response=guardrail_json_response,
                guardrail_status=status,
                request_data=request_data,
                start_time=start_time.timestamp(),
                end_time=end_time.timestamp(),
                duration=(end_time - start_time).total_seconds(),
                masked_entity_count={},
                event_type=event_type,
            )

    @staticmethod
    def _raise_for_config_error(response: HttpxResponse) -> None:
        if response.status_code in CONFIG_ERROR_STATUS_CODES:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "RepelloAI Argus guardrail is misconfigured",
                    "status_code": response.status_code,
                },
            )

    def _verdict_blocks(
        self, repelloai_response: RepelloAIAnalyzeResponse | None
    ) -> bool:
        if repelloai_response is None:
            return False
        verdict = repelloai_response.get("verdict")
        if verdict == BLOCKED_VERDICT:
            return True
        if verdict in (PASSED_VERDICT, FLAGGED_VERDICT):
            return False
        verbose_proxy_logger.warning(
            "RepelloAI Argus returned an unrecognized verdict (%s) - blocking.",
            verdict,
        )
        return True

    def _handle_unreachable(self, error: Exception) -> RepelloAIAnalyzeResponse | None:
        verbose_proxy_logger.warning("RepelloAI Argus unreachable: %s", str(error))
        if self.unreachable_fallback == "fail_closed":
            raise HTTPException(
                status_code=500,
                detail={"error": "RepelloAI Argus guardrail unreachable"},
            )
        return None

    def _raise_if_blocked(
        self, repelloai_response: RepelloAIAnalyzeResponse | None
    ) -> None:
        if repelloai_response is None:
            return
        if self._verdict_blocks(repelloai_response):
            raise HTTPException(
                status_code=400,
                detail=self._format_blocked_detail(repelloai_response),
            )
        self._log_flagged_verdict(repelloai_response)

    @classmethod
    def _format_blocked_detail(
        cls, repelloai_response: RepelloAIAnalyzeResponse
    ) -> str:
        policies = repelloai_response.get("policies_violated")
        if not isinstance(policies, list) or not policies:
            return "Blocked by RepelloAI Argus guardrail."

        formatted_policies: list[str] = []
        for policy in policies:
            policy_name = policy.get("policy_name") or "unknown_policy"
            details: list[str] = []
            action_taken = policy.get("action_taken")
            if action_taken:
                details.append(f"action: {action_taken}")
            policy_details = policy.get("details")
            if isinstance(policy_details, dict):
                score = policy_details.get("score")
                if score is not None:
                    details.append(f"score: {score}")
            suffix = f" ({', '.join(details)})" if details else ""
            formatted_policies.append(f"{policy_name}{suffix}")

        if not formatted_policies:
            return "Blocked by RepelloAI Argus guardrail."
        return f"Blocked by RepelloAI Argus guardrail. Policies violated: {'; '.join(formatted_policies)}."

    @staticmethod
    def _log_flagged_verdict(repelloai_response: RepelloAIAnalyzeResponse) -> None:
        if repelloai_response.get("verdict") == FLAGGED_VERDICT:
            verbose_proxy_logger.warning(
                "RepelloAI Argus flagged content (allowed): %s",
                repelloai_response.get("policies_violated"),
            )

    @staticmethod
    def _extract_prompt_message_text(data: dict[str, object]) -> list[str]:
        messages = build_inspection_messages(data)
        return [
            content
            for message in messages
            if isinstance(content := message.get("content"), str) and content
        ]

    @staticmethod
    def _extract_input_text_parts(content: object) -> list[str]:
        if not _is_object_list(content):
            return []
        return [
            text
            for part in content
            if _is_object_dict(part) and part.get("type") == "input_text"
            if isinstance(text := part.get("text"), str) and text
        ]

    @staticmethod
    def _extract_prompt_field_text(data: dict[str, object]) -> list[str]:
        prompt = data.get("prompt")
        if isinstance(prompt, str) and prompt:
            return [prompt]
        if _is_object_list(prompt):
            return [item for item in prompt if isinstance(item, str) and item]
        return []

    @classmethod
    def _extract_prompt_text(cls, data: dict[str, object]) -> str | None:
        texts = cls._extract_prompt_message_text(data)
        texts.extend(cls._extract_prompt_field_text(data))

        instructions = data.get("instructions")
        if isinstance(instructions, str) and instructions:
            texts.append(instructions)

        raw_messages = data.get("messages")
        if _is_object_list(raw_messages):
            for message in raw_messages:
                texts.extend(cls._extract_tool_call_args_from_message(message))

        raw_input = data.get("input")
        if _is_object_list(raw_input):
            for item in raw_input:
                if _is_object_dict(item):
                    if "role" not in item:
                        continue
                    texts.extend(cls._extract_tool_call_args_from_message(item))
                    texts.extend(cls._extract_input_text_parts(item.get("content")))

        texts.extend(cls._extract_tool_definition_text(data))
        return "\n".join(text for text in texts if text) if texts else None

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: litellm.DualCache,
        data: dict[str, object],
        call_type: CallTypesLiteral,
    ) -> Exception | str | dict[str, object] | None:
        verbose_proxy_logger.debug("RepelloAI Argus: pre_call_hook")

        event_type = GuardrailEventHooks.pre_call
        if (
            self.should_run_guardrail(  # pyright: ignore[reportUnknownMemberType]
                data=data, event_type=event_type
            )
            is not True
        ):
            return data

        text = self._extract_prompt_text(data)
        if not text:
            verbose_proxy_logger.warning(
                "RepelloAI Argus: no inspectable prompt text in data - skipping."
            )
            return data

        repelloai_response = await self._call_analyze(
            text=text,
            stage="prompt",
            request_data=data,
            event_type=event_type,
        )
        self._raise_if_blocked(repelloai_response)

        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        return data

    async def async_post_call_success_hook(
        self,
        data: dict[str, object],
        user_api_key_dict: UserAPIKeyAuth,
        response: LLMResponseTypes,
    ):
        verbose_proxy_logger.debug("RepelloAI Argus: post_call_success_hook")

        event_type = GuardrailEventHooks.post_call
        if (
            self.should_run_guardrail(  # pyright: ignore[reportUnknownMemberType]
                data=data, event_type=event_type
            )
            is not True
        ):
            return response

        text = self._extract_response_text(response)
        if not text:
            verbose_proxy_logger.warning(
                "RepelloAI Argus: no inspectable response text - skipping."
            )
            return response

        repelloai_response = await self._call_analyze(
            text=text,
            stage="response",
            request_data=data,
            event_type=event_type,
        )
        self._raise_if_blocked(repelloai_response)

        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncGenerator[ModelResponseStream, None],
        request_data: dict[str, object],
    ) -> AsyncGenerator[ModelResponseStream, None]:
        from litellm import main as litellm_main

        event_type = GuardrailEventHooks.post_call
        if (
            self.should_run_guardrail(  # pyright: ignore[reportUnknownMemberType]
                data=request_data, event_type=event_type
            )
            is not True
        ):
            async for chunk in response:
                yield chunk
            return

        chunks: list[ModelResponseStream] = []
        async for chunk in response:
            chunks.append(chunk)

        assembled = litellm_main.stream_chunk_builder(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            chunks=chunks
        )
        text = (
            self._extract_response_text(assembled)
            if isinstance(assembled, ModelResponse)
            else None
        )
        if text:
            repelloai_response = await self._call_analyze(
                text=text,
                stage="response",
                request_data=request_data,
                event_type=event_type,
            )
            if repelloai_response is not None:
                self._log_flagged_verdict(repelloai_response)
            if self._verdict_blocks(repelloai_response):
                from litellm.proxy.proxy_server import StreamingCallbackError

                raise StreamingCallbackError("Blocked by RepelloAI Argus guardrail")
            add_guardrail_to_applied_guardrails_header(
                request_data=request_data, guardrail_name=self.guardrail_name
            )
        else:
            verbose_proxy_logger.warning(
                "RepelloAI Argus: no inspectable text in streamed response; skipping scan. "
                "guardrail=%s assembled_type=%s",
                self.guardrail_name,
                type(assembled).__name__,
            )

        for chunk in chunks:
            yield chunk

    @staticmethod
    def _extract_response_text(response: object) -> str | None:
        if _is_object_dict(response):
            response_dict = response
        elif isinstance(response, ModelResponse):
            response_dict = (
                response.model_dump()  # pyright: ignore[reportUnknownMemberType]
            )
        else:
            output_text = getattr(response, "output_text", None)
            if isinstance(output_text, str) and output_text:
                return output_text
            response_dict = {}

        text = RepelloAIGuardrail._extract_chat_completion_text(response_dict)
        if text:
            return text
        return RepelloAIGuardrail._extract_responses_api_text(response_dict)

    @classmethod
    def _extract_chat_completion_text(
        cls, response_dict: dict[str, object]
    ) -> str | None:
        choices = response_dict.get("choices")
        if not _is_object_list(choices):
            return None
        parts: list[str] = []
        for choice in choices:
            if not _is_object_dict(choice):
                continue
            message = choice.get("message")
            if _is_object_dict(message):
                content = message.get("content")
                if isinstance(content, str) and content:
                    parts.append(content)
                parts.extend(cls._extract_tool_call_args_from_message(message))
            text = choice.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        return "\n".join(parts) if parts else None

    @staticmethod
    def _extract_responses_api_text(response_dict: dict[str, object]) -> str | None:
        output = response_dict.get("output")
        if not _is_object_list(output):
            return None
        texts: list[str] = []
        for output_item in output:
            if not _is_object_dict(output_item):
                continue
            item_type = output_item.get("type")
            if item_type == "function_call":
                arguments = output_item.get("arguments")
                if isinstance(arguments, str) and arguments:
                    texts.append(arguments)
                continue
            if item_type != "message":
                continue
            content = output_item.get("content")
            if not _is_object_list(content):
                continue
            for content_item in content:
                if not _is_object_dict(content_item):
                    continue
                if content_item.get("type") not in ("output_text", "text"):
                    continue
                text = content_item.get("text")
                if isinstance(text, str) and text:
                    texts.append(text)
        return "".join(texts) if texts else None

    @staticmethod
    def get_config_model() -> type[GuardrailConfigModel[BaseModel]] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.repelloai import (
            RepelloAIGuardrailConfigModel,
        )

        return RepelloAIGuardrailConfigModel
