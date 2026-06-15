from datetime import datetime
from typing import AsyncGenerator, Dict, List, Literal, Optional, Type, Union

from fastapi import HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails._content_utils import build_inspection_messages
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
from litellm.types.proxy.guardrails.guardrail_hooks.repelloai import (
    RepelloAIAnalyzeResponse,
)
from litellm.types.utils import (
    CallTypesLiteral,
    GuardrailStatus,
    ModelResponse,
    ModelResponseStream,
)

DEFAULT_REPELLOAI_API_BASE = "https://argusapi.repello.ai/sdk/v1"
DEFAULT_REPELLOAI_TIMEOUT = 30.0
BLOCKED_VERDICT = "blocked"
FLAGGED_VERDICT = "flagged"
PASSED_VERDICT = "passed"
UnreachableFallback = Literal["fail_closed", "fail_open"]

# Argus returns these for a permanently broken guardrail (bad key, unknown
# asset_id, malformed payload), not a transient outage. They must always
# block, never honour fail_open.
CONFIG_ERROR_STATUS_CODES = frozenset({400, 401, 403, 404, 422})


class RepelloAIGuardrailMissingSecrets(Exception):
    pass


class RepelloAIGuardrail(CustomGuardrail):
    @staticmethod
    def _get_field(obj: object, key: str) -> object:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    @classmethod
    def _extract_tool_call_args_from_message(cls, message: object) -> List[str]:
        args: List[str] = []

        tool_calls = cls._get_field(message, "tool_calls")
        if isinstance(tool_calls, list):
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

    @classmethod
    def _iter_schema_text(cls, node: object) -> List[str]:
        texts: List[str] = []
        stack: List[object] = [node]
        scalar_keys = ("name", "description", "title", "const", "default")
        list_keys = ("enum", "examples")

        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                for key in scalar_keys:
                    value = current.get(key)
                    if isinstance(value, str) and value:
                        texts.append(value)
                for key in list_keys:
                    items = current.get(key)
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, str) and item:
                                texts.append(item)
                stack.extend(reversed(list(current.values())))
            elif isinstance(current, list):
                stack.extend(reversed(current))

        return texts

    @classmethod
    def _extract_tool_definition_text(cls, data: Dict) -> List[str]:
        texts: List[str] = []

        for tool in data.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            if isinstance(function, dict):
                texts.extend(cls._iter_schema_text(function))

        for function in data.get("functions") or []:
            if isinstance(function, dict):
                texts.extend(cls._iter_schema_text(function))

        return texts

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        asset_id: Optional[str] = None,
        unreachable_fallback: UnreachableFallback = "fail_closed",
        **kwargs,
    ):
        """RepelloAI Argus guardrail.

        Scans prompts (pre_call) and responses (post_call) by calling the
        hosted RepelloAI Argus API. The set of policies enforced is configured per
        asset_id in the Repello dashboard.

        Args:
            api_key: Repello API key. Falls back to the ARGUS_API_KEY env var
                (or the legacy REPELLOAI_API_KEY).
            api_base: Repello API base URL. Defaults to the hosted endpoint.
            asset_id: Repello asset whose dashboard policies are enforced. Required.
            unreachable_fallback: Behaviour when the Repello API is unreachable /
                errors: fail_closed (block, the default) or fail_open
                (allow + warn).
        """
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
            params={"timeout": DEFAULT_REPELLOAI_TIMEOUT},
        )
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
        self.unreachable_fallback: UnreachableFallback = (
            "fail_open" if unreachable_fallback == "fail_open" else "fail_closed"
        )
        super().__init__(**kwargs)

    async def _call_analyze(
        self,
        text: str,
        stage: str,
        request_data: Dict,
        event_type: GuardrailEventHooks,
    ) -> Optional[RepelloAIAnalyzeResponse]:
        """stage ("prompt" or "response") selects both the endpoint path and the
        scan_data key. Returns the parsed response, or None when the API is
        unreachable and unreachable_fallback is fail_open (the caller then allows
        the request through).
        """
        endpoint = f"{self.api_base}/analyze/{stage}"
        request: Dict = {
            "asset_id": self.asset_id or "",
            "scan_data": {stage: text},
        }

        status: GuardrailStatus = "success"
        guardrail_json_response: Union[str, dict, List[dict]] = ""
        start_time: datetime = datetime.now()
        repelloai_response: Optional[RepelloAIAnalyzeResponse] = None
        try:
            verbose_proxy_logger.debug("RepelloAI Argus request: %s", request)
            response = await self.async_handler.post(
                url=endpoint,
                headers={"X-API-Key": self.repelloai_api_key},
                json=request,
            )
            self._raise_for_config_error(response)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError(
                    f"RepelloAI Argus returned a non-object response: {type(payload)}"
                )
            repelloai_response = RepelloAIAnalyzeResponse(**payload)
            verbose_proxy_logger.debug(
                "RepelloAI Argus response: %s", repelloai_response
            )
            if self._verdict_blocks(repelloai_response):
                status = "guardrail_intervened"
            return repelloai_response
        except HTTPException as e:
            # Misconfiguration / fail_closed -> block. Surface, never fail open.
            status = "guardrail_failed_to_respond"
            detail = e.detail
            if isinstance(detail, (dict, list)):
                guardrail_json_response = detail
            else:
                guardrail_json_response = str(detail)
            raise
        except Exception as e:
            status = "guardrail_failed_to_respond"
            guardrail_json_response = str(e)
            return self._handle_unreachable(e)
        finally:
            end_time = datetime.now()
            if repelloai_response is not None:
                guardrail_json_response = dict(repelloai_response)
            self.add_standard_logging_guardrail_information_to_request_data(
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
    def _raise_for_config_error(response) -> None:
        """Surface auth/config failures instead of silently failing open.

        These status codes mean the guardrail itself is misconfigured (bad API
        key, unknown asset_id, malformed payload), not a transient network blip.
        unreachable_fallback must not turn a permanently broken guardrail into a
        silent no-op, so these always block.
        """
        if response.status_code in CONFIG_ERROR_STATUS_CODES:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "RepelloAI Argus guardrail is misconfigured",
                    "status_code": response.status_code,
                },
            )

    def _verdict_blocks(
        self, repelloai_response: Optional[RepelloAIAnalyzeResponse]
    ) -> bool:
        """Return True if the verdict should block the request.

        Blocks on the explicit blocked verdict and on any unrecognized verdict
        (None, empty, or an unexpected value) so an upstream schema change can't
        silently disable enforcement. passed/flagged are allowed.
        """
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

    def _handle_unreachable(
        self, error: Exception
    ) -> Optional[RepelloAIAnalyzeResponse]:
        """Apply the unreachable_fallback policy when the API call fails.

        fail_closed blocks the request; fail_open logs a warning and
        returns None so the caller lets the request through.
        """
        verbose_proxy_logger.warning("RepelloAI Argus unreachable: %s", str(error))
        if self.unreachable_fallback == "fail_closed":
            raise HTTPException(
                status_code=500,
                detail={"error": "RepelloAI Argus guardrail unreachable"},
            )
        return None

    def _raise_if_blocked(
        self, repelloai_response: Optional[RepelloAIAnalyzeResponse]
    ) -> None:
        if repelloai_response is None:
            return
        if self._verdict_blocks(repelloai_response):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Blocked by RepelloAI Argus guardrail",
                    "policies_violated": repelloai_response.get("policies_violated"),
                },
            )
        self._log_flagged_verdict(repelloai_response)

    @staticmethod
    def _log_flagged_verdict(
        repelloai_response: RepelloAIAnalyzeResponse,
    ) -> None:
        if repelloai_response.get("verdict") == FLAGGED_VERDICT:
            verbose_proxy_logger.warning(
                "RepelloAI Argus flagged content (allowed): %s",
                repelloai_response.get("policies_violated"),
            )

    @staticmethod
    def _extract_prompt_message_text(data: Dict) -> List[str]:
        messages = build_inspection_messages(data)
        return [
            message.get("content")
            for message in messages
            if isinstance(message.get("content"), str) and message.get("content")
        ]

    @classmethod
    def _extract_prompt_text(cls, data: Dict) -> Optional[str]:
        texts = cls._extract_prompt_message_text(data)

        raw_messages = data.get("messages")
        if isinstance(raw_messages, list):
            for message in raw_messages:
                texts.extend(cls._extract_tool_call_args_from_message(message))

        raw_input = data.get("input")
        if isinstance(raw_input, list):
            for item in raw_input:
                if isinstance(item, dict) and "role" in item:
                    texts.extend(cls._extract_tool_call_args_from_message(item))

        texts.extend(cls._extract_tool_definition_text(data))
        return "\n".join(text for text in texts if text) if texts else None

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: litellm.DualCache,
        data: Dict,
        call_type: CallTypesLiteral,
    ) -> Optional[Union[Exception, str, Dict]]:
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        verbose_proxy_logger.debug("RepelloAI Argus: pre_call_hook")

        event_type: GuardrailEventHooks = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
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
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        verbose_proxy_logger.debug("RepelloAI Argus: post_call_success_hook")

        event_type: GuardrailEventHooks = GuardrailEventHooks.post_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
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
        response,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        from litellm.main import stream_chunk_builder
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type: GuardrailEventHooks = GuardrailEventHooks.post_call
        if (
            self.should_run_guardrail(data=request_data, event_type=event_type)
            is not True
        ):
            async for chunk in response:
                yield chunk
            return

        chunks: List[ModelResponseStream] = []
        async for chunk in response:
            chunks.append(chunk)

        assembled = stream_chunk_builder(chunks=chunks)
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

        for chunk in chunks:
            yield chunk

    @staticmethod
    def _extract_response_text(response: object) -> Optional[str]:
        """Extract inspectable assistant text from chat or Responses API shapes."""
        if hasattr(response, "output_text"):
            output_text = getattr(response, "output_text")
            if isinstance(output_text, str) and output_text:
                return output_text

        if isinstance(response, dict):
            response_dict = response
        elif hasattr(response, "model_dump"):
            response_dict = response.model_dump()
        else:
            response_dict = {}
        text = RepelloAIGuardrail._extract_chat_completion_text(response_dict)
        if text:
            return text
        return RepelloAIGuardrail._extract_responses_api_text(response_dict)

    @staticmethod
    def _extract_chat_completion_text(response_dict: Dict) -> Optional[str]:
        parts: List[str] = []
        choices = response_dict.get("choices")
        if not isinstance(choices, list):
            return None
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and content:
                parts.append(content)
        return "\n".join(parts) if parts else None

    @staticmethod
    def _extract_responses_api_text(response_dict: Dict) -> Optional[str]:
        texts: List[str] = []
        output = response_dict.get("output")
        if not isinstance(output, list):
            return None
        for output_item in output:
            if not isinstance(output_item, dict):
                continue
            if output_item.get("type") != "message":
                continue
            content = output_item.get("content")
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") not in ("output_text", "text"):
                    continue
                text = content_item.get("text")
                if isinstance(text, str) and text:
                    texts.append(text)
        return "".join(texts) if texts else None

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.repelloai import (
            RepelloAIGuardrailConfigModel,
        )

        return RepelloAIGuardrailConfigModel
