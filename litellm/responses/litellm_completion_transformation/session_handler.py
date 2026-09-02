import asyncio
import json
from typing import TYPE_CHECKING, Any, Final, cast

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import REDACTED_BY_LITELLM, REDACTED_TOOL_CALL_ARGUMENTS_PLACEHOLDER
from litellm.proxy._types import SpendLogsMetadata, SpendLogsPayload
from litellm.proxy.spend_tracking.cold_storage_handler import ColdStorageHandler
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionResponseMessage,
    GenericChatCompletionMessage,
    ResponseInputParam,
)
from litellm.types.utils import ChatCompletionMessageToolCall, Message, ModelResponse

if TYPE_CHECKING:
    from litellm.responses.litellm_completion_transformation.transformation import (
        ChatCompletionSession,
    )
else:
    ChatCompletionSession = Any

########################################################
# Cold Storage Handler
########################################################
COLD_STORAGE_HANDLER: Final = ColdStorageHandler()
########################################################


def _normalize_redacted_tool_call_arguments(message: Message) -> None:
    """Redaction stores the bare sentinel (invalid JSON) in tool-call arguments;
    normalize replayed history to "{}" so provider converters can parse it."""
    for tool_call in message.tool_calls or []:
        if (function := getattr(tool_call, "function", None)) is not None and function.arguments == REDACTED_BY_LITELLM:
            function.arguments = REDACTED_TOOL_CALL_ARGUMENTS_PLACEHOLDER
    function_call: Final = message.function_call
    if function_call is not None and function_call.arguments == REDACTED_BY_LITELLM:
        function_call.arguments = REDACTED_TOOL_CALL_ARGUMENTS_PLACEHOLDER


class ResponsesSessionHandler:
    @staticmethod
    async def get_chat_completion_message_history_for_previous_response_id(
        previous_response_id: str,
    ) -> ChatCompletionSession:
        """
        Return the chat completion message history for a previous response id
        """
        from litellm.responses.litellm_completion_transformation.transformation import (
            ChatCompletionSession,
        )

        verbose_proxy_logger.debug("inside get_chat_completion_message_history_for_previous_response_id")
        all_spend_logs: list[
            SpendLogsPayload
        ] = await ResponsesSessionHandler.get_all_spend_logs_for_previous_response_id(previous_response_id)
        verbose_proxy_logger.debug("found %s spend logs for this response id", len(all_spend_logs))

        litellm_session_id: str | None = None
        if len(all_spend_logs) > 0:
            litellm_session_id = all_spend_logs[0].get("session_id")

        chat_completion_message_history: list[
            AllMessageValues
            | GenericChatCompletionMessage
            | ChatCompletionMessageToolCall
            | ChatCompletionResponseMessage
            | Message
        ] = []
        for spend_log in all_spend_logs:
            chat_completion_message_history = (
                await ResponsesSessionHandler.extend_chat_completion_message_with_spend_log_payload(
                    spend_log=spend_log,
                    chat_completion_message_history=chat_completion_message_history,
                )
            )

        verbose_proxy_logger.debug(
            "chat_completion_message_history %s",
            json.dumps(chat_completion_message_history, indent=4, default=str),
        )
        return ChatCompletionSession(
            messages=chat_completion_message_history,
            litellm_session_id=litellm_session_id,
        )

    @staticmethod
    async def extend_chat_completion_message_with_spend_log_payload(
        spend_log: SpendLogsPayload,
        chat_completion_message_history: list[
            AllMessageValues
            | GenericChatCompletionMessage
            | ChatCompletionMessageToolCall
            | ChatCompletionResponseMessage
            | Message
        ],
    ):
        """
        Extend the chat completion message history with the spend log payload
        """
        from litellm.responses.litellm_completion_transformation.transformation import (
            LiteLLMCompletionResponsesConfig,
        )

        proxy_server_request_dict: Final = await ResponsesSessionHandler.get_proxy_server_request_from_spend_log(
            spend_log=spend_log,
        )
        response_input_param: str | ResponseInputParam | None = None
        _messages: str | ResponseInputParam | None = None

        ############################################################
        # Add Input messages for this Spend Log
        ############################################################
        if proxy_server_request_dict:
            _response_input_param: Final = proxy_server_request_dict.get("input", None)
            _messages = proxy_server_request_dict.get("messages", None)
            if isinstance(_response_input_param, (str, list)):
                response_input_param = _response_input_param
            elif isinstance(_response_input_param, dict):
                response_input_param = cast(
                    ResponseInputParam,
                    [_response_input_param],  # mutable-ok: a lone input item still has to arrive as a list
                )

        if response_input_param:
            chat_completion_messages = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
                input=response_input_param,
                responses_api_request=proxy_server_request_dict or {},
                replay_reasoning=True,
            )
            chat_completion_message_history.extend(chat_completion_messages)

        ############################################################
        # Check if `messages` field is present in the proxy server request dict
        ############################################################
        elif _messages:
            # ensure all messages are /chat/completions/messages
            # certain requests can be stored as Responses API format - this ensures they are transformed to /chat/completions/messages
            chat_completion_messages = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
                input=_messages,
                responses_api_request=proxy_server_request_dict or {},
                replay_reasoning=True,
            )
            chat_completion_message_history.extend(chat_completion_messages)

        ############################################################
        # Add Output messages for this Spend Log
        ############################################################
        _response_output: Final = spend_log.get("response", "{}")
        if isinstance(_response_output, dict) and _response_output and _response_output != {}:
            # transform `ChatCompletion Response` to `ResponsesAPIResponse`
            model_response: Final = ModelResponse(**_response_output)
            for choice in model_response.choices:
                if hasattr(choice, "message"):
                    _normalize_redacted_tool_call_arguments(choice.message)
                    chat_completion_message_history.append(choice.message)
        return chat_completion_message_history

    @staticmethod
    async def get_proxy_server_request_from_spend_log(
        spend_log: SpendLogsPayload,
    ) -> dict | None:
        """
        Get the parsed proxy server request from the spend log
        """
        proxy_server_request: Final[str | dict] = spend_log.get("proxy_server_request") or "{}"
        proxy_server_request_dict: dict | None = None
        if isinstance(proxy_server_request, dict):
            proxy_server_request_dict = proxy_server_request
        else:
            proxy_server_request_dict = json.loads(proxy_server_request)

        ############################################################
        # Check if user has setup cold storage for session handling
        ############################################################
        if ResponsesSessionHandler._should_check_cold_storage_for_full_payload(proxy_server_request_dict):
            # Try to get cold storage object key from spend log metadata
            _proxy_server_request_dict: dict | None = None
            cold_storage_object_key = ResponsesSessionHandler._get_cold_storage_object_key_from_spend_log(spend_log)
            if cold_storage_object_key:
                # Use the object key directly from metadata
                _proxy_server_request_dict = (
                    await ResponsesSessionHandler.get_proxy_server_request_from_cold_storage_with_object_key(
                        object_key=cold_storage_object_key,
                    )
                )
            if _proxy_server_request_dict:
                proxy_server_request_dict = _proxy_server_request_dict

        return proxy_server_request_dict

    @staticmethod
    def _get_cold_storage_object_key_from_spend_log(
        spend_log: SpendLogsPayload,
    ) -> str | None:
        """
        Extract the cold storage object key from spend log metadata.

        Args:
            spend_log: The spend log payload containing metadata

        Returns:
            Optional[str]: The cold storage object key if found, None otherwise
        """
        try:
            metadata_str: Final = spend_log.get("metadata", "{}")
            if isinstance(metadata_str, str):
                metadata_dict: Final[SpendLogsMetadata] = json.loads(metadata_str)
                return metadata_dict.get("cold_storage_object_key")
            elif isinstance(metadata_str, dict):
                return metadata_str.get("cold_storage_object_key")
            return None
        except (json.JSONDecodeError, TypeError, AttributeError):
            verbose_proxy_logger.debug("Failed to parse metadata from spend log to extract cold storage object key")
            return None

    @staticmethod
    async def get_proxy_server_request_from_cold_storage_with_object_key(
        object_key: str,
    ) -> dict | None:
        """
        Get the proxy server request from cold storage using the object key directly.

        Args:
            object_key: The S3/GCS object key to retrieve

        Returns:
            Optional[dict]: The proxy server request dict or None if not found
        """
        verbose_proxy_logger.debug("inside get_proxy_server_request_from_cold_storage_with_object_key...")

        proxy_server_request_dict: Final = (
            await COLD_STORAGE_HANDLER.get_proxy_server_request_from_cold_storage_with_object_key(
                object_key=object_key,
            )
        )

        return proxy_server_request_dict

    @staticmethod
    def _should_check_cold_storage_for_full_payload(
        proxy_server_request_dict: dict | None,
    ) -> bool:
        """
        Only check cold storage when both are true
        1. `LITELLM_TRUNCATED_PAYLOAD_FIELD` is in the proxy server request dict
        2. `litellm.cold_storage_custom_logger` is not None
        """
        from litellm.constants import LITELLM_TRUNCATED_PAYLOAD_FIELD

        configured_cold_storage_custom_logger: Final = litellm.cold_storage_custom_logger
        if configured_cold_storage_custom_logger is None:
            return False
        if proxy_server_request_dict is None:
            return True
        if len(proxy_server_request_dict) == 0:
            return True
        if LITELLM_TRUNCATED_PAYLOAD_FIELD in str(proxy_server_request_dict):
            return True
        return False

    @staticmethod
    async def get_all_spend_logs_for_previous_response_id(
        previous_response_id: str,
    ) -> list[SpendLogsPayload]:
        """
        Get all spend logs for a previous response id


        SQL query

        SELECT session_id FROM spend_logs WHERE response_id = previous_response_id, SELECT * FROM spend_logs WHERE session_id = session_id

        A just-finished turn gets a short second chance: the worker that served it may
        still be writing its spend log when the follow-up arrives, and an empty result
        drops the whole conversation instead of erroring. Deployments that write no spend
        logs at all have nothing to wait for, so they keep the single original query.
        """
        from litellm.constants import (
            RESPONSES_SESSION_LOOKUP_MAX_ATTEMPTS,
            RESPONSES_SESSION_LOOKUP_RETRY_INTERVAL,
        )
        from litellm.proxy.proxy_server import disable_spend_logs, prisma_client

        verbose_proxy_logger.debug("decoding response id=%s", previous_response_id)

        decoded_response_id: Final = ResponsesAPIRequestUtils._decode_responses_api_response_id(previous_response_id)
        response_id: Final = decoded_response_id.get("response_id", previous_response_id)
        if prisma_client is None:
            return []

        query: Final = """
            WITH matching_session AS (
                SELECT session_id
                FROM "LiteLLM_SpendLogs"
                WHERE request_id = $1
            )
            SELECT *
            FROM "LiteLLM_SpendLogs"
            WHERE session_id IN (SELECT session_id FROM matching_session)
            ORDER BY "endTime" ASC;
        """

        max_attempts: Final = 1 if disable_spend_logs else RESPONSES_SESSION_LOOKUP_MAX_ATTEMPTS
        for attempt in range(max_attempts):
            if attempt:
                await asyncio.sleep(RESPONSES_SESSION_LOOKUP_RETRY_INTERVAL)
            if spend_logs := await prisma_client.db.query_raw(query, response_id):
                verbose_proxy_logger.debug(
                    "Found the following spend logs for previous response id %s: %s",
                    response_id,
                    json.dumps(spend_logs, indent=4, default=str),
                )
                return spend_logs

        verbose_proxy_logger.debug("Found no spend logs for previous response id %s", response_id)
        return []  # mutable-ok: an empty result the caller only reads
