import json
from typing import TYPE_CHECKING, Any, List, Optional, Union, cast

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import SpendLogsPayload
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
COLD_STORAGE_HANDLER = ColdStorageHandler()
########################################################

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

        verbose_proxy_logger.debug(
            "inside get_chat_completion_message_history_for_previous_response_id"
        )
        all_spend_logs: List[
            SpendLogsPayload
        ] = await ResponsesSessionHandler.get_all_spend_logs_for_previous_response_id(
            previous_response_id
        )
        verbose_proxy_logger.debug(
            "found %s spend logs for this response id", len(all_spend_logs)
        )

        litellm_session_id: Optional[str] = None
        if len(all_spend_logs) > 0:
            litellm_session_id = all_spend_logs[0].get("session_id")

        chat_completion_message_history: List[
            Union[
                AllMessageValues,
                GenericChatCompletionMessage,
                ChatCompletionMessageToolCall,
                ChatCompletionResponseMessage,
                Message,
            ]
        ] = []
        for spend_log in all_spend_logs:
            chat_completion_message_history = await ResponsesSessionHandler.extend_chat_completion_message_with_spend_log_payload(
                spend_log=spend_log,
                chat_completion_message_history=chat_completion_message_history,
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
        chat_completion_message_history: List[
            Union[
                AllMessageValues,
                GenericChatCompletionMessage,
                ChatCompletionMessageToolCall,
                ChatCompletionResponseMessage,
                Message,
            ]
        ]
    ):
        """
        Extend the chat completion message history with the spend log payload
        """
        from litellm.responses.litellm_completion_transformation.transformation import (
            LiteLLMCompletionResponsesConfig,
        )

        proxy_server_request_dict = await ResponsesSessionHandler.get_proxy_server_request_from_spend_log(
            spend_log=spend_log,
        )
        response_input_param: Optional[Union[str, ResponseInputParam]] = None
        _messages: Optional[Union[str, ResponseInputParam]] = None

        ############################################################
        # Add Input messages for this Spend Log
        ############################################################
        if proxy_server_request_dict:
            _response_input_param = proxy_server_request_dict.get("input", None)
            _messages = proxy_server_request_dict.get("messages", None)
            if isinstance(_response_input_param, str):
                response_input_param = _response_input_param
            elif isinstance(_response_input_param, dict):
                response_input_param = cast(
                    ResponseInputParam, _response_input_param
                )

        if response_input_param:
            chat_completion_messages = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
                input=response_input_param,
                responses_api_request=proxy_server_request_dict or {},
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
            )
            chat_completion_message_history.extend(chat_completion_messages)

        ############################################################
        # Add Output messages for this Spend Log
        ############################################################
        _response_output = spend_log.get("response", "{}")
        if isinstance(_response_output, dict) and _response_output and _response_output != {}:
            # transform `ChatCompletion Response` to `ResponsesAPIResponse`
            model_response = ModelResponse(**_response_output)
            for choice in model_response.choices:
                if hasattr(choice, "message"):
                    chat_completion_message_history.append(
                        getattr(choice, "message")
                    )
        return chat_completion_message_history
    
    @staticmethod
    async def get_proxy_server_request_from_spend_log(
        spend_log: SpendLogsPayload,
    ) -> Optional[dict]:
        """
        Get the parsed proxy server request from the spend log
        """
        proxy_server_request: Union[str, dict] = (
            spend_log.get("proxy_server_request") or "{}"
        )
        proxy_server_request_dict: Optional[dict] = None
        if isinstance(proxy_server_request, dict):
            proxy_server_request_dict = proxy_server_request
        else:
            proxy_server_request_dict = json.loads(proxy_server_request)
        

        ############################################################
        # Check if user has setup cold storage for session handling
        ############################################################
        if ResponsesSessionHandler._should_check_cold_storage_for_full_payload(proxy_server_request_dict):
            # Try to get cold storage object key from spend log metadata
            _proxy_server_request_dict: Optional[dict] = None
            cold_storage_object_key = ResponsesSessionHandler._get_cold_storage_object_key_from_spend_log(spend_log)
            if cold_storage_object_key:
                # Use the object key directly from metadata
                _proxy_server_request_dict = await ResponsesSessionHandler.get_proxy_server_request_from_cold_storage_with_object_key(
                    object_key=cold_storage_object_key,
                )
            if _proxy_server_request_dict:
                proxy_server_request_dict = _proxy_server_request_dict
        
        return proxy_server_request_dict
        
    @staticmethod
    def _get_cold_storage_object_key_from_spend_log(spend_log: SpendLogsPayload) -> Optional[str]:
        """
        Extract the cold storage object key from spend log metadata.
        
        Args:
            spend_log: The spend log payload containing metadata
            
        Returns:
            Optional[str]: The cold storage object key if found, None otherwise
        """
        try:
            metadata_str = spend_log.get("metadata", "{}")
            if isinstance(metadata_str, str):
                metadata_dict = json.loads(metadata_str)
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
    ) -> Optional[dict]:
        """
        Get the proxy server request from cold storage using the object key directly.
        
        Args:
            object_key: The S3/GCS object key to retrieve
            
        Returns:
            Optional[dict]: The proxy server request dict or None if not found
        """
        verbose_proxy_logger.debug("inside get_proxy_server_request_from_cold_storage_with_object_key...")

        proxy_server_request_dict = await COLD_STORAGE_HANDLER.get_proxy_server_request_from_cold_storage_with_object_key(
            object_key=object_key,
        )

        return proxy_server_request_dict

    @staticmethod
    def _should_check_cold_storage_for_full_payload(
        proxy_server_request_dict: Optional[dict],
    ) -> bool:
        """
        Only check cold storage when both are true 
        1. `LITELLM_TRUNCATED_PAYLOAD_FIELD` is in the proxy server request dict
        2. `litellm.cold_storage_custom_logger` is not None
        """
        from litellm.constants import LITELLM_TRUNCATED_PAYLOAD_FIELD
        configured_cold_storage_custom_logger = litellm.cold_storage_custom_logger
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
    ) -> List[SpendLogsPayload]:
        """
        Get all spend logs for a previous response id


        SQL query

        SELECT session_id FROM spend_logs WHERE response_id = previous_response_id, SELECT * FROM spend_logs WHERE session_id = session_id
        """
        from litellm.proxy.proxy_server import prisma_client

        verbose_proxy_logger.debug("decoding response id=%s", previous_response_id)

        decoded_response_id = (
            ResponsesAPIRequestUtils._decode_responses_api_response_id(
                previous_response_id
            )
        )
        previous_response_id = decoded_response_id.get(
            "response_id", previous_response_id
        )
        if prisma_client is None:
            return []

        query = """
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

        spend_logs = await prisma_client.db.query_raw(query, previous_response_id)

        verbose_proxy_logger.debug(
            "Found the following spend logs for previous response id %s: %s",
            previous_response_id,
            json.dumps(spend_logs, indent=4, default=str),
        )

        return spend_logs
