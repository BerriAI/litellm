import json
from typing import TYPE_CHECKING, Any, List, Optional, Union, cast

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import SpendLogsPayload
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


class _ENTERPRISE_ResponsesSessionHandler:
    @staticmethod
    async def get_chat_completion_message_history_for_previous_response_id(
        previous_response_id: str,
    ) -> ChatCompletionSession:
        """
        Return the chat completion message history for a previous response id
        """
        from litellm.responses.litellm_completion_transformation.transformation import (
            ChatCompletionSession,
            LiteLLMCompletionResponsesConfig,
        )

        verbose_proxy_logger.debug(
            "inside get_chat_completion_message_history_for_previous_response_id"
        )
        all_spend_logs: List[
            SpendLogsPayload
        ] = await _ENTERPRISE_ResponsesSessionHandler.get_all_spend_logs_for_previous_response_id(
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
            proxy_server_request: Union[str, dict] = (
                spend_log.get("proxy_server_request") or "{}"
            )
            proxy_server_request_dict: Optional[dict] = None
            response_input_param: Optional[Union[str, ResponseInputParam]] = None
            if isinstance(proxy_server_request, dict):
                proxy_server_request_dict = proxy_server_request
            else:
                proxy_server_request_dict = json.loads(proxy_server_request)

            ############################################################
            # Add Input messages for this Spend Log
            ############################################################
            if proxy_server_request_dict:
                _response_input_param = proxy_server_request_dict.get("input", None)
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
            # Add Output messages for this Spend Log
            ############################################################
            _response_output = spend_log.get("response", "{}")
            if isinstance(_response_output, dict):
                # transform `ChatCompletion Response` to `ResponsesAPIResponse`
                model_response = ModelResponse(**_response_output)
                for choice in model_response.choices:
                    if hasattr(choice, "message"):
                        chat_completion_message_history.append(
                            getattr(choice, "message")
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
