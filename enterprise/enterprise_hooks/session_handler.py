from litellm.proxy._types import SpendLogsPayload
from litellm.integrations.custom_logger import CustomLogger
from litellm._logging import verbose_proxy_logger
from typing import Optional, List, Tuple, Union
import json
from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIResponse
from litellm.responses.utils import ResponsesAPIRequestUtils, DecodedResponseId
class _ENTERPRISE_ResponsesSessionHandler(CustomLogger):
    def __init__(self):
        pass

    @staticmethod
    async def get_chain_of_previous_input_output_pairs(
        previous_response_id: str,
    ) -> Tuple[List[Tuple[Union[str, ResponseInputParam], ResponsesAPIResponse]], Optional[str]]:
        """
        Get all input and output from the spend logs
        """
        from litellm.responses.litellm_completion_transformation.transformation import LiteLLMCompletionResponsesConfig
        all_spend_logs: List[SpendLogsPayload] = await _ENTERPRISE_ResponsesSessionHandler.get_all_spend_logs_for_previous_response_id(previous_response_id)
        
        litellm_session_id: Optional[str] = None
        if len(all_spend_logs) > 0:
            litellm_session_id = all_spend_logs[0].get("session_id")

        responses_api_session_elements: List[Tuple[ResponseInputParam, ResponsesAPIResponse]] = []
        for spend_log in all_spend_logs:
            proxy_server_request: Union[str, dict] = spend_log.get("proxy_server_request") or "{}"
            proxy_server_request_dict: Optional[dict] = None
            response_input_param: Optional[Union[str, ResponseInputParam]] = None
            response_output: Optional[ResponsesAPIResponse] = None
            if isinstance(proxy_server_request, dict):
                proxy_server_request_dict = proxy_server_request
            else:
                proxy_server_request_dict = json.loads(proxy_server_request)
            # Get Response Input Param from `proxy_server_request`
            if proxy_server_request_dict:
                _response_input_param = proxy_server_request_dict.get("input", None)
                if isinstance(_response_input_param, str):
                    response_input_param = _response_input_param
                elif isinstance(_response_input_param, dict):
                    response_input_param = ResponseInputParam(**_response_input_param)


            # Get Response Output from `spend_log`
            _response_output = spend_log.get("response", "{}")
            if isinstance(_response_output, dict):
                # transform `ChatCompletion Response` to `ResponsesAPIResponse`
                response_output = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                    request_input=response_input_param,
                    responses_api_request={},
                    chat_completion_response=_response_output
                )        
            responses_api_session_elements.append((response_input_param, response_output))
        return responses_api_session_elements, litellm_session_id

    @staticmethod
    async def get_all_spend_logs_for_previous_response_id(
        previous_response_id: str
    ) -> List[SpendLogsPayload]:
        """
        Get all spend logs for a previous response id


        SQL query

        SELECT session_id FROM spend_logs WHERE response_id = previous_response_id, SELECT * FROM spend_logs WHERE session_id = session_id
        """
        from litellm.proxy.proxy_server import prisma_client
        decoded_response_id = ResponsesAPIRequestUtils._decode_responses_api_response_id(previous_response_id)
        previous_response_id = decoded_response_id.get("response_id", previous_response_id)
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
            ORDER BY "endTime" DESC;
        """

        spend_logs = await prisma_client.db.query_raw(
            query,
            previous_response_id
        )

        verbose_proxy_logger.debug(
            "Found the following spend logs for previous response id %s: %s",
            previous_response_id,
            json.dumps(spend_logs, indent=4, default=str)
        )


        return spend_logs
        

    
    
    
