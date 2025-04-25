from litellm.proxy._types import SpendLogsPayload
from litellm.integrations.custom_logger import CustomLogger
from litellm._logging import verbose_proxy_logger
from typing import Optional, List, Tuple
import json
from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIResponse

class _ENTERPRISE_ResponsesSessionHandler(CustomLogger):
    def __init__(self):
        pass

    @staticmethod
    async def get_chain_of_previous_input_output_pairs(
        previous_response_id: str,
    ) -> List[Tuple[ResponseInputParam, ResponsesAPIResponse]]:
        """
        Get all input and output from the spend logs
        """
        all_spend_logs: List[SpendLogsPayload] = await _ENTERPRISE_ResponsesSessionHandler.get_all_spend_logs_for_previous_response_id(previous_response_id)
        responses_api_session_elements: List[Tuple[ResponseInputParam, ResponsesAPIResponse]] = []
        for spend_log in all_spend_logs:
            proxy_server_request: str = spend_log.get("proxy_server_request") or "{}"
            proxy_server_request_dict: dict = json.loads(proxy_server_request)

            _response_input_param = proxy_server_request_dict.get("input", None)
            response_input_param: Optional[ResponseInputParam] = None
            if _response_input_param is str:
                response_input_param = _response_input_param
            elif isinstance(_response_input_param, dict):
                response_input_param = ResponseInputParam(**_response_input_param)
            else:
                continue

            _response_output = spend_log.get("response", "{}")
            if isinstance(_response_output, dict):
                response_output = ResponsesAPIResponse(**_response_output)
            else:
                continue
            
            responses_api_session_elements.append((response_input_param, response_output))

                

        return responses_api_session_elements

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
        if prisma_client is None:
            return []

        query = """
            WITH matching_session AS (
                SELECT session_id
                FROM "LiteLLM_SpendLogs"
                WHERE response_id = $1
            ),

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
            spend_logs
        )


        return spend_logs
        

    
    
    
