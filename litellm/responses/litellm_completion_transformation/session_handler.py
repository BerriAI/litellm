"""
Responses API has previous_response_id, which is the id of the previous response.

LiteLLM needs to maintain a cache of the previous response input, output, previous_response_id, and model.

This class handles that cache.
"""

from typing import List, Optional, Tuple, Union

from typing_extensions import TypedDict

from litellm.caching import InMemoryCache
from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIResponse

RESPONSES_API_PREVIOUS_RESPONSES_CACHE = InMemoryCache()
MAX_PREV_SESSION_INPUTS = 50


class ResponsesAPISessionElement(TypedDict, total=False):
    input: Union[str, ResponseInputParam]
    output: ResponsesAPIResponse
    response_id: str
    previous_response_id: Optional[str]


class SessionHandler:

    def add_completed_response_to_cache(
        self, response_id: str, session_element: ResponsesAPISessionElement
    ):
        RESPONSES_API_PREVIOUS_RESPONSES_CACHE.set_cache(
            key=response_id, value=session_element
        )

    def get_chain_of_previous_input_output_pairs(
        self, previous_response_id: str
    ) -> List[Tuple[ResponseInputParam, ResponsesAPIResponse]]:
        response_api_inputs: List[Tuple[ResponseInputParam, ResponsesAPIResponse]] = []
        current_previous_response_id = previous_response_id

        count_session_elements = 0
        while current_previous_response_id:
            if count_session_elements > MAX_PREV_SESSION_INPUTS:
                break
            session_element = RESPONSES_API_PREVIOUS_RESPONSES_CACHE.get_cache(
                key=current_previous_response_id
            )
            if session_element:
                response_api_inputs.append(
                    (session_element.get("input"), session_element.get("output"))
                )
                current_previous_response_id = session_element.get(
                    "previous_response_id"
                )
            else:
                break
            count_session_elements += 1
        return response_api_inputs
