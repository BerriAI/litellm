"""
Common helpers / utils across al OpenAI endpoints
"""

import json
from typing import Any, Dict, List, Optional, Union

import httpx
import openai

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class OpenAIError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
        headers: Optional[Union[dict, httpx.Headers]] = None,
    ):
        self.status_code = status_code
        self.message = message
        self.headers = headers
        if request:
            self.request = request
        else:
            self.request = httpx.Request(method="POST", url="https://api.openai.com/v1")
        if response:
            self.response = response
        else:
            self.response = httpx.Response(
                status_code=status_code, request=self.request
            )
        super().__init__(
            status_code=status_code,
            message=self.message,
            headers=self.headers,
            request=self.request,
            response=self.response,
        )


####### Error Handling Utils for OpenAI API #######################
###################################################################
def drop_params_from_unprocessable_entity_error(
    e: Union[openai.UnprocessableEntityError, httpx.HTTPStatusError],
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Helper function to read OpenAI UnprocessableEntityError and drop the params that raised an error from the error message.

    Args:
    e (UnprocessableEntityError): The UnprocessableEntityError exception
    data (Dict[str, Any]): The original data dictionary containing all parameters

    Returns:
    Dict[str, Any]: A new dictionary with invalid parameters removed
    """
    invalid_params: List[str] = []
    if isinstance(e, httpx.HTTPStatusError):
        error_json = e.response.json()
        error_message = error_json.get("error", {})
        error_body = error_message
    else:
        error_body = e.body
    if (
        error_body is not None
        and isinstance(error_body, dict)
        and error_body.get("message")
    ):
        message = error_body.get("message", {})
        if isinstance(message, str):
            try:
                message = json.loads(message)
            except json.JSONDecodeError:
                message = {"detail": message}
        detail = message.get("detail")

        if isinstance(detail, List) and len(detail) > 0 and isinstance(detail[0], dict):
            for error_dict in detail:
                if (
                    error_dict.get("loc")
                    and isinstance(error_dict.get("loc"), list)
                    and len(error_dict.get("loc")) == 2
                ):
                    invalid_params.append(error_dict["loc"][1])

    new_data = {k: v for k, v in data.items() if k not in invalid_params}

    return new_data
