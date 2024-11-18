"""
Common helpers / utils across al OpenAI endpoints
"""

import json
from typing import Any, Dict, List

import openai


####### Error Handling Utils for OpenAI API #######################
###################################################################
def drop_params_from_unprocessable_entity_error(
    e: openai.UnprocessableEntityError, data: Dict[str, Any]
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
    if e.body is not None and isinstance(e.body, dict) and e.body.get("message"):
        message = e.body.get("message", {})
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
