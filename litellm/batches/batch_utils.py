import json
from typing import Any, List, Literal, Tuple

import litellm
from litellm._logging import verbose_logger
from litellm.types.llms.openai import Batch
from litellm.types.utils import CallTypes, Usage


async def _handle_completed_batch(
    batch: Batch,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai"],
) -> Tuple[float, Usage, List[str]]:
    """Helper function to process a completed batch and handle logging"""
    # Get batch results
    file_content_dictionary = await _get_batch_output_file_content_as_dictionary(
        batch, custom_llm_provider
    )

    # Calculate costs and usage
    batch_cost = await _batch_cost_calculator(
        custom_llm_provider=custom_llm_provider,
        file_content_dictionary=file_content_dictionary,
    )
    batch_usage = _get_batch_job_total_usage_from_file_content(
        file_content_dictionary=file_content_dictionary,
        custom_llm_provider=custom_llm_provider,
    )

    batch_models = _get_batch_models_from_file_content(file_content_dictionary)

    return batch_cost, batch_usage, batch_models


def _get_batch_models_from_file_content(
    file_content_dictionary: List[dict],
) -> List[str]:
    """
    Get the models from the file content
    """
    batch_models = []
    for _item in file_content_dictionary:
        if _batch_response_was_successful(_item):
            _response_body = _get_response_from_batch_job_output_file(_item)
            _model = _response_body.get("model")
            if _model:
                batch_models.append(_model)
    return batch_models


async def _batch_cost_calculator(
    file_content_dictionary: List[dict],
    custom_llm_provider: Literal["openai", "azure", "vertex_ai"] = "openai",
) -> float:
    """
    Calculate the cost of a batch based on the output file id
    """
    if custom_llm_provider == "vertex_ai":
        raise ValueError("Vertex AI does not support file content retrieval")
    total_cost = _get_batch_job_cost_from_file_content(
        file_content_dictionary=file_content_dictionary,
        custom_llm_provider=custom_llm_provider,
    )
    verbose_logger.debug("total_cost=%s", total_cost)
    return total_cost


async def _get_batch_output_file_content_as_dictionary(
    batch: Batch,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai"] = "openai",
) -> List[dict]:
    """
    Get the batch output file content as a list of dictionaries
    """
    from litellm.files.main import afile_content

    if custom_llm_provider == "vertex_ai":
        raise ValueError("Vertex AI does not support file content retrieval")

    if batch.output_file_id is None:
        raise ValueError("Output file id is None cannot retrieve file content")

    _file_content = await afile_content(
        file_id=batch.output_file_id,
        custom_llm_provider=custom_llm_provider,
    )
    return _get_file_content_as_dictionary(_file_content.content)


def _get_file_content_as_dictionary(file_content: bytes) -> List[dict]:
    """
    Get the file content as a list of dictionaries from JSON Lines format
    """
    try:
        _file_content_str = file_content.decode("utf-8")
        # Split by newlines and parse each line as a separate JSON object
        json_objects = []
        for line in _file_content_str.strip().split("\n"):
            if line:  # Skip empty lines
                json_objects.append(json.loads(line))
        verbose_logger.debug("json_objects=%s", json.dumps(json_objects, indent=4))
        return json_objects
    except Exception as e:
        raise e


def _get_batch_job_cost_from_file_content(
    file_content_dictionary: List[dict],
    custom_llm_provider: Literal["openai", "azure", "vertex_ai"] = "openai",
) -> float:
    """
    Get the cost of a batch job from the file content
    """
    try:
        total_cost: float = 0.0
        # parse the file content as json
        verbose_logger.debug(
            "file_content_dictionary=%s", json.dumps(file_content_dictionary, indent=4)
        )
        for _item in file_content_dictionary:
            if _batch_response_was_successful(_item):
                _response_body = _get_response_from_batch_job_output_file(_item)
                total_cost += litellm.completion_cost(
                    completion_response=_response_body,
                    custom_llm_provider=custom_llm_provider,
                    call_type=CallTypes.aretrieve_batch.value,
                )
                verbose_logger.debug("total_cost=%s", total_cost)
        return total_cost
    except Exception as e:
        verbose_logger.error("error in _get_batch_job_cost_from_file_content", e)
        raise e


def _get_batch_job_total_usage_from_file_content(
    file_content_dictionary: List[dict],
    custom_llm_provider: Literal["openai", "azure", "vertex_ai"] = "openai",
) -> Usage:
    """
    Get the tokens of a batch job from the file content
    """
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    for _item in file_content_dictionary:
        if _batch_response_was_successful(_item):
            _response_body = _get_response_from_batch_job_output_file(_item)
            usage: Usage = _get_batch_job_usage_from_response_body(_response_body)
            total_tokens += usage.total_tokens
            prompt_tokens += usage.prompt_tokens
            completion_tokens += usage.completion_tokens
    return Usage(
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _get_batch_job_usage_from_response_body(response_body: dict) -> Usage:
    """
    Get the tokens of a batch job from the response body
    """
    _usage_dict = response_body.get("usage", None) or {}
    usage: Usage = Usage(**_usage_dict)
    return usage


def _get_response_from_batch_job_output_file(batch_job_output_file: dict) -> Any:
    """
    Get the response from the batch job output file
    """
    _response: dict = batch_job_output_file.get("response", None) or {}
    _response_body = _response.get("body", None) or {}
    return _response_body


def _batch_response_was_successful(batch_job_output_file: dict) -> bool:
    """
    Check if the batch job response status == 200
    """
    _response: dict = batch_job_output_file.get("response", None) or {}
    return _response.get("status_code", None) == 200
