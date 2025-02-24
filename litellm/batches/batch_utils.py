import asyncio
import datetime
import json
import threading
from typing import Any, List, Literal, Optional

import litellm
from litellm._logging import verbose_logger
from litellm.constants import (
    BATCH_STATUS_POLL_INTERVAL_SECONDS,
    BATCH_STATUS_POLL_MAX_ATTEMPTS,
)
from litellm.files.main import afile_content
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.llms.openai import Batch
from litellm.types.utils import StandardLoggingPayload, Usage


async def batches_async_logging(
    batch_id: str,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai"] = "openai",
    logging_obj: Optional[LiteLLMLoggingObj] = None,
    **kwargs,
):
    """
    Async Job waits for the batch to complete and then logs the completed batch usage - cost, total tokens, prompt tokens, completion tokens


    Polls retrieve_batch until it returns a batch with status "completed" or "failed"
    """
    from .main import aretrieve_batch

    verbose_logger.debug(
        ".....in _batches_async_logging... polling retrieve to get batch status"
    )
    if logging_obj is None:
        raise ValueError(
            "logging_obj is None cannot calculate cost / log batch creation event"
        )
    for _ in range(BATCH_STATUS_POLL_MAX_ATTEMPTS):
        try:
            start_time = datetime.datetime.now()
            batch: Batch = await aretrieve_batch(batch_id, custom_llm_provider)
            verbose_logger.debug(
                "in _batches_async_logging... batch status= %s", batch.status
            )

            if batch.status == "completed":
                end_time = datetime.datetime.now()
                await _handle_completed_batch(
                    batch=batch,
                    custom_llm_provider=custom_llm_provider,
                    logging_obj=logging_obj,
                    start_time=start_time,
                    end_time=end_time,
                    **kwargs,
                )
                break
            elif batch.status == "failed":
                pass
        except Exception as e:
            verbose_logger.error("error in batches_async_logging", e)
        await asyncio.sleep(BATCH_STATUS_POLL_INTERVAL_SECONDS)


async def _handle_completed_batch(
    batch: Batch,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai"],
    logging_obj: LiteLLMLoggingObj,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    **kwargs,
) -> None:
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

    # Handle logging
    await _log_completed_batch(
        logging_obj=logging_obj,
        batch_usage=batch_usage,
        batch_cost=batch_cost,
        start_time=start_time,
        end_time=end_time,
        **kwargs,
    )


async def _log_completed_batch(
    logging_obj: LiteLLMLoggingObj,
    batch_usage: Usage,
    batch_cost: float,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    **kwargs,
) -> None:
    """Helper function to handle all logging operations for a completed batch"""
    logging_obj.call_type = "batch_success"

    standard_logging_object = _create_standard_logging_object_for_completed_batch(
        kwargs=kwargs,
        start_time=start_time,
        end_time=end_time,
        logging_obj=logging_obj,
        batch_usage_object=batch_usage,
        response_cost=batch_cost,
    )

    logging_obj.model_call_details["standard_logging_object"] = standard_logging_object

    # Launch async and sync logging handlers
    asyncio.create_task(
        logging_obj.async_success_handler(
            result=None,
            start_time=start_time,
            end_time=end_time,
            cache_hit=None,
        )
    )
    threading.Thread(
        target=logging_obj.success_handler,
        args=(None, start_time, end_time),
    ).start()


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


def _create_standard_logging_object_for_completed_batch(
    kwargs: dict,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    logging_obj: LiteLLMLoggingObj,
    batch_usage_object: Usage,
    response_cost: float,
) -> StandardLoggingPayload:
    """
    Create a standard logging object for a completed batch
    """
    standard_logging_object = logging_obj.model_call_details.get(
        "standard_logging_object", None
    )

    if standard_logging_object is None:
        raise ValueError("unable to create standard logging object for completed batch")

    # Add Completed Batch Job Usage and Response Cost
    standard_logging_object["call_type"] = "batch_success"
    standard_logging_object["response_cost"] = response_cost
    standard_logging_object["total_tokens"] = batch_usage_object.total_tokens
    standard_logging_object["prompt_tokens"] = batch_usage_object.prompt_tokens
    standard_logging_object["completion_tokens"] = batch_usage_object.completion_tokens
    return standard_logging_object
