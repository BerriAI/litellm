import json
import time
from typing import Any, List, Literal, Optional, Tuple

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.types.llms.openai import Batch
from litellm.types.utils import CallTypes, ModelResponse, Usage
from litellm.utils import token_counter, ProviderConfigManager


async def calculate_batch_cost_and_usage(
    file_content_dictionary: List[dict],
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "hosted_vllm", "anthropic"],
    model_name: Optional[str] = None,
) -> Tuple[float, Usage, List[str]]:
    """
    Calculate the cost and usage of a batch
    """
    batch_cost = _batch_cost_calculator(
        custom_llm_provider=custom_llm_provider,
        file_content_dictionary=file_content_dictionary,
        model_name=model_name,
    )
    batch_usage = _get_batch_job_total_usage_from_file_content(
        file_content_dictionary=file_content_dictionary,
        custom_llm_provider=custom_llm_provider,
        model_name=model_name,
    )
    batch_models = _get_batch_models_from_file_content(
        file_content_dictionary, model_name, custom_llm_provider=custom_llm_provider
    )

    return batch_cost, batch_usage, batch_models


async def _handle_completed_batch(
    batch: Batch,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "hosted_vllm", "anthropic"],
    model_name: Optional[str] = None,
) -> Tuple[float, Usage, List[str]]:
    """Helper function to process a completed batch and handle logging"""
    # Get batch results
    file_content_dictionary = await _get_batch_output_file_content_as_dictionary(
        batch, custom_llm_provider
    )

    # Calculate costs and usage
    batch_cost = _batch_cost_calculator(
        custom_llm_provider=custom_llm_provider,
        file_content_dictionary=file_content_dictionary,
        model_name=model_name,
    )
    batch_usage = _get_batch_job_total_usage_from_file_content(
        file_content_dictionary=file_content_dictionary,
        custom_llm_provider=custom_llm_provider,
        model_name=model_name,
    )

    batch_models = _get_batch_models_from_file_content(
        file_content_dictionary, model_name, custom_llm_provider=custom_llm_provider
    )

    return batch_cost, batch_usage, batch_models


def transform_raw_provider_response_to_openai(
    raw_response: dict,
    custom_llm_provider: str,
    model: Optional[str] = None,
    messages: Optional[list] = None,
) -> ModelResponse:
    """
    Unified method to transform any raw LLM provider response to OpenAI format.
    
    Args:
        raw_response: Raw response dictionary from any provider (Anthropic, OpenAI, etc.)
        custom_llm_provider: Provider name (e.g., "anthropic", "openai", "vertex_ai")
        model: Model name (optional, will try to extract from raw_response if not provided)
        messages: Original messages list (optional, defaults to empty list)
    
    Returns:
        ModelResponse: OpenAI-compatible response object
    """
    # Lazy import to avoid circular dependency
    from litellm.litellm_core_utils.litellm_logging import Logging
    
    # Extract model from response if not provided
    if model is None:
        model = raw_response.get("model", "unknown-model")
    
    # Default messages if not provided
    if messages is None:
        messages = []
    
    # Get provider config using ProviderConfigManager
    provider_config = ProviderConfigManager.get_provider_chat_config(
        model=model,
        provider=litellm.LlmProviders(custom_llm_provider)
    )
    
    if provider_config is None:
        raise ValueError(f"Could not get config for provider: {custom_llm_provider}")
    
    # Create a mock httpx.Response from the dict
    response_text = json.dumps(raw_response)
    mock_httpx_response = httpx.Response(
        status_code=200,
        content=response_text.encode('utf-8'),
        headers={"content-type": "application/json"}
    )
    
    # Create a minimal logging object
    logging_obj = Logging(
        model=model,
        messages=messages,
        stream=False,
        call_type=CallTypes.completion.value,
        start_time=time.time(),
        litellm_call_id=None,
        function_id=None,
    )
    
    # Create empty ModelResponse to be populated
    model_response = ModelResponse()
    
    # Call transform_response on the provider config
    transformed_response = provider_config.transform_response(
        model=model,
        raw_response=mock_httpx_response,
        model_response=model_response,
        logging_obj=logging_obj,
        request_data={},
        messages=messages,
        optional_params={},
        litellm_params={},
        encoding=litellm.encoding,
        api_key=None,
        json_mode=None,
    )
    
    return transformed_response


def _extract_raw_response_from_batch_item(
    batch_item: dict,
    custom_llm_provider: str,
) -> Optional[dict]:
    """
    Extract the raw provider response from a batch output file item.
    
    Handles different batch output formats:
    - Anthropic: {"result": {"type": "succeeded", "message": {...}}}
    - Vertex AI: {"status": "JOB_STATE_SUCCEEDED", "response": {...}}
    - OpenAI/Azure: {"response": {"status_code": 200, "body": {...}}}
    
    Args:
        batch_item: A single item from the batch output file
        custom_llm_provider: Provider name (e.g., "anthropic", "openai", "vertex_ai")
    
    Returns:
        Raw response dict or None if not successful
    """
    # Anthropic format: {"result": {"type": "succeeded", "message": {...}}}
    if custom_llm_provider == "anthropic":
        result = batch_item.get("result", {})
        if result.get("type") == "succeeded":
            return result.get("message", {})
        return None
    
    # Vertex AI format: {"status": "JOB_STATE_SUCCEEDED", "response": {...}}
    if custom_llm_provider == "vertex_ai":
        if batch_item.get("status") == "JOB_STATE_SUCCEEDED":
            return batch_item.get("response", {})
        return None
    
    # OpenAI/Azure format: {"response": {"status_code": 200, "body": {...}}}
    # Default to OpenAI format for openai, azure, hosted_vllm, etc.
    response = batch_item.get("response", {})
    if response.get("status_code") == 200:
        return response.get("body", {})
    
    return None


def _get_batch_models_from_file_content(
    file_content_dictionary: List[dict],
    model_name: Optional[str] = None,
    custom_llm_provider: str = "openai",
) -> List[str]:
    """
    Get the models from the file content
    """
    if model_name:
        return [model_name]
    batch_models = []
    for _item in file_content_dictionary:
        if _batch_response_was_successful(_item, custom_llm_provider=custom_llm_provider):
            # Extract raw response using generalized method
            raw_response = _extract_raw_response_from_batch_item(
                batch_item=_item,
                custom_llm_provider=custom_llm_provider,
            )
            if raw_response:
                _model = raw_response.get("model")
                if _model:
                    batch_models.append(_model)
    return batch_models


def _batch_cost_calculator(
    file_content_dictionary: List[dict],
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "hosted_vllm", "anthropic"] = "openai",
    model_name: Optional[str] = None,
) -> float:
    """
    Calculate the cost of a batch based on the output file id
    """
    total_cost: float = 0.0
    
    for batch_item in file_content_dictionary:
        if not _batch_response_was_successful(batch_item, custom_llm_provider=custom_llm_provider):
            continue
        
        # Extract raw response from batch item
        raw_response = _extract_raw_response_from_batch_item(
            batch_item=batch_item,
            custom_llm_provider=custom_llm_provider,
        )
        
        if raw_response is None:
            continue
        
        # Extract model from response if not provided
        actual_model = model_name or raw_response.get("model")
        if actual_model is None:
            verbose_logger.warning("Could not determine model for batch item, skipping cost calculation")
            continue
        
        try:
            # Transform to OpenAI format using generalized method
            openai_format_response = transform_raw_provider_response_to_openai(
                raw_response=raw_response,
                custom_llm_provider=custom_llm_provider,
                model=actual_model,
                messages=[],  # Messages not needed for cost calculation
            )
            
            # Calculate cost using standard OpenAI cost calculation
            cost = litellm.completion_cost(
                completion_response=openai_format_response,
                custom_llm_provider=custom_llm_provider,
                call_type=CallTypes.aretrieve_batch.value,
            )
            total_cost += cost
            verbose_logger.debug("item_cost=%s, total_cost=%s", cost, total_cost)
        except Exception as e:
            verbose_logger.warning(
                f"Error calculating cost for batch item: {e}. Skipping this item."
            )
            continue
    
    verbose_logger.debug("total_cost=%s", total_cost)
    return total_cost



async def _get_batch_output_file_content_as_dictionary(
    batch: Batch,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "hosted_vllm", "anthropic"] = "openai",
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


def _get_batch_job_total_usage_from_file_content(
    file_content_dictionary: List[dict],
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "hosted_vllm", "anthropic"] = "openai",
    model_name: Optional[str] = None,
) -> Usage:
    """
    Get the tokens of a batch job from the file content
    """
    from litellm.cost_calculator import BaseTokenUsageProcessor
    
    all_usage: List[Usage] = []
    
    for batch_item in file_content_dictionary:
        if not _batch_response_was_successful(batch_item, custom_llm_provider=custom_llm_provider):
            continue
        
        # Extract raw response from batch item
        raw_response = _extract_raw_response_from_batch_item(
            batch_item=batch_item,
            custom_llm_provider=custom_llm_provider,
        )
        
        if raw_response is None:
            continue
        
        # Extract model from response if not provided
        actual_model = model_name or raw_response.get("model")
        if actual_model is None:
            verbose_logger.warning("Could not determine model for batch item, skipping usage calculation")
            continue
        
        try:
            # Transform to OpenAI format using generalized method
            openai_format_response = transform_raw_provider_response_to_openai(
                raw_response=raw_response,
                custom_llm_provider=custom_llm_provider,
                model=actual_model,
                messages=[],  # Messages not needed for usage extraction
            )
            
            # Extract usage from transformed response
            usage_obj = getattr(openai_format_response, 'usage', None)
            if usage_obj:
                all_usage.append(usage_obj)
            else:
                # Fallback: try to extract from response dict
                response_dict = openai_format_response.dict() if hasattr(openai_format_response, 'dict') else {}
                usage = _get_batch_job_usage_from_response_body(response_dict)
                if usage and usage.total_tokens > 0:
                    all_usage.append(usage)
        except Exception as e:
            verbose_logger.warning(
                f"Error extracting usage for batch item: {e}. Skipping this item."
            )
            continue
    
    # Combine all usage objects
    if all_usage:
        combined_usage = BaseTokenUsageProcessor.combine_usage_objects(all_usage)
        return combined_usage
    
    # Return empty usage if no valid responses
    return Usage(
        total_tokens=0,
        prompt_tokens=0,
        completion_tokens=0,
    )

def _get_batch_job_input_file_usage(
    file_content_dictionary: List[dict],
    custom_llm_provider: Literal["openai", "azure", "vertex_ai"] = "openai",
    model_name: Optional[str] = None,
) -> Usage:
    """
    Count the number of tokens in the input file

    Used for batch rate limiting to count the number of tokens in the input file
    """    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    for _item in file_content_dictionary:
        body = _item.get("body", {})
        model = body.get("model", model_name or "")
        messages = body.get("messages", [])
        
        if messages:
            item_tokens = token_counter(model=model, messages=messages)
            prompt_tokens += item_tokens
        
    return Usage(
        total_tokens=prompt_tokens + completion_tokens,
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


def _batch_response_was_successful(
    batch_job_output_file: dict,
    custom_llm_provider: str = "openai",
) -> bool:
    """
    Check if the batch job response was successful.
    
    Args:
        batch_job_output_file: A single item from the batch output file
        custom_llm_provider: Provider name (e.g., "anthropic", "openai", "vertex_ai")
    
    Returns:
        True if the batch response was successful, False otherwise
    """
    # Anthropic format: {"result": {"type": "succeeded", "message": {...}}}
    if custom_llm_provider == "anthropic":
        result = batch_job_output_file.get("result", {})
        return result.get("type") == "succeeded"
    
    # Vertex AI format: {"status": "JOB_STATE_SUCCEEDED", "response": {...}}
    if custom_llm_provider == "vertex_ai":
        return batch_job_output_file.get("status") == "JOB_STATE_SUCCEEDED"
    
    # OpenAI/Azure format: {"response": {"status_code": 200, "body": {...}}}
    # Default to OpenAI format for openai, azure, hosted_vllm, etc.
    _response: dict = batch_job_output_file.get("response", None) or {}
    return _response.get("status_code", None) == 200
