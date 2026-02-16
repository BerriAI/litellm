import json
import time
from typing import Any, List, Literal, Optional, Tuple

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.types.llms.openai import Batch
from litellm.types.utils import CallTypes, ModelInfo, ModelResponse, Usage
from litellm.utils import token_counter


async def calculate_batch_cost_and_usage(
    file_content_dictionary: List[dict],
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "hosted_vllm", "anthropic"],
    model_name: Optional[str] = None,
    model_info: Optional[ModelInfo] = None,
) -> Tuple[float, Usage, List[str]]:
    """
    Calculate the cost and usage of a batch.

    Args:
        model_info: Optional deployment-level model info with custom batch
            pricing. Threaded through to batch_cost_calculator so that
            deployment-specific pricing (e.g. input_cost_per_token_batches)
            is used instead of the global cost map.
    """
    batch_cost = _batch_cost_calculator(
        custom_llm_provider=custom_llm_provider,
        file_content_dictionary=file_content_dictionary,
        model_name=model_name,
        model_info=model_info,
    )
    batch_usage = _get_batch_job_total_usage_from_file_content(
        file_content_dictionary=file_content_dictionary,
        custom_llm_provider=custom_llm_provider,
        model_name=model_name,
    )
    batch_models = _get_batch_models_from_file_content(file_content_dictionary, model_name)

    return batch_cost, batch_usage, batch_models


async def _handle_completed_batch(
    batch: Batch,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "hosted_vllm", "anthropic"],
    model_name: Optional[str] = None,
    litellm_params: Optional[dict] = None,
) -> Tuple[float, Usage, List[str]]:
    """Helper function to process a completed batch and handle logging
    
    Args:
        batch: The batch object
        custom_llm_provider: The LLM provider
        model_name: Optional model name
        litellm_params: Optional litellm parameters containing credentials (api_key, api_base, etc.)
    """
    # Get batch results
    file_content_dictionary = await _get_batch_output_file_content_as_dictionary(
        batch, custom_llm_provider, litellm_params=litellm_params
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

    batch_models = _get_batch_models_from_file_content(file_content_dictionary, model_name)

    return batch_cost, batch_usage, batch_models


def _get_batch_models_from_file_content(
    file_content_dictionary: List[dict],
    model_name: Optional[str] = None,
) -> List[str]:
    """
    Get the models from the file content
    """
    if model_name:
        return [model_name]
    batch_models = []
    for _item in file_content_dictionary:
        if _batch_response_was_successful(_item):
            _response_body = _get_response_from_batch_job_output_file(_item)
            _model = _response_body.get("model")
            if _model:
                batch_models.append(_model)
    return batch_models


def _batch_cost_calculator(
    file_content_dictionary: List[dict],
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "hosted_vllm", "anthropic"] = "openai",
    model_name: Optional[str] = None,
    model_info: Optional[ModelInfo] = None,
) -> float:
    """
    Calculate the cost of a batch based on the output file id
    """
    # Handle Vertex AI with specialized method
    if custom_llm_provider == "vertex_ai" and model_name:
        batch_cost, _ = calculate_vertex_ai_batch_cost_and_usage(file_content_dictionary, model_name)
        verbose_logger.debug("vertex_ai_total_cost=%s", batch_cost)
        return batch_cost
    
    # For other providers, use the existing logic
    total_cost = _get_batch_job_cost_from_file_content(
        file_content_dictionary=file_content_dictionary,
        custom_llm_provider=custom_llm_provider,
        model_info=model_info,
    )
    verbose_logger.debug("total_cost=%s", total_cost)
    return total_cost


def calculate_vertex_ai_batch_cost_and_usage(
    vertex_ai_batch_responses: List[dict],
    model_name: Optional[str] = None,
) -> Tuple[float, Usage]:
    """
    Calculate both cost and usage from Vertex AI batch responses
    """
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )
    total_cost = 0.0
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    
    for response in vertex_ai_batch_responses:
        if response.get("status") == "JOB_STATE_SUCCEEDED":  # Check if response was successful
            # Transform Vertex AI response to OpenAI format if needed

            # Create required arguments for the transformation method
            model_response = ModelResponse()
            
            # Ensure model_name is not None
            actual_model_name = model_name or "gemini-2.5-flash"
            
            # Create a real LiteLLM logging object
            logging_obj = Logging(
                model=actual_model_name,
                messages=[{"role": "user", "content": "batch_request"}],
                stream=False,
                call_type=CallTypes.aretrieve_batch,
                start_time=time.time(),
                litellm_call_id="batch_" + str(uuid.uuid4()),
                function_id="batch_processing",
                litellm_trace_id=str(uuid.uuid4()),
                kwargs={"optional_params": {}}
            )
            
            # Add the optional_params attribute that the Vertex AI transformation expects
            logging_obj.optional_params = {}
            raw_response = httpx.Response(200)  # Mock response object
            
            openai_format_response = VertexGeminiConfig()._transform_google_generate_content_to_openai_model_response(
                completion_response=response["response"],
                model_response=model_response,
                model=actual_model_name,
                logging_obj=logging_obj,
                raw_response=raw_response,
            )
            
            # Calculate cost using existing function
            cost = litellm.completion_cost(
                completion_response=openai_format_response,
                custom_llm_provider="vertex_ai",
                call_type=CallTypes.aretrieve_batch.value,
            )
            total_cost += cost
            
            # Extract usage from the transformed response
            usage_obj = getattr(openai_format_response, 'usage', None)
            if usage_obj:
                usage = usage_obj
            else:
                # Fallback: create usage from response dict
                response_dict = openai_format_response.dict() if hasattr(openai_format_response, 'dict') else {}
                usage = _get_batch_job_usage_from_response_body(response_dict)
            
            total_tokens += usage.total_tokens
            prompt_tokens += usage.prompt_tokens
            completion_tokens += usage.completion_tokens
    
    return total_cost, Usage(
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


async def _get_batch_output_file_content_as_dictionary(
    batch: Batch,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "hosted_vllm", "anthropic"] = "openai",
    litellm_params: Optional[dict] = None,
) -> List[dict]:
    """
    Get the batch output file content as a list of dictionaries
    
    Args:
        batch: The batch object
        custom_llm_provider: The LLM provider
        litellm_params: Optional litellm parameters containing credentials (api_key, api_base, etc.)
                       Required for Azure and other providers that need authentication
    """
    from litellm.files.main import afile_content
    from litellm.proxy.openai_files_endpoints.common_utils import (
        _is_base64_encoded_unified_file_id,
    )

    if custom_llm_provider == "vertex_ai":
        raise ValueError("Vertex AI does not support file content retrieval")

    if batch.output_file_id is None:
        raise ValueError("Output file id is None cannot retrieve file content")

    file_id = batch.output_file_id
    is_base64_unified_file_id = _is_base64_encoded_unified_file_id(file_id)
    if is_base64_unified_file_id:
        try:
            file_id = is_base64_unified_file_id.split("llm_output_file_id,")[1].split(";")[0]
            verbose_logger.debug(f"Extracted LLM output file ID from unified file ID: {file_id}")
        except (IndexError, AttributeError) as e:
            verbose_logger.error(f"Failed to extract LLM output file ID from unified file ID: {batch.output_file_id}, error: {e}")

    # Build kwargs for afile_content with credentials from litellm_params
    file_content_kwargs = {
        "file_id": file_id,
        "custom_llm_provider": custom_llm_provider,
    }
    
    # Extract and add credentials for file access
    credentials = _extract_file_access_credentials(litellm_params)
    file_content_kwargs.update(credentials)
    
    _file_content = await afile_content(**file_content_kwargs)
    return _get_file_content_as_dictionary(_file_content.content)


def _extract_file_access_credentials(litellm_params: Optional[dict]) -> dict:
    """
    Extract credentials from litellm_params for file access operations.
    
    This method extracts relevant authentication and configuration parameters
    needed for accessing files across different providers (Azure, Vertex AI, etc.).
    
    Args:
        litellm_params: Dictionary containing litellm parameters with credentials
        
    Returns:
        Dictionary containing only the credentials needed for file access
    """
    credentials = {}
    
    if litellm_params:
        # List of credential keys that should be passed to file operations
        credential_keys = [
            "api_key", "api_base", "api_version", "organization",
            "azure_ad_token", "azure_ad_token_provider",
            "vertex_project", "vertex_location", "vertex_credentials",
            "timeout", "max_retries"
        ]
        for key in credential_keys:
            if key in litellm_params:
                credentials[key] = litellm_params[key]
    
    return credentials


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
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "hosted_vllm", "anthropic"] = "openai",
    model_info: Optional[ModelInfo] = None,
) -> float:
    """
    Get the cost of a batch job from the file content
    """
    from litellm.cost_calculator import batch_cost_calculator

    try:
        total_cost: float = 0.0
        # parse the file content as json
        verbose_logger.debug(
            "file_content_dictionary=%s", json.dumps(file_content_dictionary, indent=4)
        )
        for _item in file_content_dictionary:
            if _batch_response_was_successful(_item):
                _response_body = _get_response_from_batch_job_output_file(_item)
                if model_info is not None:
                    usage = _get_batch_job_usage_from_response_body(_response_body)
                    model = _response_body.get("model", "")
                    prompt_cost, completion_cost = batch_cost_calculator(
                        usage=usage,
                        model=model,
                        custom_llm_provider=custom_llm_provider,
                        model_info=model_info,
                    )
                    total_cost += prompt_cost + completion_cost
                else:
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
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "hosted_vllm", "anthropic"] = "openai",
    model_name: Optional[str] = None,
) -> Usage:
    """
    Get the tokens of a batch job from the file content
    """
    # Handle Vertex AI with specialized method
    if custom_llm_provider == "vertex_ai" and model_name:
        _, batch_usage = calculate_vertex_ai_batch_cost_and_usage(file_content_dictionary, model_name)
        return batch_usage
    
    # For other providers, use the existing logic
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