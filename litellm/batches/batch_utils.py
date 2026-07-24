import json
import logging
from dataclasses import dataclass
from functools import reduce
from typing import Any, Iterable, Iterator, List, Literal, Optional, Tuple

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.llm_cost_calc.utils import _parse_prompt_tokens_details
from litellm.types.llms.openai import Batch
from litellm.types.utils import CallTypes, ModelInfo, Usage
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
    batch_models = _get_batch_models_from_file_content(file_content_dictionary, model_name, custom_llm_provider)

    return batch_cost, batch_usage, batch_models


@dataclass(frozen=True, slots=True)
class _BatchTotals:
    cost: float = 0.0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    models: tuple[str, ...] = ()


def _line_cost(
    response_body: dict,
    usage: Usage,
    custom_llm_provider: str,
    model_name: str | None,
    model_info: ModelInfo | None,
) -> float:
    """Cost of a single successful batch output line, mirroring the per-item
    branch in ``_get_batch_job_cost_from_file_content``."""
    from litellm.cost_calculator import batch_cost_calculator

    if model_info is not None or custom_llm_provider in ("anthropic", "bedrock"):
        if custom_llm_provider == "bedrock" and model_name:
            model = model_name
        else:
            model = response_body.get("model") or model_name or ""
        prompt_cost, completion_cost = batch_cost_calculator(
            usage=usage,
            model=model,
            custom_llm_provider=custom_llm_provider,
            model_info=model_info,
        )
        return prompt_cost + completion_cost
    return litellm.completion_cost(
        completion_response=response_body,
        custom_llm_provider=custom_llm_provider,
        call_type=CallTypes.aretrieve_batch.value,
    )


def _fold_batch_totals(
    totals: _BatchTotals,
    item: dict,
    custom_llm_provider: str,
    model_name: str | None,
    model_info: ModelInfo | None,
) -> _BatchTotals:
    """Accumulate one output line into the running totals. Unsuccessful lines are
    skipped, matching the list-based cost/usage/model helpers."""
    if not _batch_response_was_successful(item, custom_llm_provider):
        return totals
    response_body = _get_response_from_batch_job_output_file(item, custom_llm_provider)
    usage = _get_batch_job_usage_from_response_body(response_body, custom_llm_provider)
    prompt_details = _parse_prompt_tokens_details(usage)
    line_model = response_body.get("model")
    return _BatchTotals(
        cost=totals.cost + _line_cost(response_body, usage, custom_llm_provider, model_name, model_info),
        total_tokens=totals.total_tokens + usage.total_tokens,
        prompt_tokens=totals.prompt_tokens + usage.prompt_tokens,
        completion_tokens=totals.completion_tokens + usage.completion_tokens,
        cache_read_tokens=totals.cache_read_tokens + prompt_details["cache_hit_tokens"],
        cache_creation_tokens=totals.cache_creation_tokens + prompt_details["cache_creation_tokens"],
        models=totals.models + ((line_model,) if (line_model and not model_name) else ()),
    )


def _stream_vertex_ai_cost_and_usage(file_content: bytes, model_name: str) -> tuple[float, Usage, list[str]]:
    cost, usage = calculate_vertex_ai_batch_cost_and_usage(_iter_jsonl_entries(file_content), model_name)
    return cost, usage, [model_name]


async def calculate_batch_cost_and_usage_from_content(
    file_content: bytes,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "hosted_vllm", "anthropic"],
    model_name: str | None = None,
    model_info: ModelInfo | None = None,
) -> tuple[float, Usage, list[str]]:
    """Single-pass, memory-bounded equivalent of ``calculate_batch_cost_and_usage``.

    The list-based path parses the whole JSONL output into a ``List[dict]`` and
    then iterates it three times. For image-generation batches each row embeds a
    large base64 payload, so that list can transiently hold hundreds of MB to low
    GB on top of the raw bytes. This parses one row at a time and folds it into
    running totals, so peak memory stays bounded to the raw bytes plus a single row.
    """
    if custom_llm_provider == "vertex_ai" and model_name and litellm.disable_vertex_batch_output_transformation:
        return _stream_vertex_ai_cost_and_usage(file_content, model_name)

    totals = reduce(
        lambda acc, item: _fold_batch_totals(acc, item, custom_llm_provider, model_name, model_info),
        _iter_jsonl_entries(file_content),
        _BatchTotals(),
    )
    cache_token_params = {
        key: tokens
        for key, tokens in (
            ("cache_read_input_tokens", totals.cache_read_tokens),
            ("cache_creation_input_tokens", totals.cache_creation_tokens),
        )
        if tokens > 0
    }
    usage = Usage(
        total_tokens=totals.total_tokens,
        prompt_tokens=totals.prompt_tokens,
        completion_tokens=totals.completion_tokens,
        **cache_token_params,
    )
    models = [model_name] if model_name else list(totals.models)
    return totals.cost, usage, models


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

    batch_models = _get_batch_models_from_file_content(file_content_dictionary, model_name, custom_llm_provider)

    return batch_cost, batch_usage, batch_models


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
        if _batch_response_was_successful(_item, custom_llm_provider):
            _response_body = _get_response_from_batch_job_output_file(_item, custom_llm_provider)
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
    if (
        custom_llm_provider == "vertex_ai"
        and model_name
        and getattr(litellm, "disable_vertex_batch_output_transformation", False)
    ):
        batch_cost, _ = calculate_vertex_ai_batch_cost_and_usage(file_content_dictionary, model_name)
        verbose_logger.debug("vertex_ai_total_cost=%s", batch_cost)
        return batch_cost

    # For other providers, use the existing logic
    total_cost = _get_batch_job_cost_from_file_content(
        file_content_dictionary=file_content_dictionary,
        custom_llm_provider=custom_llm_provider,
        model_name=model_name,
        model_info=model_info,
    )
    verbose_logger.debug("total_cost=%s", total_cost)
    return total_cost


def calculate_vertex_ai_batch_cost_and_usage(
    vertex_ai_batch_responses: Iterable[dict],
    model_name: Optional[str] = None,
) -> Tuple[float, Usage]:
    """
    Calculate both cost and usage from raw Vertex AI batch responses.

    Used only when ``litellm.disable_vertex_batch_output_transformation = True``.
    In that case the GCS predictions.jsonl is returned as-is, with each line in
    the native Vertex format:

      {"request": ..., "response": {"candidates": [...], "usageMetadata": {...}}}

    usageMetadata contains promptTokenCount, candidatesTokenCount, totalTokenCount.
    """
    from litellm.cost_calculator import batch_cost_calculator

    total_cost = 0.0
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    actual_model_name = model_name or "gemini-2.0-flash-001"

    for response in vertex_ai_batch_responses:
        response_body = response.get("response")
        if response_body is None:
            continue

        usage_metadata = response_body.get("usageMetadata", {})
        _prompt = usage_metadata.get("promptTokenCount", 0) or 0
        _completion = usage_metadata.get("candidatesTokenCount", 0) or 0
        _total = usage_metadata.get("totalTokenCount", 0) or (_prompt + _completion)

        line_usage = Usage(
            prompt_tokens=_prompt,
            completion_tokens=_completion,
            total_tokens=_total,
        )

        try:
            p_cost, c_cost = batch_cost_calculator(
                usage=line_usage,
                model=actual_model_name,
                custom_llm_provider="vertex_ai",
            )
            total_cost += p_cost + c_cost
        except Exception as e:
            verbose_logger.debug("vertex_ai batch cost calculation error for line: %s", str(e))

        prompt_tokens += _prompt
        completion_tokens += _completion
        total_tokens += _total

    verbose_logger.info(
        "vertex_ai batch cost: cost=%s, prompt=%d, completion=%d, total=%d",
        total_cost,
        prompt_tokens,
        completion_tokens,
        total_tokens,
    )

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
            verbose_logger.error(
                f"Failed to extract LLM output file ID from unified file ID: {batch.output_file_id}, error: {e}"
            )

    # Build kwargs for afile_content with credentials from litellm_params
    file_content_kwargs = {
        "file_id": file_id,
        "custom_llm_provider": custom_llm_provider,
    }

    # Extract and add credentials for file access
    credentials = _extract_file_access_credentials(litellm_params)
    file_content_kwargs.update(credentials)

    _file_content = await afile_content(**file_content_kwargs)  # type: ignore[reportArgumentType]
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
            "api_key",
            "api_base",
            "api_version",
            "organization",
            "azure_ad_token",
            "azure_ad_token_provider",
            "vertex_project",
            "vertex_location",
            "vertex_credentials",
            "timeout",
            "max_retries",
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
        if verbose_logger.isEnabledFor(logging.DEBUG):
            verbose_logger.debug("json_objects=%s", json.dumps(json_objects, indent=4))
        return json_objects
    except Exception as e:
        raise e


def _iter_jsonl_lines(file_content: bytes) -> Iterator[bytes]:
    """
    Yield non-empty JSONL lines (unparsed) one at a time, so a caller can parse
    each row in its own try/except and a single malformed line cannot abort the
    whole pass. Peak memory stays bounded for large batch files.
    """
    start, length, newline = 0, len(file_content), ord("\n")
    while start < length:
        idx = file_content.find(newline, start)
        if idx == -1:
            chunk, start = file_content[start:], length
        else:
            chunk, start = file_content[start:idx], idx + 1
        line = chunk.strip()
        if line:
            yield line


def _iter_jsonl_entries(file_content: bytes) -> Iterator[dict]:
    """
    Yield parsed JSONL entries one at a time without materializing the whole file
    as a list, so peak memory stays bounded. Raises on a malformed line;
    callers that must survive bad rows should iterate ``_iter_jsonl_lines``
    and parse per-row instead.
    """
    for line in _iter_jsonl_lines(file_content):
        yield json.loads(line)


# A batch request's input tokens scale roughly with its serialized size, so this
# is a conservative per-row fallback when the token counter cannot measure a row.
_BATCH_TOKEN_ESTIMATE_BYTES_PER_TOKEN = 4


def _estimate_batch_entry_tokens(raw_line: bytes) -> int:
    """Conservative token estimate for a batch row the token counter cannot measure
    (or that cannot be parsed). Keeps the batch token total non-zero so a crafted
    row cannot evade the TPM limit, without hard-rejecting a legitimate batch."""
    return max(1, len(raw_line) // _BATCH_TOKEN_ESTIMATE_BYTES_PER_TOKEN)


def _count_entry_tokens(
    entry: dict,
    model_name: Optional[str] = None,
) -> int:
    """Token-count a single batch input entry's body (chat / text / embedding)."""
    body = entry.get("body", {}) or {}
    model = body.get("model", model_name or "")

    messages = body.get("messages")
    if messages:
        return token_counter(model=model, messages=messages)

    prompt = body.get("prompt")
    if prompt:
        return _count_prompt_or_input_tokens(model=model, value=prompt)

    input_data = body.get("input")
    if input_data:
        return _count_prompt_or_input_tokens(model=model, value=input_data)

    return 0


def _get_batch_job_cost_from_file_content(
    file_content_dictionary: List[dict],
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "hosted_vllm", "anthropic"] = "openai",
    model_name: Optional[str] = None,
    model_info: Optional[ModelInfo] = None,
) -> float:
    """
    Get the cost of a batch job from the file content
    """
    from litellm.cost_calculator import batch_cost_calculator

    try:
        total_cost: float = 0.0
        # parse the file content as json
        if verbose_logger.isEnabledFor(logging.DEBUG):
            verbose_logger.debug("file_content_dictionary=%s", json.dumps(file_content_dictionary, indent=4))
        for _item in file_content_dictionary:
            if _batch_response_was_successful(_item, custom_llm_provider):
                _response_body = _get_response_from_batch_job_output_file(_item, custom_llm_provider)
                if model_info is not None or custom_llm_provider in ("anthropic", "bedrock"):
                    usage = _get_batch_job_usage_from_response_body(_response_body, custom_llm_provider)
                    # Bedrock batch output lines report a short internal model id
                    # (e.g. "claude-sonnet-4-6") that is not in the cost map; use the
                    # deployment model name for pricing when available.
                    if custom_llm_provider == "bedrock" and model_name:
                        model = model_name
                    else:
                        model = _response_body.get("model") or model_name or ""
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
    if (
        custom_llm_provider == "vertex_ai"
        and model_name
        and getattr(litellm, "disable_vertex_batch_output_transformation", False)
    ):
        _, batch_usage = calculate_vertex_ai_batch_cost_and_usage(file_content_dictionary, model_name)
        return batch_usage

    # For other providers, use the existing logic
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    for _item in file_content_dictionary:
        if _batch_response_was_successful(_item, custom_llm_provider):
            _response_body = _get_response_from_batch_job_output_file(_item, custom_llm_provider)
            usage: Usage = _get_batch_job_usage_from_response_body(_response_body, custom_llm_provider)
            total_tokens += usage.total_tokens
            prompt_tokens += usage.prompt_tokens
            completion_tokens += usage.completion_tokens
            prompt_details = _parse_prompt_tokens_details(usage)
            cache_read_tokens += prompt_details["cache_hit_tokens"]
            cache_creation_tokens += prompt_details["cache_creation_tokens"]
    cache_token_params = {
        key: tokens
        for key, tokens in (
            ("cache_read_input_tokens", cache_read_tokens),
            ("cache_creation_input_tokens", cache_creation_tokens),
        )
        if tokens > 0
    }
    return Usage(
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        **cache_token_params,
    )


def _count_prompt_or_input_tokens(model: str, value: Any) -> int:
    """Token-count a ``prompt`` / ``input`` field that the OpenAI batch
    schema allows in four shapes:

    - ``str``: a single text prompt.
    - ``list[str]``: multiple text prompts.
    - ``list[int]``: a pre-tokenized prompt (each int counts as 1 token).
    - ``list[list[int]]``: multiple pre-tokenized prompts.

    Pre-fix only the string shapes were counted, so a caller could send
    a large ``list[list[int]]`` payload and slip past TPM rate limits
    with a recorded cost of zero tokens.
    """
    if isinstance(value, str):
        return token_counter(model=model, text=value)
    if isinstance(value, list):
        total = 0
        for chunk in value:
            if isinstance(chunk, str):
                total += token_counter(model=model, text=chunk)
            elif isinstance(chunk, int):
                # Single pre-tokenized prompt at the top level: each
                # int counts as one token.
                total += 1
            elif isinstance(chunk, list):
                # Nested pre-tokenized prompt: every int contributes a
                # token. Mixed string/int items still count.
                total += sum(1 if isinstance(t, int) else 0 for t in chunk)
                total += sum(token_counter(model=model, text=t) for t in chunk if isinstance(t, str))
        return total
    return 0


def _get_batch_job_usage_from_response_body(response_body: dict, custom_llm_provider: str = "openai") -> Usage:
    """
    Get the tokens of a batch job from the response body
    """
    if custom_llm_provider in ("anthropic", "bedrock"):
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        return AnthropicConfig().calculate_usage(
            usage_object=response_body.get("usage", None) or {},
            reasoning_content=None,
        )
    _usage_dict = response_body.get("usage", None) or {}
    usage: Usage = Usage(**_usage_dict)
    return usage


def _get_anthropic_result_from_batch_results_line(batch_results_line: dict) -> dict:
    """
    Get the ``result`` object from a line of an Anthropic message batch results JSONL file.

    Anthropic batch results lines look like:
    ``{"custom_id": ..., "result": {"type": "succeeded", "message": {..., "usage": {...}}}}``
    """
    return batch_results_line.get("result", None) or {}


def _get_response_from_batch_job_output_file(batch_job_output_file: dict, custom_llm_provider: str = "openai") -> Any:
    """
    Get the response from the batch job output file
    """
    if custom_llm_provider == "anthropic":
        return _get_anthropic_result_from_batch_results_line(batch_job_output_file).get("message", None) or {}
    if custom_llm_provider == "bedrock":
        return batch_job_output_file.get("modelOutput", None) or {}
    _response: dict = batch_job_output_file.get("response", None) or {}
    _response_body = _response.get("body", None) or {}
    return _response_body


def _batch_response_was_successful(batch_job_output_file: dict, custom_llm_provider: str = "openai") -> bool:
    """
    Check if the batch job response was successful

    OpenAI-shaped output rows report ``response.status_code == 200``; Anthropic
    message batch results lines report ``result.type == "succeeded"``; Bedrock
    batch output lines report ``modelOutput`` (and no ``error``).
    """
    if custom_llm_provider == "anthropic":
        return _get_anthropic_result_from_batch_results_line(batch_job_output_file).get("type") == "succeeded"
    if custom_llm_provider == "bedrock":
        return batch_job_output_file.get("modelOutput") is not None and batch_job_output_file.get("error") is None
    _response: dict = batch_job_output_file.get("response", None) or {}
    return _response.get("status_code", None) == 200
