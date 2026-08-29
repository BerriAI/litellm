import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.get_litellm_params import AWS_CREDENTIAL_KWARGS_KEYS
from litellm.litellm_core_utils.llm_cost_calc.utils import parse_prompt_tokens_details
from litellm.types.llms.openai import Batch
from litellm.types.utils import CallTypes, ModelInfo, Usage
from litellm.utils import token_counter


async def calculate_batch_cost_and_usage(
    file_content_dictionary: list[dict],
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "hosted_vllm", "anthropic"],
    model_name: str | None = None,
    model_info: ModelInfo | None = None,
) -> tuple[float, Usage, list[str]]:
    """
    Calculate the cost and usage of a batch.

    Args:
        model_info: Optional deployment-level model info with custom batch
            pricing. Threaded through to batch_cost_calculator so that
            deployment-specific pricing (e.g. input_cost_per_token_batches)
            is used instead of the global cost map.
    """
    if (
        custom_llm_provider == "vertex_ai"
        and model_name
        and getattr(litellm, "disable_vertex_batch_output_transformation", False)
    ):
        batch_cost, batch_usage = calculate_vertex_ai_batch_cost_and_usage(file_content_dictionary, model_name)
        return batch_cost, batch_usage, [model_name]

    return _aggregate_batch_cost_usage_models(
        entries=file_content_dictionary,
        custom_llm_provider=custom_llm_provider,
        model_name=model_name,
        model_info=model_info,
    )


async def _handle_completed_batch(
    batch: Batch,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "hosted_vllm", "anthropic"],
    model_name: str | None = None,
    litellm_params: dict | None = None,
    model_info: ModelInfo | None = None,
) -> tuple[float, Usage, list[str]]:
    """Fetch a completed batch's output file and aggregate its cost, usage, and
    models in a single pass over the JSONL lines, so the parsed file content is
    never materialized in memory.

    Args:
        batch: The batch object
        custom_llm_provider: The LLM provider
        model_name: Optional model name
        litellm_params: Optional litellm parameters containing credentials (api_key, api_base, etc.)
        model_info: Optional deployment-level model info with custom pricing,
            threaded through so a deployment's configured rates win over the
            global cost map.
    """
    # A completed batch whose request lines all failed has no output file - the
    # results are written to a separate error_file_id and output_file_id is None.
    # There is nothing to price or measure, so report an empty result set instead
    # of calling _fetch_batch_output_file_content, which raises on a missing
    # output file. Without this guard the logging worker crashes on every
    # aretrieve_batch poll and the completed batch's zero-cost accounting is lost.
    # The generic retrieval helper keeps raising for callers that explicitly ask
    # for a missing output file.
    if batch.output_file_id is None:
        return 0.0, Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0), []

    file_content = await _fetch_batch_output_file_content(batch, custom_llm_provider, litellm_params=litellm_params)

    if (
        custom_llm_provider == "vertex_ai"
        and model_name
        and getattr(litellm, "disable_vertex_batch_output_transformation", False)
    ):
        batch_cost, batch_usage = calculate_vertex_ai_batch_cost_and_usage(
            _get_file_content_as_dictionary(file_content), model_name
        )
        return batch_cost, batch_usage, [model_name]

    return _aggregate_batch_cost_usage_models(
        entries=_iter_batch_output_entries(file_content),
        custom_llm_provider=custom_llm_provider,
        model_name=model_name,
        model_info=model_info,
    )


@dataclass(frozen=True, slots=True)
class _BatchOutputLineStats:
    cost: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    model: str | None


def _iter_successful_output_line_stats(
    entries: Iterable[dict],
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "hosted_vllm", "anthropic", "bedrock"],
    model_name: str | None,
    model_info: ModelInfo | None,
) -> Iterator[_BatchOutputLineStats]:
    for entry in entries:
        stats = _safe_output_line_stats(entry, custom_llm_provider, model_name, model_info)
        if stats is not None:
            yield stats


def _safe_output_line_stats(
    entry: Mapping[str, Any],
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "hosted_vllm", "anthropic", "bedrock"],
    model_name: str | None,
    model_info: ModelInfo | None,
) -> _BatchOutputLineStats | None:
    """Return the stats for one batch output line, or None for a line that is
    unsuccessful or cannot be costed, so a single bad line never aborts the
    whole batch's cost accounting."""
    custom_id: Final = entry.get("custom_id") if isinstance(entry, dict) else None
    try:
        if not _batch_response_was_successful(entry, custom_llm_provider):
            return None
        return _compute_output_line_stats(entry, custom_llm_provider, model_name, model_info)
    except Exception as e:  # noqa: BLE001  # any single line's costing failure must not abort the whole batch
        verbose_logger.warning(
            "batch output line could not be costed, so it is billed at $0 and the rest of the batch "
            "is still billed. custom_id=%s error=%s",
            custom_id,
            str(e),
        )
        return None


def _compute_output_line_stats(
    entry: Mapping[str, Any],
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "hosted_vllm", "anthropic", "bedrock"],
    model_name: str | None,
    model_info: ModelInfo | None,
) -> _BatchOutputLineStats:
    response_body: Final = _get_response_from_batch_job_output_file(entry, custom_llm_provider)
    usage: Final = _get_batch_job_usage_from_response_body(response_body, custom_llm_provider)
    prompt_details: Final = parse_prompt_tokens_details(usage)
    raw_model: Final = response_body.get("model")
    response_model: Final = raw_model if isinstance(raw_model, str) and raw_model else None
    return _BatchOutputLineStats(
        cost=_output_line_cost(
            response_body=response_body,
            usage=usage,
            custom_llm_provider=custom_llm_provider,
            model_name=model_name,
            response_model=response_model,
            model_info=model_info,
        ),
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        cache_read_tokens=prompt_details["cache_hit_tokens"],
        cache_creation_tokens=prompt_details["cache_creation_tokens"],
        model=response_model,
    )


def _output_line_cost(
    response_body: Mapping[str, Any],
    usage: Usage,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "hosted_vllm", "anthropic", "bedrock"],
    model_name: str | None,
    response_model: str | None,
    model_info: ModelInfo | None,
) -> float:
    from litellm.cost_calculator import batch_cost_calculator

    if model_info is None and custom_llm_provider not in ("anthropic", "bedrock"):
        return litellm.completion_cost(
            completion_response=response_body,
            custom_llm_provider=custom_llm_provider,
            call_type=CallTypes.aretrieve_batch.value,
        )
    cost_model: Final = (
        model_name if custom_llm_provider == "bedrock" and model_name else response_model or model_name or ""
    )
    prompt_cost, completion_cost = batch_cost_calculator(
        usage=usage,
        model=cost_model,
        custom_llm_provider=custom_llm_provider,
        model_info=model_info,
    )
    return prompt_cost + completion_cost


def _aggregate_batch_cost_usage_models(
    entries: Iterable[dict],
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "hosted_vllm", "anthropic", "bedrock"],
    model_name: str | None = None,
    model_info: ModelInfo | None = None,
) -> tuple[float, Usage, list[str]]:
    """Aggregate cost, usage, and models from batch output entries in a single
    pass, holding one small stats record per line instead of the parsed file."""
    line_stats: Final = tuple(_iter_successful_output_line_stats(entries, custom_llm_provider, model_name, model_info))

    cache_token_params: Final = {
        key: tokens
        for key, tokens in (
            ("cache_read_input_tokens", sum(stats.cache_read_tokens for stats in line_stats)),
            ("cache_creation_input_tokens", sum(stats.cache_creation_tokens for stats in line_stats)),
        )
        if tokens > 0
    }
    batch_usage: Final = Usage(
        total_tokens=sum(stats.total_tokens for stats in line_stats),
        prompt_tokens=sum(stats.prompt_tokens for stats in line_stats),
        completion_tokens=sum(stats.completion_tokens for stats in line_stats),
        **cache_token_params,
    )
    batch_models: Final = [model_name] if model_name else [stats.model for stats in line_stats if stats.model]
    total_cost: Final = sum((stats.cost for stats in line_stats), 0.0)
    verbose_logger.debug("batch output aggregate: cost=%s usage=%s models=%s", total_cost, batch_usage, batch_models)
    return total_cost, batch_usage, batch_models


def calculate_vertex_ai_batch_cost_and_usage(
    vertex_ai_batch_responses: list[dict],
    model_name: str | None = None,
) -> tuple[float, Usage]:
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
    actual_model_name: Final = model_name or "gemini-2.0-flash-001"

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


def _provider_output_file_id(output_file_id: str) -> str:
    """
    Resolve the file id the provider actually knows: unified ids yield their embedded
    llm_output_file_id, model-encoded ids decode to the raw provider id, raw ids pass through.
    """
    from litellm.proxy.openai_files_endpoints.common_utils import (
        _is_base64_encoded_unified_file_id,
        get_original_file_id,
    )

    unified_file_id: Final = _is_base64_encoded_unified_file_id(output_file_id)
    if not unified_file_id:
        return get_original_file_id(output_file_id)
    try:
        extracted: Final = unified_file_id.split("llm_output_file_id,")[1].split(";")[0]
    except (IndexError, AttributeError) as e:
        verbose_logger.error(
            "Failed to extract LLM output file ID from unified file ID: %s, error: %s",
            output_file_id,
            e,
        )
        return output_file_id
    verbose_logger.debug("Extracted LLM output file ID from unified file ID: %s", extracted)
    return extracted


async def _fetch_batch_output_file_content(
    batch: Batch,
    custom_llm_provider: Literal["openai", "azure", "vertex_ai", "hosted_vllm", "anthropic"] = "openai",
    litellm_params: dict | None = None,
) -> bytes:
    """
    Fetch the batch output file and return its raw JSONL bytes

    Args:
        batch: The batch object
        custom_llm_provider: The LLM provider
        litellm_params: Optional litellm parameters containing credentials (api_key, api_base, etc.)
                       Required for Azure and other providers that need authentication
    """
    from litellm.files.main import afile_content

    if batch.output_file_id is None:
        raise ValueError("Output file id is None cannot retrieve file content")

    file_id: Final = _provider_output_file_id(batch.output_file_id)

    # Build kwargs for afile_content with credentials from litellm_params
    file_content_kwargs: Final = {
        "file_id": file_id,
        "custom_llm_provider": custom_llm_provider,
    }

    # Extract and add credentials for file access
    credentials: Final = _extract_file_access_credentials(litellm_params)
    file_content_kwargs.update(credentials)

    _file_content: Final = await afile_content(**file_content_kwargs)
    return _file_content.content


def _extract_file_access_credentials(litellm_params: dict | None) -> dict:
    """
    Extract credentials from litellm_params for file access operations.

    This method extracts relevant authentication and configuration parameters
    needed for accessing files across different providers (Azure, Vertex AI, etc.).

    Args:
        litellm_params: Dictionary containing litellm parameters with credentials

    Returns:
        Dictionary containing only the credentials needed for file access
    """
    credentials: Final = {}

    if litellm_params:
        # List of credential keys that should be passed to file operations
        credential_keys: Final = (
            "api_key",
            "api_base",
            "api_version",
            "organization",
            "azure_ad_token",
            "azure_ad_token_provider",
            "vertex_project",
            "vertex_location",
            "vertex_credentials",
            "gcs_bucket_name",
            "bucket_name",
            "timeout",
            "max_retries",
            "_litellm_internal_model_credentials",
            *AWS_CREDENTIAL_KWARGS_KEYS,
        )
        for key in credential_keys:
            if key in litellm_params:
                credentials[key] = litellm_params[key]

    return credentials


def _get_file_content_as_dictionary(file_content: bytes) -> list[dict]:
    """
    Get the file content as a list of dictionaries from JSON Lines format,
    skipping malformed lines
    """
    return list(_iter_batch_output_entries(file_content))


def _iter_batch_input_lines(file_content: bytes) -> Iterator[bytes]:
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


def _iter_batch_output_entries(file_content: bytes) -> Iterator[dict]:
    """
    Yield parsed batch output JSONL entries one at a time without materializing
    the whole file as a list, so peak memory stays bounded. A malformed or
    non-object line is skipped with a warning so one bad line never aborts the
    whole batch's cost accounting.
    """
    for line in _iter_batch_input_lines(file_content):
        entry = _parse_batch_output_line(line)
        if entry is not None:
            yield entry


def _parse_batch_output_line(line: bytes) -> dict | None:
    try:
        parsed: Final = json.loads(line)
    except ValueError as e:
        verbose_logger.warning("skipping malformed batch output line: %s", str(e))
        return None
    if isinstance(parsed, dict):
        return parsed
    verbose_logger.warning("skipping non-object batch output line of type %s", type(parsed).__name__)
    return None


# A batch request's input tokens scale roughly with its serialized size, so this
# is a conservative per-row fallback when the token counter cannot measure a row.
_BATCH_TOKEN_ESTIMATE_BYTES_PER_TOKEN: Final = 4


def _estimate_batch_entry_tokens(raw_line: bytes) -> int:
    """Conservative token estimate for a batch row the token counter cannot measure
    (or that cannot be parsed). Keeps the batch token total non-zero so a crafted
    row cannot evade the TPM limit, without hard-rejecting a legitimate batch."""
    return max(1, len(raw_line) // _BATCH_TOKEN_ESTIMATE_BYTES_PER_TOKEN)


def _count_entry_tokens(
    entry: dict,
    model_name: str | None = None,
) -> int:
    """Token-count a single batch input entry's body (chat / text / embedding)."""
    body: Final = entry.get("body", {}) or {}
    model: Final = body.get("model", model_name or "")

    messages: Final = body.get("messages")
    if messages:
        return token_counter(model=model, messages=messages)

    prompt: Final = body.get("prompt")
    if prompt:
        return _count_prompt_or_input_tokens(model=model, value=prompt)

    input_data: Final = body.get("input")
    if input_data:
        return _count_prompt_or_input_tokens(model=model, value=input_data)

    return 0


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


def _get_batch_job_usage_from_response_body(
    response_body: Mapping[str, Any], custom_llm_provider: str = "openai"
) -> Usage:
    """
    Get the tokens of a batch job from the response body
    """
    if custom_llm_provider in ("anthropic", "bedrock"):
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig
        from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

        usage_object: Final = response_body.get("usage", None) or {}
        if custom_llm_provider == "bedrock" and AmazonConverseConfig.is_converse_usage_shape(usage_object):
            return AmazonConverseConfig().usage_from_batch_output(usage_object)
        anthropic_usage: Final = AnthropicConfig().calculate_usage(
            usage_object=usage_object,
            reasoning_content=None,
        )
        if usage_object and anthropic_usage.total_tokens == 0:
            verbose_logger.warning(
                "batch output line reported usage this parser does not understand, so it will be billed at $0. "
                "provider=%s usage_keys=%s",
                custom_llm_provider,
                sorted(usage_object.keys()),
            )
        return anthropic_usage
    from litellm.responses.utils import ResponseAPILoggingUtils

    _usage_dict: Final = response_body.get("usage", None) or {}
    if ResponseAPILoggingUtils._is_response_api_usage(_usage_dict):
        return ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(_usage_dict)
    usage: Final[Usage] = Usage(**_usage_dict)
    return usage


def _get_anthropic_result_from_batch_results_line(batch_results_line: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Get the ``result`` object from a line of an Anthropic message batch results JSONL file.

    Anthropic batch results lines look like:
    ``{"custom_id": ..., "result": {"type": "succeeded", "message": {..., "usage": {...}}}}``
    """
    return batch_results_line.get("result", None) or {}


def _get_response_from_batch_job_output_file(
    batch_job_output_file: Mapping[str, Any], custom_llm_provider: str = "openai"
) -> Mapping[str, Any]:
    """
    Get the response from the batch job output file
    """
    if custom_llm_provider == "anthropic":
        return _get_anthropic_result_from_batch_results_line(batch_job_output_file).get("message", None) or {}
    if custom_llm_provider == "bedrock":
        return batch_job_output_file.get("modelOutput", None) or {}
    _response: Final[dict] = batch_job_output_file.get("response", None) or {}
    _response_body: Final = _response.get("body", None) or {}
    return _response_body


def _batch_response_was_successful(
    batch_job_output_file: Mapping[str, Any], custom_llm_provider: str = "openai"
) -> bool:
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
    _response: Final[dict] = batch_job_output_file.get("response", None) or {}
    return _response.get("status_code", None) == 200
