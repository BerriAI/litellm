import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Final, TypeAlias, cast  # noqa: TID251  # sum_usage_metadata builds a TypedDict from computed keys

import httpx

from litellm._logging import verbose_logger
from litellm.exceptions import Timeout
from litellm.types.llms.vertex_ai import (
    Candidates,
    GenerateContentResponseBody,
    HttpxPartType,
    RequestBody,
    UsageMetadata,
)

from ..common_utils import VertexAIError

_SUMMED_USAGE_COUNTS: Final = (
    "promptTokenCount",
    "candidatesTokenCount",
    "totalTokenCount",
    "cachedContentTokenCount",
    "thoughtsTokenCount",
    "toolUsePromptTokenCount",
    "responseTokenCount",
)

_CONCATENATED_USAGE_DETAILS: Final = (
    "promptTokensDetails",
    "cacheTokensDetails",
    "candidatesTokensDetails",
    "responseTokensDetails",
    "toolUsePromptTokensDetails",
)

HOP_FAILURES: Final = (VertexAIError, httpx.HTTPError, Timeout)

CONTINUABLE_FINISH_REASONS: Final = frozenset({"MAX_TOKENS", "CONTINUATION"})

LongDecodeHops: TypeAlias = tuple[GenerateContentResponseBody, ...]
AsyncHopPost: TypeAlias = Callable[[RequestBody], Awaitable[httpx.Response]]
SyncHopPost: TypeAlias = Callable[[RequestBody], httpx.Response]


@dataclass(frozen=True, slots=True)
class AsyncStreamResume:
    request_body: RequestBody
    open_hop: AsyncHopPost


@dataclass(frozen=True, slots=True)
class SyncStreamResume:
    request_body: RequestBody
    open_hop: SyncHopPost


StreamResume: TypeAlias = AsyncStreamResume | SyncStreamResume


def output_cap_of(request_body: RequestBody) -> int | None:
    generation_config: Final = request_body.get("generationConfig")
    if generation_config is None or "max_output_tokens" not in generation_config:
        return None
    return generation_config["max_output_tokens"]


def _first_candidate(body: GenerateContentResponseBody) -> Candidates | None:
    candidates: Final = body.get("candidates")
    return candidates[0] if candidates else None


def output_tokens_of(usage: UsageMetadata) -> int:
    return usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)


def next_hop_token(
    body: GenerateContentResponseBody,
    hop_usage: UsageMetadata,
    produced_before: int,
    output_cap: int | None,
) -> str | None:
    if output_cap is None:
        return None
    candidate: Final = _first_candidate(body)
    if candidate is None or candidate.get("finishReason") not in CONTINUABLE_FINISH_REASONS:
        return None
    token: Final = candidate.get("continuationToken")
    hop_tokens: Final = output_tokens_of(hop_usage)
    if not token or hop_tokens <= 0 or produced_before + hop_tokens >= output_cap:
        return None
    return token


def hop_request(request_body: RequestBody, token: str) -> RequestBody:
    return {**request_body, "continuationToken": token}


def sum_usage_metadata(first: UsageMetadata, *rest: UsageMetadata) -> UsageMetadata:
    usages: Final = (first, *rest)
    counts: Final = {
        key: sum(usage.get(key, 0) for usage in usages)
        for key in _SUMMED_USAGE_COUNTS
        if any(key in usage for usage in usages)
    }
    details: Final = {
        key: list(chain.from_iterable(usage.get(key, ()) for usage in usages))
        for key in _CONCATENATED_USAGE_DETAILS
        if any(key in usage for usage in usages)
    }
    return cast(UsageMetadata, {**usages[-1], **counts, **details})  # cast-ok: a TypedDict cannot take computed keys


def _candidate_parts(candidate: Candidates) -> Sequence[HttpxPartType]:
    content: Final = candidate.get("content")
    return content.get("parts", ()) if content is not None else ()


def merge_hops(hops: LongDecodeHops) -> GenerateContentResponseBody:
    last: Final = hops[-1]
    if len(hops) == 1:
        return last
    hop_candidates: Final = tuple(c for c in map(_first_candidate, hops) if c is not None)
    merged_first: Final[Candidates] = {
        **(_first_candidate(last) or Candidates()),
        "content": {"role": "model", "parts": list(chain.from_iterable(map(_candidate_parts, hop_candidates)))},
    }
    return {
        **last,
        "candidates": [merged_first, *(last.get("candidates") or [])[1:]],
        "usageMetadata": sum_usage_metadata(*(hop.get("usageMetadata", UsageMetadata()) for hop in hops)),
    }


def parse_body(response: httpx.Response) -> GenerateContentResponseBody:
    try:
        return GenerateContentResponseBody(**response.json())
    except Exception as e:
        raise VertexAIError(
            message=(
                f"Error converting to valid response block={e}. "
                "File an issue if litellm error - https://github.com/BerriAI/litellm/issues"
            ),
            status_code=422,
            headers=response.headers,
        )


def hops_original_response(hops: LongDecodeHops, first_response: httpx.Response) -> str:
    return first_response.text if len(hops) == 1 else json.dumps(hops)


def log_failed_hop(error: Exception) -> None:
    verbose_logger.warning(
        "Gemini long decode: a continuation hop failed, returning the answer so far with its continuation_token: %s",
        error,
    )


def _unary_next_hop_token(hops: LongDecodeHops, output_cap: int | None) -> str | None:
    last: Final = hops[-1]
    return next_hop_token(
        last,
        last.get("usageMetadata", UsageMetadata()),
        sum(output_tokens_of(hop.get("usageMetadata", UsageMetadata())) for hop in hops[:-1]),
        output_cap,
    )


async def collect_hops_async(
    first: GenerateContentResponseBody,
    request_body: RequestBody,
    post: AsyncHopPost,
) -> LongDecodeHops:
    return await _extend_hops_async((first,), output_cap_of(request_body), request_body, post)


async def _extend_hops_async(
    hops: LongDecodeHops,
    output_cap: int | None,
    request_body: RequestBody,
    post: AsyncHopPost,
) -> LongDecodeHops:
    token: Final = _unary_next_hop_token(hops, output_cap)
    if token is None:
        return hops
    try:
        body: Final = parse_body(await post(hop_request(request_body, token)))
    except HOP_FAILURES as e:
        log_failed_hop(e)
        return hops
    return await _extend_hops_async((*hops, body), output_cap, request_body, post)


def collect_hops_sync(
    first: GenerateContentResponseBody,
    request_body: RequestBody,
    post: SyncHopPost,
) -> LongDecodeHops:
    return _extend_hops_sync((first,), output_cap_of(request_body), request_body, post)


def _extend_hops_sync(
    hops: LongDecodeHops,
    output_cap: int | None,
    request_body: RequestBody,
    post: SyncHopPost,
) -> LongDecodeHops:
    token: Final = _unary_next_hop_token(hops, output_cap)
    if token is None:
        return hops
    try:
        body: Final = parse_body(post(hop_request(request_body, token)))
    except HOP_FAILURES as e:
        log_failed_hop(e)
        return hops
    return _extend_hops_sync((*hops, body), output_cap, request_body, post)
