import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate, chain
from typing import Final, TypeAlias, TypeVar, cast  # noqa: TID251  # sum_usage_metadata casts computed keys

import httpx

from litellm._logging import verbose_logger
from litellm.exceptions import Timeout
from litellm.types.llms.vertex_ai import (
    Candidates,
    Citation,
    CitationMetadata,
    GenerateContentResponseBody,
    GroundingMetadata,
    GroundingSegment,
    GroundingSupport,
    HttpxPartType,
    RequestBody,
    UrlContextMetadata,
    UsageMetadata,
)
from litellm.types.utils import PromptTokensDetailsWrapper, Usage

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

HOP_END_FINISH_REASON: Final = "CONTINUATION"
CUT_OFF_FINISH_REASONS: Final = frozenset({"MAX_TOKENS", HOP_END_FINISH_REASON})

_VERTEX_V1_PATH: Final = "/v1/projects/"
_VERTEX_PATH_ACCEPTING_CONTINUATION_TOKEN: Final = "/v1beta1/projects/"

LongDecodeHops: TypeAlias = tuple[GenerateContentResponseBody, ...]
AsyncHopPost: TypeAlias = Callable[[RequestBody], Awaitable[httpx.Response]]
SyncHopPost: TypeAlias = Callable[[RequestBody], httpx.Response]
_Metadata: Final = TypeVar("_Metadata", bound=Mapping[str, object])


@dataclass(frozen=True, slots=True)
class AsyncStreamResume:
    request_body: RequestBody
    open_hop: AsyncHopPost


@dataclass(frozen=True, slots=True)
class SyncStreamResume:
    request_body: RequestBody
    open_hop: SyncHopPost


StreamResume: TypeAlias = AsyncStreamResume | SyncStreamResume


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


def continuation_cap_of(request_body: RequestBody) -> int | None:
    generation_config: Final = request_body.get("generationConfig")
    if generation_config is None or "max_output_tokens" not in generation_config:
        return None
    if "candidate_count" in generation_config and _int_or_none(generation_config["candidate_count"]) != 1:
        return None
    return _int_or_none(generation_config["max_output_tokens"])


def client_continuation_token(candidate: Candidates) -> str | None:
    if candidate.get("finishReason") not in CUT_OFF_FINISH_REASONS:
        return None
    return candidate.get("continuationToken")


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
    if candidate is None or candidate.get("finishReason") != HOP_END_FINISH_REASON:
        return None
    token: Final = candidate.get("continuationToken")
    hop_tokens: Final = output_tokens_of(hop_usage)
    if not token or hop_tokens <= 0 or produced_before + hop_tokens >= output_cap:
        return None
    return token


def hop_request(request_body: RequestBody, token: str) -> RequestBody:
    return {**request_body, "continuationToken": token}


def url_for(url: str, request_body: RequestBody) -> str:
    if "continuationToken" not in request_body:
        return url
    return url.replace(_VERTEX_V1_PATH, _VERTEX_PATH_ACCEPTING_CONTINUATION_TOKEN, 1)


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
    return (content.get("parts") or ()) if content is not None else ()


def per_hop(metadata: _Metadata | tuple[_Metadata, ...]) -> tuple[_Metadata, ...]:
    return metadata if isinstance(metadata, tuple) else (metadata,)


def _candidate_grounding(candidate: Candidates) -> tuple[GroundingMetadata, ...]:
    grounding: Final = candidate.get("groundingMetadata")
    return () if grounding is None else per_hop(grounding)


def _answer_text(part: HttpxPartType) -> str:
    text: Final = part.get("text")
    return text if isinstance(text, str) and not part.get("thought") else ""


def answer_bytes(candidate: Candidates) -> int:
    return sum(len(text.encode()) for text in map(_answer_text, _candidate_parts(candidate)))


def _shifted_support(support: GroundingSupport, offset: int) -> GroundingSupport:
    segment: Final = support.get("segment")
    end: Final = None if segment is None else _int_or_none(segment.get("endIndex"))
    if segment is None or end is None:
        return support
    shifted: Final[GroundingSegment] = {
        **segment,
        "startIndex": (segment.get("startIndex") or 0) + offset,
        "endIndex": end + offset,
    }
    return {**support, "segment": shifted}


def _shifted_grounding(grounding: GroundingMetadata, offset: int) -> GroundingMetadata:
    if not offset or "groundingSupports" not in grounding:
        return grounding
    return {
        **grounding,
        "groundingSupports": tuple(_shifted_support(support, offset) for support in grounding["groundingSupports"]),
    }


def _hop_grounding(candidate: Candidates, offset: int) -> tuple[GroundingMetadata, ...]:
    return tuple(_shifted_grounding(grounding, offset) for grounding in _candidate_grounding(candidate))


def _shifted_citation(citation: Citation, offset: int) -> Citation:
    end: Final = _int_or_none(citation.get("endIndex"))
    if end is None:
        return citation
    return {**citation, "startIndex": (citation.get("startIndex") or 0) + offset, "endIndex": end + offset}


def _shifted_citation_metadata(metadata: CitationMetadata, offset: int) -> CitationMetadata:
    citations: Final = metadata.get("citations")
    sources: Final = metadata.get("citationSources")
    with_citations: Final[CitationMetadata] = (
        metadata
        if citations is None
        else {**metadata, "citations": tuple(_shifted_citation(c, offset) for c in citations)}
    )
    return (
        with_citations
        if sources is None
        else {**with_citations, "citationSources": tuple(_shifted_citation(c, offset) for c in sources)}
    )


def _hop_citations(candidate: Candidates, offset: int) -> tuple[CitationMetadata, ...]:
    metadata: Final = candidate.get("citationMetadata")
    if metadata is None:
        return ()
    return tuple(_shifted_citation_metadata(item, offset) for item in per_hop(metadata))


def with_spans_shifted(candidate: Candidates, offset: int) -> Candidates:
    if not offset:
        return candidate
    grounding: Final = _hop_grounding(candidate, offset)
    citations: Final = _hop_citations(candidate, offset)
    with_grounding: Final[Candidates] = {**candidate, "groundingMetadata": grounding} if grounding else candidate
    return {**with_grounding, "citationMetadata": citations} if citations else with_grounding


def _hop_url_context(candidate: Candidates) -> tuple[UrlContextMetadata, ...]:
    url_context: Final = candidate.get("urlContextMetadata")
    return () if url_context is None else per_hop(url_context)


def _with_hop_metadata(candidate: Candidates, hop_candidates: Sequence[Candidates]) -> Candidates:
    offsets: Final = tuple(accumulate(map(answer_bytes, hop_candidates[:-1]), initial=0))
    grounding: Final = tuple(chain.from_iterable(map(_hop_grounding, hop_candidates, offsets)))
    with_grounding: Final[Candidates] = {**candidate, "groundingMetadata": grounding} if grounding else candidate
    citations: Final = tuple(chain.from_iterable(map(_hop_citations, hop_candidates, offsets)))
    with_citations: Final[Candidates] = (
        {**with_grounding, "citationMetadata": citations} if citations else with_grounding
    )
    url_context: Final = tuple(chain.from_iterable(map(_hop_url_context, hop_candidates)))
    return {**with_citations, "urlContextMetadata": url_context} if url_context else with_citations


def merge_hops(hops: LongDecodeHops) -> GenerateContentResponseBody:
    last: Final = hops[-1]
    if len(hops) == 1:
        return last
    hop_candidates: Final = tuple(c for c in map(_first_candidate, hops) if c is not None)
    merged_first: Final[Candidates] = {
        **(_first_candidate(last) or Candidates()),
        "content": {"role": "model", "parts": list(chain.from_iterable(map(_candidate_parts, hop_candidates)))},
    }
    merged: Final[GenerateContentResponseBody] = {
        **last,
        "candidates": [_with_hop_metadata(merged_first, hop_candidates), *(last.get("candidates") or [])[1:]],
        "usageMetadata": sum_usage_metadata(*(hop.get("usageMetadata", UsageMetadata()) for hop in hops)),
    }
    first_response_id: Final = hops[0].get("responseId")
    return {**merged, "responseId": first_response_id} if first_response_id else merged


def initial_prompt_tokens_of(hops: LongDecodeHops) -> int | None:
    if len(hops) == 1:
        return None
    return hops[0].get("usageMetadata", UsageMetadata()).get("promptTokenCount")


def with_initial_prompt_tokens(usage: Usage, initial_prompt_tokens: int | None) -> Usage:
    if initial_prompt_tokens is None:
        return usage
    details: Final = usage.prompt_tokens_details
    marked: Final = (
        details.model_copy(update={"initial_prompt_tokens": initial_prompt_tokens})
        if isinstance(details, PromptTokensDetailsWrapper)
        else PromptTokensDetailsWrapper(initial_prompt_tokens=initial_prompt_tokens)
    )
    return usage.model_copy(update={"prompt_tokens_details": marked})


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


def log_failed_hop(error: Exception | str) -> None:
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
    output_cap: Final = continuation_cap_of(request_body)
    hops: LongDecodeHops = (first,)  # rebind-ok: grows by one hop per loop pass, CI bans the recursive form
    while (token := _unary_next_hop_token(hops, output_cap)) is not None:
        try:
            body = parse_body(await post(hop_request(request_body, token)))
        except HOP_FAILURES as e:
            log_failed_hop(e)
            return hops
        hops = (*hops, body)
    return hops


def collect_hops_sync(
    first: GenerateContentResponseBody,
    request_body: RequestBody,
    post: SyncHopPost,
) -> LongDecodeHops:
    output_cap: Final = continuation_cap_of(request_body)
    hops: LongDecodeHops = (first,)  # rebind-ok: grows by one hop per loop pass, CI bans the recursive form
    while (token := _unary_next_hop_token(hops, output_cap)) is not None:
        try:
            body = parse_body(post(hop_request(request_body, token)))
        except HOP_FAILURES as e:
            log_failed_hop(e)
            return hops
        hops = (*hops, body)
    return hops
