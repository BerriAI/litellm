"""Rebuilds the generateContent grounding fields chat callers read from what the Interactions API returns."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate, chain, groupby
from typing import Final, TypeAlias, TypeVar

from pydantic import JsonValue, TypeAdapter
from pydantic_core import to_jsonable_python

JsonMapping: TypeAlias = Mapping[str, JsonValue]
_Item = TypeVar("_Item")

_SEARCH_CALL: Final = "google_search_call"
_SEARCH_RESULT: Final = "google_search_result"
_URL_CONTEXT_RESULT: Final = "url_context_result"
GROUNDING_STEP_TYPES: Final = frozenset({_SEARCH_CALL, _SEARCH_RESULT, _URL_CONTEXT_RESULT})


@dataclass(frozen=True, slots=True)
class VertexMetadata:
    grounding: tuple[JsonMapping, ...]
    url_context: tuple[JsonMapping, ...]
    citations: tuple[JsonMapping, ...] = ()


@dataclass(frozen=True, slots=True)
class _Citation:
    start: int
    end: int
    text: str
    uri: str
    title: str


def _dicts(raw: JsonValue | None) -> tuple[JsonMapping, ...]:
    return tuple(item for item in raw if isinstance(item, dict)) if isinstance(raw, list) else ()


def _strings(raw: JsonValue | None) -> tuple[str, ...]:
    return tuple(item for item in raw if isinstance(item, str)) if isinstance(raw, list) else ()


def _queries(arguments: JsonValue | None) -> tuple[str, ...]:
    return _strings(arguments.get("queries")) if isinstance(arguments, dict) else ()


def _is_recitation(anno: JsonMapping, is_grounded: bool) -> bool:
    return anno.get("type") == "url_citation" and not (is_grounded and anno.get("url"))


def _block_citations(block: JsonMapping, byte_offset: int, is_grounded: bool) -> tuple[_Citation, ...]:
    encoded: Final = str(block.get("text") or "").encode()
    spans: Final = tuple(
        (anno, anno.get("start_index"), anno.get("end_index"), anno.get("url") or anno.get("uri") or anno.get("source"))
        for anno in _dicts(block.get("annotations"))
        if not _is_recitation(anno, is_grounded)
    )
    return tuple(
        _Citation(
            start=start + byte_offset,
            end=end + byte_offset,
            text=encoded[start:end].decode(errors="ignore"),
            uri=uri,
            title=title if isinstance(title := anno.get("title"), str) else uri,
        )
        for anno, start, end, uri in spans
        if isinstance(start, int) and isinstance(end, int) and isinstance(uri, str) and uri
    )


def _typed(steps: Sequence[JsonMapping], step_type: str) -> tuple[JsonMapping, ...]:
    return tuple(step for step in steps if step.get("type") == step_type)


def _flat(groups: Iterable[Sequence[_Item]]) -> tuple[_Item, ...]:
    return tuple(chain.from_iterable(groups))


def _placed_text_blocks(steps: Sequence[JsonMapping]) -> tuple[tuple[JsonMapping, int], ...]:
    content: Final = _flat(_dicts(step.get("content")) for step in _typed(steps, "model_output"))
    blocks: Final = _typed(content, "text")
    lengths: Final = tuple(len(str(block.get("text") or "").encode()) for block in blocks)
    return tuple(zip(blocks, accumulate(lengths, initial=0)))


def _is_grounded(steps: Sequence[JsonMapping]) -> bool:
    return any(step.get("type") in GROUNDING_STEP_TYPES for step in steps)


def _citations(steps: Sequence[JsonMapping]) -> tuple[_Citation, ...]:
    is_grounded: Final = _is_grounded(steps)
    return _flat(_block_citations(block, offset, is_grounded) for block, offset in _placed_text_blocks(steps))


def _block_recitations(block: JsonMapping, byte_offset: int, is_grounded: bool) -> tuple[JsonMapping, ...]:
    return tuple(
        {
            "startIndex": start + byte_offset,
            "endIndex": end + byte_offset,
            "uri": url if isinstance(url := anno.get("url"), str) else "",
            **({"title": title} if isinstance(title := anno.get("title"), str) and title else {}),
        }
        for anno in _dicts(block.get("annotations"))
        if _is_recitation(anno, is_grounded)
        and isinstance(start := anno.get("start_index"), int)
        and isinstance(end := anno.get("end_index"), int)
    )


def _passage_span(passage: JsonMapping) -> tuple[int, int]:
    start, end = passage["startIndex"], passage["endIndex"]
    return (start if isinstance(start, int) else 0, end if isinstance(end, int) else 0)


def _recitations(steps: Sequence[JsonMapping]) -> Mapping[str, object]:
    is_grounded: Final = _is_grounded(steps)
    recited: Final = _flat(
        _block_recitations(block, offset, is_grounded) for block, offset in _placed_text_blocks(steps)
    )
    longest_by_start: Final = {
        (passage["startIndex"], passage["uri"]): passage for passage in sorted(recited, key=_passage_span)
    }
    return {"citations": tuple(longest_by_start.values())} if recited else {}


def _support(
    span: tuple[int, int, str], cited: Sequence[_Citation], source_positions: Mapping[tuple[str, str], int]
) -> Mapping[str, object]:
    start, end, text = span
    return {
        "segment": {"startIndex": start, "endIndex": end, "text": text},
        "groundingChunkIndices": tuple(source_positions[(citation.uri, citation.title)] for citation in cited),
    }


def _span_of(citation: _Citation) -> tuple[int, int, str]:
    return (citation.start, citation.end, citation.text)


def _grounding(steps: Sequence[JsonMapping]) -> Mapping[str, object]:
    citations: Final = _citations(steps)
    sources: Final = tuple(dict.fromkeys((cited.uri, cited.title) for cited in citations))
    source_positions: Final = {source: position for position, source in enumerate(sources)}
    by_span: Final = groupby(sorted(citations, key=_span_of), key=_span_of)
    supports: Final = tuple(_support(span, tuple(cited), source_positions) for span, cited in by_span)
    queries: Final = _flat(_queries(step.get("arguments")) for step in _typed(steps, _SEARCH_CALL))
    found: Final = _flat(_dicts(step.get("result")) for step in _typed(steps, _SEARCH_RESULT))
    suggestions: Final = tuple(
        suggestion
        for result in found
        if isinstance(suggestion := result.get("search_suggestions") or result.get("rendered_content"), str)
        and suggestion
    )
    return {
        **({"searchEntryPoint": {"renderedContent": suggestions[-1]}} if suggestions else {}),
        **(
            {"groundingChunks": tuple({"web": {"uri": uri, "title": title}} for uri, title in sources)}
            if sources
            else {}
        ),
        **({"groundingSupports": supports} if supports else {}),
        **({"webSearchQueries": queries} if queries else {}),
    }


def _url_context(steps: Sequence[JsonMapping]) -> Mapping[str, object]:
    fetched: Final = _flat(_dicts(step.get("result")) for step in _typed(steps, _URL_CONTEXT_RESULT))
    retrieved: Final = tuple(
        {
            "retrievedUrl": url,
            "urlRetrievalStatus": f"URL_RETRIEVAL_STATUS_{str(page.get('status') or 'unspecified').upper()}",
        }
        for page in fetched
        if isinstance(url := page.get("url"), str)
    )
    return {"urlMetadata": retrieved} if retrieved else {}


def vertex_metadata_of(steps: Sequence[JsonValue]) -> VertexMetadata:
    step_maps: Final = tuple(step for step in steps if isinstance(step, dict))
    as_json: Final = TypeAdapter(dict[str, JsonValue])
    grounding: Final = as_json.validate_python(to_jsonable_python(_grounding(step_maps)))
    url_context: Final = as_json.validate_python(to_jsonable_python(_url_context(step_maps)))
    recited: Final = as_json.validate_python(to_jsonable_python(_recitations(step_maps)))
    return VertexMetadata(
        grounding=(grounding,) if grounding else (),
        url_context=(url_context,) if url_context else (),
        citations=(recited,) if recited else (),
    )
