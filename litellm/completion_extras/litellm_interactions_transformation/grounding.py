"""Rebuilds the generateContent grounding fields chat callers read from what the Interactions API returns."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

from pydantic import JsonValue

JsonMapping: TypeAlias = Mapping[str, JsonValue]

_SEARCH_CALL: Final = "google_search_call"
_SEARCH_RESULT: Final = "google_search_result"
_URL_CONTEXT_RESULT: Final = "url_context_result"
GROUNDING_STEP_TYPES: Final = frozenset({_SEARCH_CALL, _SEARCH_RESULT, _URL_CONTEXT_RESULT})


@dataclass(frozen=True, slots=True)
class VertexMetadata:
    grounding: tuple[dict[str, JsonValue], ...]
    url_context: tuple[dict[str, JsonValue], ...]


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


def _block_citations(block: JsonMapping, byte_offset: int) -> tuple[_Citation, ...]:
    encoded: Final = str(block.get("text") or "").encode()
    spans: Final = tuple(
        (anno, anno.get("start_index"), anno.get("end_index"), anno.get("url") or anno.get("uri"))
        for anno in _dicts(block.get("annotations"))
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


def _citations(steps: Sequence[JsonMapping]) -> tuple[_Citation, ...]:
    blocks: Final = tuple(
        block
        for step in steps
        if step.get("type") == "model_output"
        for block in _dicts(step.get("content"))
        if block.get("type") == "text"
    )
    lengths: Final = tuple(len(str(block.get("text") or "").encode()) for block in blocks)
    offsets: Final = tuple(sum(lengths[:position]) for position in range(len(blocks)))
    return tuple(cited for block, offset in zip(blocks, offsets) for cited in _block_citations(block, offset))


def _grounding(steps: Sequence[JsonMapping]) -> dict[str, JsonValue]:
    citations: Final = _citations(steps)
    sources: Final = tuple(dict.fromkeys((cited.uri, cited.title) for cited in citations))
    spans: Final = tuple(dict.fromkeys((cited.start, cited.end, cited.text) for cited in citations))
    queries: Final = tuple(
        query for step in steps if step.get("type") == _SEARCH_CALL for query in _queries(step.get("arguments"))
    )
    suggestions: Final = tuple(
        suggestion
        for step in steps
        if step.get("type") == _SEARCH_RESULT
        for found in _dicts(step.get("result"))
        if isinstance(suggestion := found.get("search_suggestions"), str) and suggestion
    )
    supports: Final[list[JsonValue]] = [
        {
            "segment": {"startIndex": start, "endIndex": end, "text": text},
            "groundingChunkIndices": [
                sources.index((cited.uri, cited.title))
                for cited in citations
                if (cited.start, cited.end, cited.text) == (start, end, text)
            ],
        }
        for start, end, text in spans
    ]
    return {
        **({"searchEntryPoint": {"renderedContent": suggestions[-1]}} if suggestions else {}),
        **({"groundingChunks": [{"web": {"uri": uri, "title": title}} for uri, title in sources]} if sources else {}),
        **({"groundingSupports": supports} if supports else {}),
        **({"webSearchQueries": list(queries)} if queries else {}),
    }


def _url_context(steps: Sequence[JsonMapping]) -> dict[str, JsonValue]:
    retrieved: Final[list[JsonValue]] = [
        {
            "retrievedUrl": url,
            "urlRetrievalStatus": f"URL_RETRIEVAL_STATUS_{str(fetched.get('status') or 'unspecified').upper()}",
        }
        for step in steps
        if step.get("type") == _URL_CONTEXT_RESULT
        for fetched in _dicts(step.get("result"))
        if isinstance(url := fetched.get("url"), str)
    ]
    return {"urlMetadata": retrieved} if retrieved else {}


def vertex_metadata_of(steps: Sequence[JsonValue]) -> VertexMetadata:
    step_maps: Final = tuple(step for step in steps if isinstance(step, dict))
    grounding: Final = _grounding(step_maps)
    url_context: Final = _url_context(step_maps)
    return VertexMetadata(
        grounding=(grounding,) if grounding else (),
        url_context=(url_context,) if url_context else (),
    )
