import json
import re
from bisect import bisect_left
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import partial, reduce
from itertools import accumulate, chain, groupby, islice
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import assert_never

import litellm
from litellm.litellm_core_utils.core_helpers import as_str_mapping, drops_unsupported_params
from litellm.litellm_core_utils.prompt_templates.factory import sanitize_anthropic_tool_use_id
from litellm.llms.anthropic.messages.native.utils import normalize_native_messages
from litellm.types.utils import TokenCountMessagesFormat

_CONVERTIBLE_BLOCK_TYPES: Final = MappingProxyType(
    {
        "user": frozenset({"text", "image", "document", "tool_result"}),
        "assistant": frozenset({"text", "tool_use", "thinking", "redacted_thinking"}),
    }
)
_SOURCED_BLOCK_TYPES: Final = frozenset({"image", "document"})
_TOOL_RESULT_BLOCK_TYPES: Final = frozenset({"text", "image", "document"})
_WORD_OR_CHARACTER: Final = re.compile(r"\w+|\W")
_WORD: Final = re.compile(r"\w+")
_MAX_ALIGNED_CELLS: Final = 4096
_COMPARED_WORDS: Final = 256
_MAX_DIFFED_TOKENS: Final = 8000
_ANCHOR_SCORE: Final = 3.0


_JSON_MESSAGES: Final = TypeAdapter(tuple[Mapping[str, JsonValue], ...])


def _json_messages(messages: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, JsonValue], ...]:
    try:
        return _JSON_MESSAGES.validate_python(messages)
    except ValidationError:
        return ()


def _source_label(block_type: str, source: JsonValue) -> str | None:
    if not isinstance(source, dict):
        return f"{block_type} without a source"
    match source.get("type"):
        case "base64" if source.get("data"):
            return None
        case "url" if source.get("url"):
            return None
        case "base64" | "url" as source_type:
            return f"{block_type} with an empty {source_type} source"
        case str() as source_type:
            return f"{block_type} with a {source_type} source"
        case _:
            return f"{block_type} with an untyped source"


def _block_label(block: JsonValue, convertible: frozenset[str]) -> str | None:
    if not isinstance(block, dict):
        return None
    block_type: Final = block.get("type")
    if not isinstance(block_type, str):
        return "a block without a type"
    if block_type not in convertible:
        return block_type
    return _source_label(block_type, block.get("source")) if block_type in _SOURCED_BLOCK_TYPES else None


def _tool_result_labels(block: JsonValue) -> Iterator[str]:
    if not isinstance(block, dict) or block.get("type") != "tool_result":
        return iter(())
    inner: Final = block.get("content")
    if not isinstance(inner, list):
        return iter(())
    return (
        f"{label} inside a tool_result"
        for item in inner
        if (label := _block_label(item, _TOOL_RESULT_BLOCK_TYPES)) is not None
    )


def _content_labels(block: JsonValue, convertible: frozenset[str]) -> Iterator[str]:
    label: Final = _block_label(block, convertible)
    return chain(() if label is None else (label,), _tool_result_labels(block))


def _message_labels(message: Mapping[str, JsonValue]) -> Iterator[str]:
    role: Final = message.get("role")
    convertible: Final = _CONVERTIBLE_BLOCK_TYPES.get(role) if isinstance(role, str) else None
    content: Final = message.get("content")
    if convertible is None or not isinstance(content, list):
        return iter(())
    return chain.from_iterable(_content_labels(block, convertible) for block in content)


def unconvertible_anthropic_blocks(messages: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    normalized: Final = _json_messages(normalize_native_messages(messages))
    return tuple(sorted(set(chain.from_iterable(_message_labels(message) for message in normalized))))


def unconvertible_blocks_error(
    messages: Sequence[Mapping[str, object]] | None, messages_format: TokenCountMessagesFormat
) -> str | None:
    match messages_format:
        case "openai":
            return None
        case "anthropic":
            labels: Final = unconvertible_anthropic_blocks(messages or ())
            return (
                "count_tokens can't convert these content blocks for this model, so the count would leave them out: "
                + ", ".join(labels)
                if labels
                else None
            )
        case _:
            assert_never(messages_format)


_OBJECTS: Final = TypeAdapter(tuple[object, ...])


def _list_items(value: object) -> tuple[object, ...]:
    return _OBJECTS.validate_python(value) if isinstance(value, list) else ()


_ScanPart: TypeAlias = Callable[[object], object]
_Carried: TypeAlias = Callable[[object], bool]


def _scannable_text(text: object) -> str:
    return json.dumps(text, default=str, ensure_ascii=False)


def _is_non_string_text(block: object) -> bool:
    match as_str_mapping(block):
        case {"type": "text", "text": str()}:
            return False
        case {"type": "text", "text": _}:
            return True
        case _:
            return False


def tool_result_text_part(block: object) -> str | None:
    match block:
        case {"type": "text", "text": str(text)}:
            return text
        case {"type": "text", "text": object()}:
            return None
        case {"type": "text"}:
            return ""
        case _:
            return None


def tool_result_text_parts(tool_result: Mapping[str, object]) -> tuple[str, ...]:
    parts: Final = map(tool_result_text_part, _list_items(tool_result.get("content")))
    return tuple(part for part in parts if part is not None)


def _user_content_blocks(message: object) -> tuple[object, ...]:
    match as_str_mapping(message):
        case {"role": "user", "content": content}:
            return _list_items(content)
        case _:
            return ()


def _with_user_blocks(message: object, transform: Callable[[object], object]) -> object:
    match as_str_mapping(message):
        case {"role": "user", "content": list()} as user_message:
            return {**user_message, "content": [transform(block) for block in _user_content_blocks(user_message)]}
        case _:
            return message


def _non_string_text_errors(label: str, content: object) -> Iterator[str]:
    return (
        f"{label} content.{index}.text: Input should be a valid string"
        for index, text_block in enumerate(_list_items(content))
        if _is_non_string_text(text_block)
    )


def _tool_result_text_errors(block: object) -> Iterator[str]:
    match block:
        case {"type": "tool_result", "tool_use_id": object() as tool_use_id, "content": object() as content}:
            return _non_string_text_errors(f"tool_result {tool_use_id}", content)
        case {"type": "tool_result", "content": object() as content}:
            return _non_string_text_errors("tool_result", content)
        case _:
            return iter(())


def malformed_tool_result_text_error(messages: Iterable[object]) -> str | None:
    blocks: Final = chain.from_iterable(map(_user_content_blocks, messages))
    return next(chain.from_iterable(map(_tool_result_text_errors, blocks)), None)


def malformed_tool_result_count_error(
    messages: Iterable[Mapping[str, object]] | None,
    messages_format: TokenCountMessagesFormat,
    deployment: Mapping[str, object] | None,
) -> str | None:
    if messages_format != "anthropic" or deployment_drops_params(deployment):
        return None
    return malformed_tool_result_text_error(messages or ())


def _scannable_part(part: object) -> object:
    match as_str_mapping(part):
        case {"type": "text", "text": str()}:
            return part
        case {"type": "text", "text": text} as text_block:
            return {**text_block, "text": _scannable_text(text)}
        case _:
            return part


def _scannable_block(block: object) -> object:
    match as_str_mapping(block):
        case {"type": "tool_result", "content": list()} as tool_result:
            return {
                **tool_result,
                "content": [_scannable_part(part) for part in _list_items(tool_result.get("content"))],
            }
        case _:
            return block


def _scannable_messages(messages: object) -> object:
    items: Final = _list_items(messages)
    return [_with_user_blocks(message, _scannable_block) for message in items] if items else messages


def scannable_tool_result_request(request: object) -> object:
    match as_str_mapping(request):
        case {"messages": messages} as request_fields:
            return {**request_fields, "messages": _scannable_messages(messages)}
        case None:
            return request
        case request_fields:
            return {**request_fields}


def _content_blocks(content: object) -> tuple[object, ...]:
    match content:
        case str() if content:
            return ({"type": "text", "text": content},)
        case _:
            return _list_items(content)


def _rewritten_part(original: object, rewritten: object) -> object:
    match (as_str_mapping(original), as_str_mapping(rewritten)):
        case ({"type": "text"} as text_block, {"type": "text", "text": str(text)}):
            unchanged: Final = text == tool_result_text_part(_scannable_part(original))
            return original if unchanged else {**text_block, "text": text}
        case ({"type": str(block_type), "source": source}, {"source": rewritten_source}) if (
            block_type in _SOURCED_BLOCK_TYPES and source == rewritten_source
        ):
            return original
        case _:
            return rewritten


def _carried_tool_result_part(scan_part: _ScanPart, part: object) -> bool:
    return scan_part(_scannable_part(part)) is not None


_UnitKey: TypeAlias = tuple[str, str]
_Position: TypeAlias = tuple[int, int]
_Contents: TypeAlias = tuple[tuple[object, ...], ...]
_Opcode: TypeAlias = tuple[str, int, int, int, int]


@dataclass(frozen=True, slots=True)
class _Unit:
    key: _UnitKey
    positions: tuple[_Position, ...]
    blocks: tuple[object, ...]


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)


def _unit_key(block: object) -> _UnitKey:
    match as_str_mapping(block):
        case {"type": "text"}:
            return ("text", tool_result_text_part(_scannable_part(block)) or "")
        case {"type": "image" | "document", "source": source}:
            return ("source", _canonical_json(source))
        case {"type": "tool_use", "id": str(tool_use_id)}:
            return ("tool_use", sanitize_anthropic_tool_use_id(tool_use_id))
        case {"type": "tool_result", "tool_use_id": str(tool_use_id)}:
            return ("tool_result", sanitize_anthropic_tool_use_id(tool_use_id))
        case {"type": str(block_type)}:
            return (block_type, _canonical_json(block))
        case _:
            return ("", _canonical_json(block))


def _unit_kind(unit: _Unit) -> str:
    return unit.key[0]


def _unit_order(role: str, unit: _Unit) -> tuple[int, _Position]:
    match (role, _unit_kind(unit)):
        case ("assistant", "thinking" | "redacted_thinking") | ("user", "tool_result"):
            return (0, unit.positions[0])
        case ("assistant", "tool_use"):
            return (2, unit.positions[0])
        case _:
            return (1, unit.positions[0])


_Entry: TypeAlias = tuple[_Position, object, _UnitKey]


def _text_group(texts: tuple[_Entry, ...]) -> _Unit:
    return _Unit(
        ("text", "".join(key[1] for _, _, key in texts)),
        tuple(position for position, _, _ in texts),
        tuple(block for _, block, _ in texts),
    )


def _message_units(index: int, blocks: tuple[object, ...], role: str, carried: _Carried) -> tuple[_Unit, ...]:
    entries: Final = tuple(
        ((index, position), block, _unit_key(block)) for position, block in enumerate(blocks) if carried(block)
    )
    groups_texts: Final = role == "assistant"
    singles: Final = tuple(
        _Unit(key, (position,), (block,)) for position, block, key in entries if not (groups_texts and key[0] == "text")
    )
    texts: Final = tuple(entry for entry in entries if groups_texts and entry[2][0] == "text")
    grouped: Final = (_text_group(texts),) if texts else ()
    return tuple(sorted((*singles, *grouped), key=partial(_unit_order, role)))


def _units(contents: _Contents, role: str, carried: _Carried) -> tuple[_Unit, ...]:
    return tuple(
        chain.from_iterable(_message_units(index, blocks, role, carried) for index, blocks in enumerate(contents))
    )


def _always_carried(block: object) -> bool:
    return True


def _kind_indexes(units: tuple[_Unit, ...], start: int, stop: int, kind: str) -> Iterator[int]:
    return (index for index in range(start, stop) if _unit_kind(units[index]) == kind)


@dataclass(frozen=True, slots=True)
class _Alignment:
    score: float
    pairs: tuple[tuple[int, int], ...]


_UNALIGNED: Final = _Alignment(0.0, ())


def _unit_words(unit: _Unit) -> frozenset[str]:
    if _unit_kind(unit) != "text":
        return frozenset()
    return frozenset(match.group() for match in islice(_WORD.finditer(unit.key[1]), _COMPARED_WORDS))


def _pair_score(unit: _Unit, words: frozenset[str], partner: _Unit, partner_words: frozenset[str]) -> float | None:
    if unit.key == partner.key:
        return _ANCHOR_SCORE
    if _unit_kind(unit) != _unit_kind(partner):
        return None
    return 1.0 + len(words & partner_words) / max(len(words), len(partner_words), 1)


def _best_alignment(
    up: _Alignment, left: _Alignment, diagonal: _Alignment, score: float | None, pair: tuple[int, int]
) -> _Alignment:
    kept: Final = up if up.score >= left.score else left
    if score is None or diagonal.score + score <= kept.score:
        return kept
    return _Alignment(diagonal.score + score, (*diagonal.pairs, pair))


def _best_pairs(units: tuple[_Unit, ...], rewritten_units: tuple[_Unit, ...]) -> tuple[tuple[int, int], ...]:
    words: Final = tuple(map(_unit_words, units))
    rewritten_words: Final = tuple(map(_unit_words, rewritten_units))

    def row(previous: tuple[_Alignment, ...], index: int) -> tuple[_Alignment, ...]:
        def cell(left: _Alignment, column: int) -> _Alignment:
            score: Final = _pair_score(units[index], words[index], rewritten_units[column], rewritten_words[column])
            return _best_alignment(previous[column + 1], left, previous[column], score, (index, column))

        return tuple(accumulate(range(len(rewritten_units)), cell, initial=previous[0]))

    first_row: Final = tuple(_UNALIGNED for _ in range(len(rewritten_units) + 1))
    return reduce(row, range(len(units)), first_row)[-1].pairs


def _opcode_pairs(
    original: tuple[_Unit, ...], rewritten: tuple[_Unit, ...], opcode: _Opcode
) -> tuple[tuple[int, int], ...]:
    match opcode:
        case ("equal", i1, i2, j1, j2):
            return tuple(zip(range(i1, i2), range(j1, j2)))
        case ("replace", i1, i2, j1, j2) if (i2 - i1) * (j2 - j1) <= _MAX_ALIGNED_CELLS:
            return tuple((i1 + i, j1 + j) for i, j in _best_pairs(original[i1:i2], rewritten[j1:j2]))
        case ("replace", i1, i2, j1, j2):
            kinds: Final = frozenset(map(_unit_kind, original[i1:i2]))
            return tuple(
                chain.from_iterable(
                    zip(_kind_indexes(original, i1, i2, kind), _kind_indexes(rewritten, j1, j2, kind))
                    for kind in sorted(kinds)
                )
            )
        case _:
            return ()


def _unit_pairs(units: tuple[_Unit, ...], rewritten_units: tuple[_Unit, ...]) -> tuple[tuple[int, int], ...]:
    if len(units) * len(rewritten_units) <= _MAX_ALIGNED_CELLS:
        return _best_pairs(units, rewritten_units)
    opcodes: Final = SequenceMatcher(
        None, tuple(unit.key for unit in units), tuple(unit.key for unit in rewritten_units), autojunk=False
    ).get_opcodes()
    return tuple(chain.from_iterable(_opcode_pairs(units, rewritten_units, opcode) for opcode in opcodes))


def _insertion_anchor(index: int, originals_by_rewritten: Mapping[int, int], end: int) -> int:
    later: Final = sorted(rewritten for rewritten in originals_by_rewritten if rewritten > index)
    return originals_by_rewritten[later[0]] if later else end


def _common_prefix_length(left: str, right: str) -> int:
    return bisect_left(range(min(len(left), len(right))), True, key=lambda size: left[: size + 1] != right[: size + 1])


def _common_edges(left: str, right: str) -> tuple[int, int]:
    prefix: Final = _common_prefix_length(left, right)
    suffix: Final = min(_common_prefix_length(left[::-1], right[::-1]), len(left) - prefix, len(right) - prefix)
    return prefix, suffix


@dataclass(frozen=True, slots=True)
class _Span:
    start: int
    end: int
    rewritten_start: int
    rewritten_end: int
    prefix: int
    suffix: int


def _span(original: str, rewritten: str, start: int, end: int, rewritten_start: int, rewritten_end: int) -> _Span:
    prefix, suffix = _common_edges(original[start:end], rewritten[rewritten_start:rewritten_end])
    return _Span(start, end, rewritten_start, rewritten_end, prefix, suffix)


def _token_offsets(text: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    tokens: Final = tuple(_WORD_OR_CHARACTER.findall(text))
    return tokens, (0, *accumulate(map(len, tokens)))


def _changed_spans(
    original: str, rewritten: str, prefix: int, original_end: int, rewritten_end: int
) -> Iterator[_Span]:
    original_tokens, original_offsets = _token_offsets(original[prefix:original_end])
    rewritten_tokens, rewritten_offsets = _token_offsets(rewritten[prefix:rewritten_end])
    if len(original_tokens) + len(rewritten_tokens) > _MAX_DIFFED_TOKENS:
        return iter((_span(original, rewritten, prefix, original_end, prefix, rewritten_end),))
    return (
        _span(
            original,
            rewritten,
            prefix + original_offsets[i1],
            prefix + original_offsets[i2],
            prefix + rewritten_offsets[j1],
            prefix + rewritten_offsets[j2],
        )
        for _, i1, i2, j1, j2 in SequenceMatcher(None, original_tokens, rewritten_tokens).get_opcodes()
    )


def _diff_spans(original: str, rewritten: str) -> tuple[_Span, ...]:
    original_text: Final = original.rstrip()
    rewritten_text: Final = rewritten.rstrip()
    prefix, suffix = _common_edges(original_text, rewritten_text)
    original_end: Final = len(original_text) - suffix
    rewritten_end: Final = len(rewritten_text) - suffix
    return (
        _Span(0, prefix, 0, prefix, prefix, 0),
        *_changed_spans(original, rewritten, prefix, original_end, rewritten_end),
        _Span(original_end, len(original_text), rewritten_end, len(rewritten_text), suffix, 0),
        _span(original, rewritten, len(original_text), len(original), len(rewritten_text), len(rewritten)),
    )


def _mapped_offset(offset: int, spans: tuple[_Span, ...], ends: tuple[int, ...]) -> int:
    span: Final = spans[bisect_left(ends, offset)]
    if offset - span.start <= span.prefix:
        return span.rewritten_start + offset - span.start
    if span.end - offset <= span.suffix:
        return span.rewritten_end - (span.end - offset)
    return span.rewritten_end - span.suffix


def _split_text(texts: tuple[str, ...], rewritten: str) -> tuple[str, ...]:
    joined: Final = "".join(texts)
    if joined == rewritten:
        return texts
    spans: Final = _diff_spans(joined, rewritten)
    span_ends: Final = tuple(span.end for span in spans)
    boundaries: Final = accumulate(map(len, texts[:-1]))
    ends: Final = (*(_mapped_offset(boundary, spans, span_ends) for boundary in boundaries), len(rewritten))
    return tuple(rewritten[start:end] for start, end in zip((0, *ends[:-1]), ends))


def _split_text_part(block: object, text: str) -> tuple[object, ...]:
    return (_rewritten_part(block, {"type": "text", "text": text}),) if text else ()


def _paired_blocks(unit: _Unit, partner: _Unit | None) -> _Contents:
    if partner is None:
        return tuple(() for _ in unit.blocks)
    if len(unit.blocks) == len(partner.blocks):
        return tuple((_rewritten_part(block, rewritten),) for block, rewritten in zip(unit.blocks, partner.blocks))
    texts: Final = _split_text(tuple(_unit_key(block)[1] for block in unit.blocks), partner.key[1])
    return tuple(map(_split_text_part, unit.blocks, texts))


def _unit_emissions(
    unit: _Unit, partner: _Unit | None, inserted: tuple[object, ...]
) -> Iterator[tuple[_Position, tuple[object, ...]]]:
    outputs: Final = _paired_blocks(unit, partner)
    return zip(unit.positions, ((*inserted, *outputs[0]), *outputs[1:]))


def _aligned_contents(original: _Contents, rewritten: _Contents, role: str, carried: _Carried) -> _Contents:
    units: Final = _units(original, role, carried)
    rewritten_units: Final = _units(rewritten, role, _always_carried)
    partners: Final = MappingProxyType(dict(_unit_pairs(units, rewritten_units)))
    originals_by_rewritten: Final = MappingProxyType({rewritten: original for original, rewritten in partners.items()})
    insertions: Final = tuple(
        (_insertion_anchor(index, originals_by_rewritten, len(units)), index)
        for index in range(len(rewritten_units))
        if index not in originals_by_rewritten
    )

    def inserted_at(anchor: int) -> tuple[object, ...]:
        return tuple(chain.from_iterable(rewritten_units[index].blocks for at, index in insertions if at == anchor))

    emissions: Final = MappingProxyType(
        dict(
            chain.from_iterable(
                _unit_emissions(
                    unit, rewritten_units[partners[index]] if index in partners else None, inserted_at(index)
                )
                for index, unit in enumerate(units)
            )
        )
    )
    contents: Final = tuple(
        tuple(chain.from_iterable(emissions.get((index, position), (block,)) for position, block in enumerate(blocks)))
        for index, blocks in enumerate(original)
    )
    return (*contents[:-1], (*contents[-1], *inserted_at(len(units))))


def _merged_content(original: object, rewritten: object, carried: _Carried) -> object:
    original_blocks: Final = _list_items(original)
    if not isinstance(original, list):
        return rewritten
    return list(_aligned_contents((original_blocks,), (_content_blocks(rewritten),), "user", carried)[0])


def _restored_tool_result(
    rewritten: Mapping[str, object], original: Mapping[str, object], carried: _Carried
) -> Mapping[str, object]:
    return {
        **original,
        **rewritten,
        "content": _merged_content(original.get("content"), rewritten.get("content"), carried),
    }


_ToolResultEntry: TypeAlias = tuple[str, Mapping[str, object]]


def _tool_result_entry(block: object) -> tuple[_ToolResultEntry, ...]:
    match as_str_mapping(block):
        case {"type": "tool_result", "tool_use_id": str(tool_use_id)} as tool_result:
            return ((sanitize_anthropic_tool_use_id(tool_use_id), tool_result),)
        case _:
            return ()


def _entry_tool_use_id(entry: _ToolResultEntry) -> str:
    return entry[0]


def _entry_tool_result(entry: _ToolResultEntry) -> Mapping[str, object]:
    return entry[1]


def _restored_block(
    block: object, originals: Mapping[str, Iterator[Mapping[str, object]]], carried: _Carried
) -> object:
    match as_str_mapping(block):
        case {"type": "tool_result", "tool_use_id": str(tool_use_id)} as tool_result if (
            sanitize_anthropic_tool_use_id(tool_use_id) in originals
        ):
            original: Final = next(originals[sanitize_anthropic_tool_use_id(tool_use_id)], tool_result)
            return _restored_tool_result(tool_result, original, carried)
        case _:
            return block


def _restored_tool_results(rewritten: Sequence[object], original: object, scan_part: _ScanPart) -> Sequence[object]:
    blocks: Final = chain.from_iterable(map(_user_content_blocks, _list_items(original)))
    entries: Final = tuple(sorted(chain.from_iterable(map(_tool_result_entry, blocks)), key=_entry_tool_use_id))
    if not entries:
        return rewritten
    originals_in_order: Final = MappingProxyType(
        {
            tool_use_id: iter(tuple(map(_entry_tool_result, group)))
            for tool_use_id, group in groupby(entries, key=_entry_tool_use_id)
        }
    )
    restore: Final = partial(
        _restored_block, originals=originals_in_order, carried=partial(_carried_tool_result_part, scan_part)
    )
    return [_with_user_blocks(message, restore) for message in rewritten]


_Turn: TypeAlias = tuple[Mapping[str, object], ...]
_Run: TypeAlias = tuple[str | None, tuple[object, ...]]


@dataclass(frozen=True, slots=True)
class _ClientMessage:
    message: Mapping[str, object]
    scanned_role: str | None
    turn_role: str | None


_Segment: TypeAlias = tuple[str | None, tuple[_ClientMessage, ...]]


def _turn_role(message: object) -> str | None:
    match as_str_mapping(message):
        case {"role": str(role)} if role in _CONVERTIBLE_BLOCK_TYPES:
            return role
        case _:
            return None


def _runs(messages: Iterable[object]) -> tuple[_Run, ...]:
    return tuple((role, tuple(group)) for role, group in groupby(messages, key=_turn_role))


def _turn_roles(runs: Iterable[_Run]) -> tuple[str, ...]:
    return tuple(role for role, _ in runs if role is not None)


def _turn(messages: Iterable[object]) -> _Turn:
    return tuple(filter(None, map(as_str_mapping, messages)))


def _carried_turn_block(role: str, scan_part: _ScanPart, block: object) -> bool:
    match as_str_mapping(block):
        case {"type": str(block_type)} if block_type not in _CONVERTIBLE_BLOCK_TYPES[role]:
            return False
        case {"type": str(block_type)} if block_type in _SOURCED_BLOCK_TYPES:
            return scan_part(block) is not None
        case _:
            return True


def _scanned_role(message: Mapping[str, object], scan_part: _ScanPart) -> str | None:
    role: Final = _turn_role(message)
    content: Final = message.get("content")
    blocks: Final = _list_items(content)
    if role is None or not isinstance(content, list):
        return role
    carried: Final = partial(_carried_turn_block, role, scan_part)
    return role if any(map(carried, blocks)) else None


def _later_turn_role(later: str | None, scanned_role: str | None) -> str | None:
    return scanned_role or later


def _client_messages(original: object, scan_part: _ScanPart) -> tuple[_ClientMessage, ...]:
    messages: Final = tuple(message for message in _turn(_list_items(original)) if _turn_role(message) is not None)
    scanned_roles: Final = tuple(_scanned_role(message, scan_part) for message in messages)
    upcoming_roles: Final = tuple(accumulate(reversed(scanned_roles), _later_turn_role))
    return tuple(map(_ClientMessage, messages, scanned_roles, reversed(upcoming_roles)))


def _client_turn_role(client_message: _ClientMessage) -> str | None:
    return client_message.turn_role


def _segments(client_messages: Iterable[_ClientMessage]) -> tuple[_Segment, ...]:
    return tuple((role, tuple(group)) for role, group in groupby(client_messages, key=_client_turn_role))


def _segment_messages(segment: _Segment) -> _Turn:
    return tuple(client_message.message for client_message in segment[1])


def _restored_message(client_message: _ClientMessage, merged: Iterator[tuple[object, ...]]) -> tuple[object, ...]:
    if client_message.scanned_role is None:
        return (client_message.message,)
    content: Final = next(merged)
    return ({**client_message.message, "content": list(content)},) if content else ()


def _message_contents(messages: Iterable[Mapping[str, object]]) -> _Contents:
    return tuple(_content_blocks(message.get("content")) for message in messages)


def _restored_segment(
    client_messages: tuple[_ClientMessage, ...], role: str, rewritten: _Turn, scan_part: _ScanPart
) -> tuple[object, ...]:
    scanned: Final = tuple(message.message for message in client_messages if message.scanned_role is not None)
    contents: Final = _aligned_contents(
        _message_contents(scanned), _message_contents(rewritten), role, partial(_carried_turn_block, role, scan_part)
    )
    merged: Final = iter(contents)
    return tuple(chain.from_iterable(_restored_message(message, merged) for message in client_messages))


def _restored_run(run: _Run, segments: Iterator[_Segment], scan_part: _ScanPart) -> tuple[object, ...]:
    match run:
        case (str(role), messages):
            return _restored_segment(next(segments)[1], role, _turn(messages), scan_part)
        case (_, messages):
            return messages


def _restored_turns(rewritten: Sequence[object], original: object, scan_part: _ScanPart) -> Sequence[object]:
    segments: Final = _segments(_client_messages(original, scan_part))
    segment_roles: Final = tuple(role for role, _ in segments if role is not None)
    if segments and not segment_roles:
        return list(_segment_messages(segments[0]))
    rewritten_runs: Final = _runs(rewritten)
    if _turn_roles(rewritten_runs) != segment_roles:
        return rewritten
    remaining: Final = iter(segments)
    restored: Final = tuple(chain.from_iterable(_restored_run(run, remaining, scan_part) for run in rewritten_runs))
    return [*restored, *chain.from_iterable(map(_segment_messages, remaining))]


def restore_client_blocks(rewritten: Sequence[object], original: object, scan_part: _ScanPart) -> Sequence[object]:
    return _restored_turns(_restored_tool_results(rewritten, original, scan_part), original, scan_part)


def _without_non_string_text(block: object) -> object:
    match as_str_mapping(block):
        case {"type": "tool_result", "content": list()} as tool_result:
            parts: Final = _list_items(tool_result.get("content"))
            return {**tool_result, "content": [part for part in parts if not _is_non_string_text(part)]}
        case _:
            return block


def _without_non_string_texts(
    message: dict[str, object],  # mutable-ok: AnthropicMessagesRequest.messages is typed list[dict]
) -> dict[str, object]:  # mutable-ok: AnthropicMessagesRequest.messages is typed list[dict]
    match message:
        case {"role": "user", "content": list()}:
            return {**message, "content": [_without_non_string_text(block) for block in _user_content_blocks(message)]}
        case _:
            return message


def drop_non_string_tool_result_texts(
    messages: Sequence[dict[str, object]],  # mutable-ok: AnthropicMessagesRequest.messages is typed list[dict]
    drop_params: object,
) -> Sequence[dict[str, object]]:  # mutable-ok: AnthropicMessagesRequest.messages is typed list[dict]
    if not drops_unsupported_params(drop_params) or malformed_tool_result_text_error(messages) is None:
        return messages
    return [_without_non_string_texts(message) for message in messages]


def deployment_drops_params(deployment: Mapping[str, object] | None) -> bool:
    match deployment:
        case {"litellm_params": {"drop_params": object() as drop_params}}:
            return drops_unsupported_params(drop_params)
        case _:
            return drops_unsupported_params(None)


def reject_malformed_tool_result_text(
    messages: Iterable[Mapping[str, object]], model: str, forwarded_kwargs: Mapping[str, object]
) -> None:
    if drops_unsupported_params(forwarded_kwargs.get("drop_params")):
        return
    error: Final = malformed_tool_result_text_error(messages)
    if error is None:
        return
    provider: Final = forwarded_kwargs.get("custom_llm_provider")
    raise litellm.BadRequestError(
        message=error, model=model, llm_provider=provider if isinstance(provider, str) else ""
    )
