import functools
from collections.abc import Iterator, Sequence
from types import MappingProxyType
from typing import Final

from .payload import (
    JsonArray,
    JsonObject,
    Slot,
    Write,
    as_array,
    as_object,
    field_writer,
    item_writer,
)

RESPONSES_OPAQUE_TYPES: Final = frozenset(
    {
        "reasoning",
        "compaction",
        "compaction_trigger",
        "computer_call_output",
        "image_generation_call",
        "item_reference",
        "mcp_list_tools",
        "tool_search_call",
        "tool_search_output",
        "additional_tools",
    }
)
"""Responses `input` items relayed whole: an opaque or binary payload, or a tool or
pointer definition with no user text. Parity with the PrivAiTe gateway scrubber."""

RESPONSES_DATA_FIELDS: Final = MappingProxyType(
    {
        "computer_call": ("action", "actions"),
        "local_shell_call": ("action",),
        "shell_call": ("action",),
        "web_search_call": ("action",),
        "apply_patch_call": ("operation",),
        "file_search_call": ("queries", "results"),
        "code_interpreter_call": ("code", "outputs"),
        "program": ("code",),
        "program_output": ("result",),
    }
)
"""Responses items whose user data sits in known structured fields, walked leaf by leaf."""

BINARY_PART_TYPES: Final = frozenset(
    {
        "input_image",
        "input_file",
        "input_audio",
        "image",
        "output_image",
        "computer_screenshot",
    }
)
"""Parts whose payload is base64 or a file id. Scrubbing one would corrupt it without
removing anything a text detector can find."""

GENERIC_TEXT_FIELDS: Final = ("output", "arguments", "input", "text", "reason")
"""Text-bearing fields of an unknown Responses item. Every one is scanned, since an
mcp_call carries both `arguments` and `output`."""


def leaf(text: object, write: Write) -> Iterator[Slot]:
    if isinstance(text, str) and text:
        yield text, write


def content_slots(content: object, write: Write) -> Iterator[Slot]:
    """A role message's `content`: a bare string, or parts whose `text` and `refusal` carry user text."""
    if isinstance(content, str):
        yield from leaf(content, write)
        return
    parts: Final = as_array(content)
    if parts is None:
        return
    for index, part in enumerate(parts):
        if isinstance(part, str):
            yield from leaf(part, item_writer(parts, index))
            continue
        fields = as_object(part)
        if fields is None or fields.get("type") in BINARY_PART_TYPES:
            continue
        for name in ("text", "refusal"):
            yield from leaf(fields.get(name), field_writer(fields, name))


def data_slots(value: object, write: Write) -> Iterator[Slot]:
    """Every string leaf of a tool payload, binary parts relayed whole and object keys never scanned.

    The payload is caller controlled, so it is walked with an explicit stack rather than
    by recursion. Children are pushed reversed, so leaves come out depth first.
    """
    pending: Final[list[tuple[object, Write]]] = [(value, write)]  # mutable-ok: local walk stack, never escapes
    while pending:
        node, write_back = pending.pop()
        if isinstance(node, str):
            yield from leaf(node, write_back)
            continue
        items = as_array(node)
        if items is not None:
            pending.extend((items[index], item_writer(items, index)) for index in range(len(items) - 1, -1, -1))
            continue
        fields = as_object(node)
        if fields is not None and fields.get("type") not in BINARY_PART_TYPES:
            pending.extend((child, field_writer(fields, key)) for key, child in reversed(fields.items()))


def _named_data_slots(fields: JsonObject, names: Sequence[str]) -> Iterator[Slot]:
    for name in names:
        if fields.get(name) is not None:
            yield from data_slots(fields[name], field_writer(fields, name))


def _typed_item_slots(fields: JsonObject, item_type: str) -> Iterator[Slot] | None:
    """The text of an item whose user data sits in a type-specific field, or None for an unknown shape."""
    if item_type == "function_call" and isinstance(fields.get("arguments"), str):
        return leaf(fields["arguments"], field_writer(fields, "arguments"))
    if item_type == "custom_tool_call" and isinstance(fields.get("input"), str):
        return leaf(fields["input"], field_writer(fields, "input"))
    data_fields: Final = RESPONSES_DATA_FIELDS.get(item_type)
    if data_fields is not None:
        return _named_data_slots(fields, data_fields)
    if item_type.endswith("_output") and "output" in fields:
        return data_slots(fields["output"], field_writer(fields, "output"))
    return None


def _generic_item_slots(fields: JsonObject) -> Iterator[Slot]:
    for name in GENERIC_TEXT_FIELDS:
        yield from leaf(fields.get(name), field_writer(fields, name))
    content: Final = fields.get("content")
    if isinstance(content, str) or as_array(content) is not None:
        yield from content_slots(content, field_writer(fields, "content"))


def _responses_item_slots(items: JsonArray, index: int) -> Iterator[Slot]:
    item: Final = items[index]
    if isinstance(item, str):
        yield from leaf(item, item_writer(items, index))
        return
    fields: Final = as_object(item)
    if fields is None:
        return
    if "role" in fields and "content" in fields:
        yield from content_slots(fields["content"], field_writer(fields, "content"))
        return
    declared_type: Final = fields.get("type")
    item_type: Final = declared_type if isinstance(declared_type, str) else ""
    if item_type in RESPONSES_OPAQUE_TYPES:
        return
    typed: Final = _typed_item_slots(fields, item_type)
    yield from typed if typed is not None else _generic_item_slots(fields)


def _write_top_level(data: JsonObject, key: str, value: str) -> None:
    """Writes a top-level string field, and the copy of it in the proxy's request-body snapshot.

    The proxy snapshots the body by shallow copy before the hooks run. A nested list or
    dict is shared with that snapshot, but a top-level string is not, so without the
    second write the original would stay in proxy_server_request.body.
    """
    data[key] = value  # rebind-ok: the pre-call hook contract rewrites the request in place
    request: Final = as_object(data.get("proxy_server_request"))
    body: Final = as_object(request.get("body")) if request is not None else None
    if body is not None and key in body:
        body[key] = value


def top_level_writer(data: JsonObject, key: str) -> Write:
    return functools.partial(_write_top_level, data, key)


def responses_input_slots(data: JsonObject) -> Iterator[Slot]:
    request_input: Final = data.get("input")
    if isinstance(request_input, str):
        yield from leaf(request_input, top_level_writer(data, "input"))
        return
    items: Final = as_array(request_input)
    if items is None:
        return
    for index in range(len(items)):
        yield from _responses_item_slots(items, index)


def prompt_variables(data: JsonObject) -> JsonObject | None:
    """Responses prompt-template variables, which carry user data. A string `prompt` is /v1/completions."""
    prompt: Final = as_object(data.get("prompt"))
    return as_object(prompt.get("variables")) if prompt is not None else None


def prompt_variable_slots(data: JsonObject) -> Iterator[Slot]:
    variables: Final = prompt_variables(data)
    if variables is None:
        return
    for key, value in variables.items():
        yield from data_slots(value, field_writer(variables, key))


def has_aux_fields(data: JsonObject) -> bool:
    """Whether the request carries a field scanned by `aux_field_slots`."""
    prediction: Final = as_object(data.get("prediction"))
    if prediction is not None and "content" in prediction:
        return True
    web_search: Final = as_object(data.get("web_search_options"))
    if web_search is not None and "user_location" in web_search:
        return True
    suffix: Final = data.get("suffix")
    if isinstance(suffix, str) and suffix:
        return True
    prompt: Final = data.get("prompt")
    return (isinstance(prompt, str) and bool(prompt)) or bool(as_array(prompt))


def aux_field_slots(data: JsonObject) -> Iterator[Slot]:
    """Request text outside messages and input that LiteLLM forwards verbatim: chat
    `prediction.content`, `web_search_options.user_location`, and the completions `suffix`
    and `prompt`. A tokenized (integer array) prompt passes through unscanned."""
    prediction: Final = as_object(data.get("prediction"))
    if prediction is not None and prediction.get("content") is not None:
        yield from data_slots(prediction["content"], field_writer(prediction, "content"))
    web_search: Final = as_object(data.get("web_search_options"))
    if web_search is not None and web_search.get("user_location") is not None:
        yield from data_slots(web_search["user_location"], field_writer(web_search, "user_location"))
    yield from leaf(data.get("suffix"), top_level_writer(data, "suffix"))
    prompt: Final = data.get("prompt")
    if isinstance(prompt, str):
        yield from leaf(prompt, top_level_writer(data, "prompt"))
        return
    parts: Final = as_array(prompt)
    if parts is None:
        return
    for index, part in enumerate(parts):
        yield from leaf(part, item_writer(parts, index))
