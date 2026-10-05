from collections.abc import Sequence
from typing import (
    Final,
)

from .payload import (
    MAX_JSON_DEPTH,
    MutableRequest,
    RequestTooDeep,
    Slot,
    SlotSink,
    as_array,
    as_object,
    collect,
    collect_entry,
    collect_json_leaves,
    collect_text_parts,
    is_container,
    read_list,
)

# Roles whose text the application author wrote and the caller never sees. Their
# PII is still redacted outbound, but it is not restorable from the reply.
PRIVILEGED_ROLES: Final = frozenset({"system", "developer"})

# How far a tool_result chain is followed. Real payloads nest one or two deep; the
# bound is what stops a crafted one from becoming an unbounded walk. A request that
# nests deeper is refused rather than forwarded, because text past the bound would
# otherwise reach the provider unredacted.
MAX_CONTENT_DEPTH: Final = 8

# JSON Schema keywords whose value has to reach the model or a validator verbatim, so the
# schema walk leaves them alone: types, formats, patterns, references and
# required-property lists. Everything else is scanned.
SCHEMA_STRUCTURAL_KEYWORDS: Final = frozenset(
    (
        "type",
        "format",
        "pattern",
        "required",
        "dependentRequired",
        "propertyOrdering",
        "discriminator",
        "contentEncoding",
        "contentMediaType",
        "$ref",
        "$id",
        "$schema",
        "$anchor",
        "$dynamicRef",
        "$dynamicAnchor",
        "$recursiveRef",
        "$recursiveAnchor",
        "$vocabulary",
    )
)

# Keywords holding JSON values rather than schemas: every string in them is collected,
# whatever the keys around it are called.
SCHEMA_VALUE_KEYWORDS: Final = frozenset(("examples", "default"))

# Keywords holding the literal values the model must reproduce. These go to the CALLER's
# vault, not the privileged one: the model emits the stand-in in its tool arguments or
# structured output, and restoring the reply turns it back into the value the schema
# allows, so the call still routes. In the non-restorable vault it would come back as a
# stand-in no validator accepts.
SCHEMA_LITERAL_KEYWORDS: Final = frozenset(("enum", "const"))

# Keywords whose value maps names to subschemas. Their keys are property names, not
# keywords, so a property called `type` or `enum` is walked like any other subschema.
SCHEMA_MAP_KEYWORDS: Final = frozenset(
    ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas", "dependencies")
)


def collect_prompt(data: MutableRequest, slots: SlotSink) -> None:
    """The Completions API sends its text in `prompt`, and its tail in `suffix`."""
    collect(data, "suffix", slots)
    prompt: Final = data.get("prompt")
    if isinstance(prompt, str):
        collect(data, "prompt", slots)
        return
    prompt_object: Final = as_object(prompt)
    if prompt_object is not None:
        # A Responses API PromptObject. `variables` are substituted into the stored
        # prompt on the provider side, so they are caller text. `id` and `version`
        # identify which prompt to use and must arrive unchanged. A variable is a string
        # or a typed input such as `{"type": "input_text", "text": ...}`.
        variables: Final = as_object(prompt_object.get("variables"))
        if variables is not None:
            for name in tuple(variables):
                collect(variables, name, slots)
                typed = as_object(variables[name])
                if typed is not None:
                    collect(typed, "text", slots)
        return
    entries: Final = as_array(prompt)
    if entries is None:
        return
    for index in range(len(entries)):
        collect_entry(entries, index, slots)


def collect_content(container: MutableRequest, slots: SlotSink) -> None:
    """Collects `content`, a string or a list of typed parts.

    An Anthropic tool_result nests its own content, so this has to descend. It walks
    with an explicit stack and a depth bound rather than by recursion: the nesting is
    caller controlled, and an unbounded descent is a JSON bomb. Content nested past the
    bound raises `RequestTooDeep` rather than being skipped.
    """
    # Walked in document order: the shield maps its replies back by position, so the
    # order spans are collected in is part of the contract.
    pending: Final[list[tuple[MutableRequest, int]]] = [(container, 0)]  # mutable-ok: local queue, never escapes.
    cursor = 0  # rebind-ok: advances through the queue.
    while cursor < len(pending):
        node, depth = pending[cursor]
        cursor += 1
        content = node.get("content")
        if isinstance(content, str):
            collect(node, "content", slots)
            continue
        if depth >= MAX_CONTENT_DEPTH and content:
            raise RequestTooDeep("content")
        for item in as_array(content) or ():
            part = as_object(item)
            if part is None:
                continue
            # Image and audio parts have no text and fall through untouched.
            collect(part, "text", slots)
            if part.get("type") == "tool_use":
                # A replayed Anthropic tool call. Its `input` is a JSON object rather than
                # a string, so a value can sit at any depth -- the reply side walks the
                # same leaves when it restores one. Its own JSON bound applies, not the
                # content one, and past it the request is refused.
                collect_json_leaves(part.get("input"), slots, strict=True)
            source = as_object(part.get("source")) if part.get("type") == "document" else None
            if source is not None:
                # An Anthropic document carries text inline: a `text` source holds it in
                # `data`, a `content` source as a string or blocks, walked like any other
                # content. Base64, URL and file sources are binary or remote, and pass
                # untouched. Its `title` and `context` are caller text too.
                collect(part, "title", slots)
                collect(part, "context", slots)
                if source.get("type") == "text":
                    collect(source, "data", slots)
                elif source.get("type") == "content":
                    pending.append((source, depth + 1))
            if "content" in part:
                pending.append((part, depth + 1))


def collect_participant_name(message: MutableRequest, slots: SlotSink) -> None:
    """Redacts `name` where it identifies a person, never where it names a function.

    On a user or assistant turn `name` is the participant, which is personal data.
    On a tool or function turn the same field carries the function's name and has
    to reach the provider unchanged, or the call no longer routes.
    """
    if message.get("role") in ("tool", "function"):
        return
    collect(message, "name", slots)


def collect_tool_arguments(message: MutableRequest, slots: SlotSink) -> None:
    """Tool arguments carry the values a user asked the model to act on."""
    for tool_call in read_list(message, "tool_calls"):
        tool_call_object = as_object(tool_call)
        function = as_object(tool_call_object.get("function")) if tool_call_object is not None else None
        if function is not None:
            collect(function, "arguments", slots)
    legacy: Final = as_object(message.get("function_call"))
    if legacy is not None:
        collect(legacy, "arguments", slots)


def collect_system(data: MutableRequest, slots: SlotSink) -> None:
    """Anthropic's /v1/messages carries its system prompt at the top level."""
    system: Final = data.get("system")
    if isinstance(system, str):
        collect(data, "system", slots)
        return
    collect_text_parts(data, "system", slots)


def collect_responses_fields(data: MutableRequest, slots: SlotSink, privileged: SlotSink) -> None:
    """The Responses API sends text outside `messages`, in `instructions` and `input`.

    `instructions` is written by the application, not by the caller, so it is
    collected into the privileged sink; `input` is the caller's own text, except for
    system and developer items in it, which go to the privileged sink like their Chat
    counterparts.
    """
    collect(data, "instructions", privileged)
    request_input: Final = data.get("input")
    if isinstance(request_input, str):
        collect(data, "input", slots)
        return
    entries: Final = as_array(request_input)
    if entries is None:
        return
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            # The embeddings and moderations shape: `input` as an array of strings.
            collect_entry(entries, index, slots)
            continue
        item = as_object(entry)
        if item is None:
            continue
        collect_content(item, privileged if item.get("role") in PRIVILEGED_ROLES else slots)
        # A function_call item holds `arguments`; a function_call_output holds `output`,
        # as a string or as a list of input_text parts. A custom_tool_call holds `input`
        # and a code_interpreter_call `code` -- the fields the reply side restores.
        collect(item, "arguments", slots)
        collect(item, "output", slots)
        collect_text_parts(item, "output", slots)
        collect(item, "input", slots)
        collect(item, "code", slots)
        # A replayed reasoning item carries the model's summary of its own reasoning,
        # which quotes whatever the conversation contained.
        collect_text_parts(item, "summary", slots)


def collect_tool_definitions(data: MutableRequest, slots: SlotSink, privileged: SlotSink) -> None:
    """Tool definitions are application-authored free text bound for the provider.

    A tool's description and the free text in its parameter schema are where callers put
    examples and customer context, so they carry PII as often as a prompt does. They are
    collected into the privileged sink, like a system prompt: redacted outbound, and never
    restorable from the reply. `enum` and `const` values are the exception, and go to the
    caller's vault -- see `SCHEMA_LITERAL_KEYWORDS`. Names and types are left as sent.

    Covers Chat `tools[].function`, the legacy `functions[]`, and the flat tool shape the
    Responses API and Anthropic share, whose schema is `parameters` or `input_schema`.
    """
    for key in ("tools", "functions"):
        for entry in as_array(data.get(key)) or ():
            tool = as_object(entry)
            if tool is None:
                continue
            function = as_object(tool.get("function"))
            for holder in (tool, function) if function is not None else (tool,):
                collect(holder, "description", privileged)
                collect_schema_text(holder.get("parameters"), slots, privileged)
                collect_schema_text(holder.get("input_schema"), slots, privileged)


def collect_schema_text(schema: object, slots: SlotSink, privileged: SlotSink) -> None:
    """Collects the text in a JSON Schema, at any depth.

    Scan by default: every string is collected except under the keywords in
    `SCHEMA_STRUCTURAL_KEYWORDS`, whose values must go out verbatim. A list of keywords
    *to* collect would leak every one it forgot -- draft-07 `dependencies`, a `$comment`,
    a vendor `x-` extension -- which is how this walk started out. Free text goes to the
    privileged sink; `enum` / `const` literals go to the caller's, so the model's use of
    them is restored.

    Structure matters in two places. Under `properties` and the other name -> subschema
    maps, keys are property names rather than keywords, so a property called `type` is a
    subschema to walk, not a keyword to skip. And `examples` / `default` hold JSON values,
    so all their strings are collected whatever the keys around them are called. Nested
    past `MAX_JSON_DEPTH`, the request is refused.
    """
    pending: Final[list[tuple[object, int]]] = [(schema, 0)]  # mutable-ok: local walk stack.
    while pending:
        node, depth = pending.pop()
        if depth > MAX_JSON_DEPTH:
            if is_container(node) and node:
                raise RequestTooDeep("schema")
            continue
        entries = as_array(node)
        if entries is not None:
            for index, item in enumerate(entries):
                collect_entry(entries, index, privileged)
                if is_container(item):
                    pending.append((item, depth + 1))
            continue
        schema_object = as_object(node)
        if schema_object is None:
            continue
        for keyword, value in tuple(schema_object.items()):
            if keyword in SCHEMA_STRUCTURAL_KEYWORDS:
                continue
            subschemas = as_object(value) if keyword in SCHEMA_MAP_KEYWORDS else None
            if keyword in SCHEMA_LITERAL_KEYWORDS:
                collect(schema_object, keyword, slots)
                collect_json_leaves(value, slots, strict=True)
            elif keyword in SCHEMA_VALUE_KEYWORDS:
                collect(schema_object, keyword, privileged)
                collect_json_leaves(value, privileged, strict=True)
            elif subschemas is not None:
                pending.extend((child, depth + 1) for child in subschemas.values())
            elif isinstance(value, str):
                collect(schema_object, keyword, privileged)
            elif is_container(value):
                pending.append((value, depth + 1))


def collect_output_contracts(data: MutableRequest, slots: SlotSink, privileged: SlotSink) -> None:
    """Text the caller sends to shape the reply rather than to prompt it.

    A predicted output (`prediction.content`) is the caller's own draft of the answer, so
    it goes with their text: the model largely repeats it, and it has to come back. A
    structured-output schema -- Chat `response_format.json_schema`, Responses
    `text.format` -- is application-authored like a tool schema, so its free text goes
    to the privileged sink, and its names and types stay as sent.
    """
    prediction: Final = as_object(data.get("prediction"))
    if prediction is not None:
        collect(prediction, "content", slots)
        collect_text_parts(prediction, "content", slots)
    response_format: Final = as_object(data.get("response_format"))
    text_options: Final = as_object(data.get("text"))
    for declared in (
        response_format.get("json_schema") if response_format is not None else None,
        text_options.get("format") if text_options is not None else None,
    ):
        wrapper = as_object(declared)
        if wrapper is not None:
            collect(wrapper, "description", privileged)
            collect_schema_text(wrapper.get("schema"), slots, privileged)


def collect_user_locations(data: MutableRequest, privileged: SlotSink) -> None:
    """Web search forwards the user's approximate location, whose `city` and `region`
    are free text and can hold a street address.

    Chat carries it in `web_search_options.user_location.approximate`; the Responses
    and Anthropic web-search tools carry it flat on the tool's `user_location`. Nothing
    restores it from a reply, hence the privileged sink.
    """
    options: Final = data.get("web_search_options")
    tools: Final = as_array(data.get("tools")) or ()
    for declared in (options, *tools):
        holder = as_object(declared)
        location = as_object(holder.get("user_location")) if holder is not None else None
        if location is None:
            continue
        approximate = as_object(location.get("approximate"))
        for container in (location, approximate) if approximate is not None else (location,):
            collect(container, "city", privileged)
            collect(container, "region", privileged)


def collect_end_user_ids(data: MutableRequest, privileged: SlotSink) -> None:
    """`user` and `safety_identifier` are forwarded to the provider and often hold an email.

    Only detected PII is replaced, so an opaque id reaches the provider unchanged. LiteLLM's
    own end-user spend tracking reads the id resolved at authentication, before this hook
    runs, so rewriting the field here does not move spend. Nothing restores these from a
    reply, hence the privileged sink.
    """
    collect(data, "user", privileged)
    collect(data, "safety_identifier", privileged)


def locate_request_texts(
    data: MutableRequest,
) -> tuple[Sequence[Slot], Sequence[Slot]]:
    """Finds every redactable span, split by whether the caller can see it.

    Anything missed here reaches the provider in the clear while the guardrail
    still reports as enabled, so the walk covers every request shape that
    carries text.

    The split exists because the response is restored against one vault only.
    Server-authored spans -- system and developer turns, Anthropic's top-level
    `system`, the Responses API `instructions`, tool and output schemas -- go into a
    vault nothing is ever restored against, so a caller who gets the model to
    echo one of their placeholders back receives the placeholder, not the value
    behind it. End-user identifiers go there too: nothing in a reply needs them.

    Tool *results* stay on the caller's side deliberately. The model reads them in
    order to answer, so it can already repeat anything in them; restoring the
    placeholder gives the caller the answer they would have had without this
    guardrail, and an agent that reads a file and quotes an address from it needs
    that address back.

    `extra_body` is walked the same way as the request itself. LiteLLM merges it over
    the transformed request just before sending, so a field there -- `input`,
    `messages`, `system` -- replaces the redacted one on the wire.
    """
    slots: Final[SlotSink] = []
    privileged: Final[SlotSink] = []
    for payload in (data, as_object(data.get("extra_body"))):
        if payload is None:
            continue
        for entry in read_list(payload, "messages"):
            message = as_object(entry)
            if message is not None:
                sink = privileged if message.get("role") in PRIVILEGED_ROLES else slots
                collect_content(message, sink)
                collect_participant_name(message, sink)
                collect_tool_arguments(message, sink)
        collect_responses_fields(payload, slots, privileged)
        collect_prompt(payload, slots)
        collect_system(payload, privileged)
        collect_tool_definitions(payload, slots, privileged)
        collect_output_contracts(payload, slots, privileged)
        collect_user_locations(payload, privileged)
        collect_end_user_ids(payload, privileged)
    return tuple(slots), tuple(privileged)
