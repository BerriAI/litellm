"""DataFog PII guardrail: offline, in-process PII redaction and blocking.

Uses the `datafog` library (https://github.com/DataFog/datafog-python) to
detect and redact PII locally — no external service, no sidecar, no network
calls. Scans run in microseconds per request, so the guardrail can sit on
every call without a latency budget.

- ``pre_call``: scans request messages. ``redact`` (default) replaces
  findings with ``[TYPE_N]`` tokens before the request leaves the gateway;
  ``block`` rejects the request with HTTP 400.
- ``during_call``: same as pre_call, but runs in parallel with the LLM call
  (block only — content cannot be modified mid-flight).
- ``post_call``: redacts findings from model responses before they reach
  the client.

Defaults to the high-precision entity types (EMAIL, PHONE, CREDIT_CARD,
SSN). Noisier types (IP_ADDRESS, DOB, ZIP, DE_* locale entities) are opt-in
via ``datafog_entity_types`` because version strings, dates, and short
numbers saturate technical text.

Errors and block messages report entity type counts only; matched PII is
never echoed into logs, exceptions, or proxy responses. Engine exceptions
are re-raised with ``from None`` and logged by type name only, since their
messages can embed the text being scanned.
"""

import json
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any, Literal

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth

DEFAULT_ENTITY_TYPES = ["EMAIL", "PHONE", "CREDIT_CARD", "SSN"]

VALID_ACTIONS = {"redact", "block"}
VALID_FAIL_POLICIES = {"open", "closed"}


def _redact_text(text: str, entity_types: list[str], locales: list[str] | None) -> tuple[str, dict[str, int]]:
    """Redact ``text``; return (redacted_text, counts per entity type)."""
    try:
        import datafog  # pyright: ignore[reportMissingTypeStubs]  # datafog does not ship pyright stubs
    except ImportError as exc:
        raise ImportError(
            "The DataFog guardrail requires the `datafog` package. Install it with: pip install datafog"
        ) from exc

    result = datafog.redact(text, engine="regex", entity_types=entity_types, locales=locales)
    counts: dict[str, int] = {}
    for entity in result.entities:
        counts[entity.type] = counts.get(entity.type, 0) + 1
    return result.redacted_text, counts


def _summary(counts: dict[str, int]) -> str:
    return ", ".join(f"{etype} x{n}" for etype, n in sorted(counts.items()))


def _merge_counts(total_counts: dict[str, int], counts: dict[str, int]) -> None:
    for etype, n in counts.items():
        total_counts[etype] = total_counts.get(etype, 0) + n


def _get_field(container: Any, field_name: str) -> Any:
    if isinstance(container, dict):
        return container.get(field_name)
    return getattr(container, field_name, None)


def _set_field(container: Any, field_name: str, value: Any) -> None:
    if isinstance(container, dict):
        container[field_name] = value
    else:
        setattr(container, field_name, value)


class DataFogGuardrail(CustomGuardrail):
    """Offline PII guardrail powered by the datafog library."""

    def __init__(
        self,
        datafog_action: Literal["redact", "block"] | None = "redact",
        datafog_entity_types: list[str] | None = None,
        datafog_locales: list[str] | None = None,
        datafog_fail_policy: Literal["open", "closed"] | None = "open",
        **kwargs: Any,
    ) -> None:
        action = datafog_action or "redact"
        fail_policy = datafog_fail_policy or "open"
        if action not in VALID_ACTIONS:
            raise ValueError(f"datafog_action must be one of: {sorted(VALID_ACTIONS)}")
        if fail_policy not in VALID_FAIL_POLICIES:
            raise ValueError(f"datafog_fail_policy must be one of: {sorted(VALID_FAIL_POLICIES)}")
        self.action = action
        self.entity_types = datafog_entity_types or DEFAULT_ENTITY_TYPES
        self.locales = datafog_locales
        self.fail_policy = fail_policy
        super().__init__(**kwargs)

    def _process_content(self, content: Any) -> tuple[Any, dict[str, int]]:
        """Redact a message content value (str or list of content parts)."""
        counts: dict[str, int] = {}
        if isinstance(content, str):
            return _redact_text(content, self.entity_types, self.locales)
        if isinstance(content, list):
            new_parts = []
            skipped_parts = 0
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    redacted, part_counts = _redact_text(part["text"], self.entity_types, self.locales)
                    new_parts.append({**part, "text": redacted})
                    for etype, n in part_counts.items():
                        counts[etype] = counts.get(etype, 0) + n
                else:
                    new_parts.append(part)
                    skipped_parts += 1
            if skipped_parts:
                verbose_proxy_logger.debug(
                    "DataFog guardrail: %d non-text content parts not scanned",
                    skipped_parts,
                )
            return new_parts, counts
        return content, counts

    def _process_tool_payload(self, payload: Any) -> tuple[Any, dict[str, int]]:
        """Redact text inside supported tool/function argument payloads."""
        if isinstance(payload, str):
            return _redact_text(payload, self.entity_types, self.locales)
        if not isinstance(payload, (list, dict)):
            return payload, {}

        counts: dict[str, int] = {}
        changed = False
        new_payload = list(payload) if isinstance(payload, list) else dict(payload)
        stack = [(payload, new_payload)]
        seen = {id(payload)}

        while stack:
            original_container, new_container = stack.pop()
            if isinstance(original_container, list):
                items = enumerate(original_container)
            else:
                items = original_container.items()

            for key, value in items:
                if isinstance(value, str):
                    redacted, value_counts = _redact_text(value, self.entity_types, self.locales)
                    if value_counts:
                        new_container[key] = redacted
                        changed = True
                        _merge_counts(counts, value_counts)
                elif isinstance(value, list):
                    if id(value) in seen:
                        continue
                    seen.add(id(value))
                    new_value = list(value)
                    new_container[key] = new_value
                    stack.append((value, new_value))
                elif isinstance(value, dict):
                    if id(value) in seen:
                        continue
                    seen.add(id(value))
                    new_value = dict(value)
                    new_container[key] = new_value
                    stack.append((value, new_value))

        return (new_payload if changed else payload), counts

    def _scan_argument_mapping(self, mapping: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
        counts: dict[str, int] = {}
        new_mapping = dict(mapping)
        changed = False
        for field_name in ("arguments", "input"):
            payload = mapping.get(field_name)
            if not isinstance(payload, (str, list, dict)):
                continue
            new_payload, payload_counts = self._process_tool_payload(payload)
            if payload_counts:
                new_mapping[field_name] = new_payload
                changed = True
                _merge_counts(counts, payload_counts)
        return (new_mapping if changed else mapping), counts

    def _scan_argument_container(self, container: Any, *, redact: bool) -> dict[str, int]:
        """Scan supported argument fields on a dict/object container."""
        counts: dict[str, int] = {}
        for field_name in ("arguments", "input"):
            payload = _get_field(container, field_name)
            if not isinstance(payload, (str, list, dict)):
                continue
            new_payload, payload_counts = self._process_tool_payload(payload)
            if payload_counts:
                if redact:
                    _set_field(container, field_name, new_payload)
                _merge_counts(counts, payload_counts)
        return counts

    def _scan_request_tool_calls(self, tool_calls: Any) -> tuple[Any, dict[str, int]]:
        if not isinstance(tool_calls, list):
            return tool_calls, {}
        counts: dict[str, int] = {}
        new_tool_calls = []
        changed = False
        for tool_call in tool_calls:
            if isinstance(tool_call, dict) and isinstance(tool_call.get("function"), dict):
                new_function, function_counts = self._scan_argument_mapping(tool_call["function"])
                if function_counts:
                    new_tool_calls.append({**tool_call, "function": new_function})
                    changed = True
                    _merge_counts(counts, function_counts)
                    continue
            new_tool_calls.append(tool_call)
        return (new_tool_calls if changed else tool_calls), counts

    def _scan_message(self, message: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
        counts: dict[str, int] = {}
        new_message = dict(message)
        changed = False

        if "content" in message:
            new_content, content_counts = self._process_content(message["content"])
            if content_counts:
                new_message["content"] = new_content
                changed = True
                _merge_counts(counts, content_counts)

        function_call = message.get("function_call")
        if isinstance(function_call, dict):
            new_function_call, function_counts = self._scan_argument_mapping(function_call)
            if function_counts:
                new_message["function_call"] = new_function_call
                changed = True
                _merge_counts(counts, function_counts)

        new_tool_calls, tool_counts = self._scan_request_tool_calls(message.get("tool_calls"))
        if tool_counts:
            new_message["tool_calls"] = new_tool_calls
            changed = True
            _merge_counts(counts, tool_counts)

        return (new_message if changed else message), counts

    def _scan_responses_input_item(self, item: Any) -> tuple[Any, dict[str, int]]:
        if isinstance(item, str):
            return _redact_text(item, self.entity_types, self.locales)
        if not isinstance(item, dict):
            return item, {}

        counts: dict[str, int] = {}
        new_item, message_counts = self._scan_message(item)
        changed = bool(message_counts)
        _merge_counts(counts, message_counts)

        argument_item, argument_counts = self._scan_argument_mapping(new_item)
        if argument_counts:
            new_item = argument_item
            changed = True
            _merge_counts(counts, argument_counts)

        for field_name in ("text", "output"):
            payload = new_item.get(field_name)
            if not isinstance(payload, (str, list, dict)):
                continue
            new_payload, payload_counts = self._process_tool_payload(payload)
            if payload_counts:
                if not changed:
                    new_item = dict(new_item)
                new_item[field_name] = new_payload
                changed = True
                _merge_counts(counts, payload_counts)

        return (new_item if changed else item), counts

    def _scan_responses_api_input(self, input_value: Any) -> tuple[Any, dict[str, int]]:
        if isinstance(input_value, str):
            return _redact_text(input_value, self.entity_types, self.locales)
        if not isinstance(input_value, list):
            return input_value, {}

        counts: dict[str, int] = {}
        new_items = []
        changed = False
        for item in input_value:
            new_item, item_counts = self._scan_responses_input_item(item)
            new_items.append(new_item)
            if item_counts:
                changed = True
                _merge_counts(counts, item_counts)

        return (new_items if changed else input_value), counts

    def _scan_top_level_prompt_fields(self, data: dict) -> tuple[dict, dict[str, int]]:
        counts: dict[str, int] = {}
        new_data = data
        for field_name in ("instructions", "system"):
            if field_name not in data:
                continue
            new_value, field_counts = self._process_content(data[field_name])
            if field_counts:
                if new_data is data:
                    new_data = dict(data)
                new_data[field_name] = new_value
                _merge_counts(counts, field_counts)
        return new_data, counts

    def _scan_schema_fields(self, data: dict) -> tuple[dict, dict[str, int]]:
        counts: dict[str, int] = {}
        new_data = data
        for field_name in ("tools", "functions", "response_format"):
            schema = data.get(field_name)
            if not isinstance(schema, (list, dict)):
                continue
            new_schema, field_counts = self._process_tool_payload(schema)
            if field_counts:
                if new_data is data:
                    new_data = dict(data)
                new_data[field_name] = new_schema
                _merge_counts(counts, field_counts)
        return new_data, counts

    def _process_completion_prompt(self, prompt: Any) -> tuple[Any, dict[str, int]]:
        if isinstance(prompt, str):
            return _redact_text(prompt, self.entity_types, self.locales)
        if not isinstance(prompt, list) or not all(isinstance(item, str) for item in prompt):
            return prompt, {}

        counts: dict[str, int] = {}
        new_prompt = []
        changed = False
        for item in prompt:
            new_item, item_counts = _redact_text(item, self.entity_types, self.locales)
            new_prompt.append(new_item)
            if item_counts:
                changed = True
                _merge_counts(counts, item_counts)
        return (new_prompt if changed else prompt), counts

    def _handle_engine_error(self, exc: Exception) -> None:
        """Apply the fail policy without leaking scanned text.

        Only the exception type is logged or re-raised: engine exception
        messages can embed the text being scanned, so chaining via ``from
        exc`` or interpolating ``str(exc)`` would leak matched PII into
        tracebacks. A missing dependency is a config error and is surfaced
        regardless of fail policy.
        """
        if isinstance(exc, ImportError):
            raise exc
        if self.fail_policy == "closed":
            raise RuntimeError(
                "DataFog guardrail failed and datafog_fail_policy is "
                f"'closed'; rejecting unscanned traffic ({type(exc).__name__})."
            ) from None
        verbose_proxy_logger.warning(
            "DataFog guardrail error (fail-open, traffic unscanned): %s",
            type(exc).__name__,
        )

    def _scan_messages(self, data: dict) -> tuple[dict, dict[str, int]]:
        """Scan/redact all message contents; return (new_data, counts)."""
        messages = data.get("messages")
        if not isinstance(messages, list):
            return data, {}

        total_counts: dict[str, int] = {}
        new_messages = []
        for message in messages:
            if isinstance(message, dict):
                new_message, counts = self._scan_message(message)
                new_messages.append(new_message)
                _merge_counts(total_counts, counts)
            else:
                new_messages.append(message)

        if not total_counts:
            return data, {}
        return {**data, "messages": new_messages}, total_counts

    def _scan_request_data(self, data: dict) -> tuple[dict, dict[str, int]]:
        """Scan/redact supported request payload fields."""
        new_data, total_counts = self._scan_messages(data)

        new_input, input_counts = self._scan_responses_api_input(new_data.get("input"))
        if input_counts:
            new_data = {**new_data, "input": new_input}
            _merge_counts(total_counts, input_counts)

        new_prompt_data, prompt_counts = self._scan_top_level_prompt_fields(new_data)
        if prompt_counts:
            new_data = new_prompt_data
            _merge_counts(total_counts, prompt_counts)

        if "prompt" in new_data:
            new_prompt, completion_prompt_counts = self._process_completion_prompt(new_data["prompt"])
            if completion_prompt_counts:
                new_data = {**new_data, "prompt": new_prompt}
                _merge_counts(total_counts, completion_prompt_counts)

        new_schema_data, schema_counts = self._scan_schema_fields(new_data)
        if schema_counts:
            new_data = new_schema_data
            _merge_counts(total_counts, schema_counts)

        if not total_counts:
            return data, {}
        return new_data, total_counts

    def _scan_text_field(self, container: Any, field_name: str, *, redact: bool) -> dict[str, int]:
        text = _get_field(container, field_name)
        if not isinstance(text, str):
            return {}
        new_text, counts = _redact_text(text, self.entity_types, self.locales)
        if counts and redact:
            _set_field(container, field_name, new_text)
        return counts

    def _scan_response_content(self, container: Any, *, redact: bool) -> dict[str, int]:
        content = _get_field(container, "content")
        if content is None:
            return {}
        new_content, counts = self._process_content(content)
        if counts and redact:
            _set_field(container, "content", new_content)
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "tool_use":
                    input_counts = self._scan_argument_container(part, redact=redact)
                    _merge_counts(counts, input_counts)
        return counts

    def _scan_response_tool_calls(self, message: Any, *, redact: bool) -> dict[str, int]:
        counts: dict[str, int] = {}
        function_call = _get_field(message, "function_call")
        if function_call is not None:
            _merge_counts(counts, self._scan_argument_container(function_call, redact=redact))

        tool_calls = _get_field(message, "tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                function = _get_field(tool_call, "function")
                if function is not None:
                    _merge_counts(counts, self._scan_argument_container(function, redact=redact))
        return counts

    def _scan_response_message(self, message: Any, *, redact: bool) -> dict[str, int]:
        counts = self._scan_response_content(message, redact=redact)
        _merge_counts(counts, self._scan_response_tool_calls(message, redact=redact))
        return counts

    def _scan_responses_api_output(self, response: Any, *, redact: bool) -> dict[str, int]:
        counts: dict[str, int] = {}
        output = _get_field(response, "output")
        if not isinstance(output, list):
            return counts
        for item in output:
            item_type = _get_field(item, "type")
            if item_type == "message":
                _merge_counts(counts, self._scan_response_content(item, redact=redact))
            elif item_type in {"function_call", "custom_tool_call"}:
                _merge_counts(counts, self._scan_argument_container(item, redact=redact))
        return counts

    @staticmethod
    def _stream_group_key(prefix: str, *parts: Any) -> tuple[Any, ...]:
        return (prefix, *parts)

    @staticmethod
    def _decode_sse_text_delta(chunk: bytes | bytearray) -> tuple[Any, str] | None:
        raw = bytes(chunk).decode("utf-8", errors="replace")
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped.startswith("data: "):
                continue
            try:
                data = json.loads(stripped[6:])
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(data, dict) or data.get("type") != "content_block_delta":
                continue
            delta = data.get("delta")
            if isinstance(delta, dict) and delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                return data.get("index"), delta["text"]
        return None

    @staticmethod
    def _replace_sse_text_delta(chunk: bytes | bytearray, new_text: str) -> bytes:
        raw = bytes(chunk).decode("utf-8", errors="replace")
        lines = raw.splitlines(keepends=True)
        for index, line in enumerate(lines):
            line_ending = ""
            line_body = line
            if line.endswith("\r\n"):
                line_ending = "\r\n"
                line_body = line[:-2]
            elif line.endswith("\n"):
                line_ending = "\n"
                line_body = line[:-1]

            leading = line_body[: len(line_body) - len(line_body.lstrip())]
            stripped = line_body.lstrip()
            if not stripped.startswith("data: "):
                continue
            try:
                data = json.loads(stripped[6:])
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(data, dict) or data.get("type") != "content_block_delta":
                continue
            delta = data.get("delta")
            if not isinstance(delta, dict) or delta.get("type") != "text_delta":
                continue
            delta["text"] = new_text
            lines[index] = f"{leading}data: {json.dumps(data, separators=(',', ':'))}{line_ending}"
            break
        return "".join(lines).encode("utf-8")

    def _append_chat_stream_text_refs(
        self,
        chunk: Any,
        refs: list[tuple[tuple[Any, ...], str, Callable[[str], None]]],
    ) -> None:
        choices = _get_field(chunk, "choices")
        if not isinstance(choices, list):
            return

        for choice_index, choice in enumerate(choices):
            choice_text = _get_field(choice, "text")
            if isinstance(choice_text, str) and choice_text:
                key = self._stream_group_key("completion", _get_field(choice, "index") or choice_index)
                refs.append((key, choice_text, lambda new_text, choice=choice: _set_field(choice, "text", new_text)))

            delta = _get_field(choice, "delta")
            if delta is None:
                continue
            content = _get_field(delta, "content")
            if not isinstance(content, str) or not content:
                continue
            key = self._stream_group_key("chat", _get_field(choice, "index") or choice_index)
            refs.append((key, content, lambda new_text, delta=delta: _set_field(delta, "content", new_text)))

    def _append_responses_stream_text_ref(
        self,
        chunk: Any,
        refs: list[tuple[tuple[Any, ...], str, Callable[[str], None]]],
    ) -> None:
        if _get_field(chunk, "type") != "response.output_text.delta":
            return
        delta = _get_field(chunk, "delta")
        if not isinstance(delta, str) or not delta:
            return
        key = self._stream_group_key(
            "responses",
            _get_field(chunk, "item_id"),
            _get_field(chunk, "output_index"),
            _get_field(chunk, "content_index"),
        )
        refs.append((key, delta, lambda new_text, chunk=chunk: _set_field(chunk, "delta", new_text)))

    def _append_anthropic_stream_text_ref(
        self,
        chunk: Any,
        refs: list[tuple[tuple[Any, ...], str, Callable[[str], None]]],
    ) -> None:
        if _get_field(chunk, "type") != "content_block_delta":
            return
        delta = _get_field(chunk, "delta")
        if _get_field(delta, "type") != "text_delta":
            return
        text = _get_field(delta, "text")
        if not isinstance(text, str) or not text:
            return
        key = self._stream_group_key("anthropic", _get_field(chunk, "index"))
        refs.append((key, text, lambda new_text, delta=delta: _set_field(delta, "text", new_text)))

    def _collect_stream_text_refs(self, chunks: list[Any]) -> list[tuple[tuple[Any, ...], str, Callable[[str], None]]]:
        refs: list[tuple[tuple[Any, ...], str, Callable[[str], None]]] = []
        for chunk_index, chunk in enumerate(chunks):
            if isinstance(chunk, (bytes, bytearray)):
                sse_delta = self._decode_sse_text_delta(chunk)
                if sse_delta:
                    content_index, text = sse_delta
                    key = self._stream_group_key("anthropic_sse", content_index)
                    refs.append(
                        (
                            key,
                            text,
                            lambda new_text, chunk_index=chunk_index: chunks.__setitem__(
                                chunk_index,
                                self._replace_sse_text_delta(chunks[chunk_index], new_text),
                            ),
                        )
                    )
                continue

            self._append_chat_stream_text_refs(chunk, refs)
            self._append_responses_stream_text_ref(chunk, refs)
            self._append_anthropic_stream_text_ref(chunk, refs)

        return refs

    def _scan_stream_text_refs(
        self,
        refs: list[tuple[tuple[Any, ...], str, Callable[[str], None]]],
        *,
        redact: bool,
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        refs_by_group: dict[tuple[Any, ...], list[tuple[str, Callable[[str], None]]]] = {}
        for key, text, setter in refs:
            refs_by_group.setdefault(key, []).append((text, setter))

        for grouped_refs in refs_by_group.values():
            stream_text = "".join(text for text, _ in grouped_refs)
            redacted_text, group_counts = _redact_text(stream_text, self.entity_types, self.locales)
            if not group_counts:
                continue
            _merge_counts(counts, group_counts)
            if not redact:
                continue
            first_ref = True
            for _, setter in grouped_refs:
                setter(redacted_text if first_ref else "")
                first_ref = False

        return counts

    def _scan_streaming_chunks(self, chunks: list[Any], *, redact: bool) -> dict[str, int]:
        refs = self._collect_stream_text_refs(chunks)
        if not refs:
            return {}
        return self._scan_stream_text_refs(refs, redact=redact)

    def _raise_block(self, total_counts: dict[str, int]) -> None:
        """Reject with HTTP 400 so the block is classified as a guardrail
        intervention by ``_is_guardrail_intervention`` rather than a
        backend failure, and reaches the client as 400 instead of 500."""
        raise HTTPException(
            status_code=400,
            detail={"error": (f"Violated DataFog PII guardrail policy: request contains {_summary(total_counts)}.")},
        )

    def _record_guardrail_logging(self, data: dict, total_counts: dict[str, int]) -> None:
        """Record the decision into standard guardrail logging."""
        try:
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_json_response=_summary(total_counts),
                request_data=data,
                guardrail_status="guardrail_intervened",
                masked_entity_count=dict(total_counts),
            )
        except Exception:  # noqa: BLE001  # logging failures must not break guardrail enforcement
            verbose_proxy_logger.debug("DataFog guardrail: could not record logging information")

    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: Any,
        data: dict,
        call_type: str,
    ) -> Exception | str | dict | None:
        if self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_call) is not True:
            return data
        try:
            new_data, total_counts = self._scan_request_data(data)
        except Exception as exc:  # noqa: BLE001  # engine failures are handled by the configured fail policy
            self._handle_engine_error(exc)
            return data

        if not total_counts:
            return data

        if self.action == "block":
            self._record_guardrail_logging(data, total_counts)
            self._raise_block(total_counts)
        self._record_guardrail_logging(new_data, total_counts)
        return new_data

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: "UserAPIKeyAuth",
        call_type: str,
    ) -> Any:
        """Block on PII during the parallel call window.

        Content cannot be rewritten mid-flight, so redact mode is a no-op
        here; only block has an effect.
        """
        if self.action != "block":
            return data
        if self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.during_call) is not True:
            return data
        try:
            _, total_counts = self._scan_request_data(data)
        except Exception as exc:  # noqa: BLE001  # engine failures are handled by the configured fail policy
            self._handle_engine_error(exc)
            return data

        if total_counts:
            self._record_guardrail_logging(data, total_counts)
            self._raise_block(total_counts)
        return data

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: "UserAPIKeyAuth",
        response: Any,
    ) -> Any:
        """Redact PII from model responses, or block when action is block.

        Redaction mutates ``response`` in place, deliberately: post_call
        guardrails share the response object, and an unredacted copy
        escaping through another callback would defeat the purpose.
        """
        if self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.post_call) is not True:
            return response
        response_counts: dict[str, int] = {}
        try:
            redact = self.action != "block"
            choices = getattr(response, "choices", None)
            if choices:
                for choice in choices:
                    message = getattr(choice, "message", None)
                    if message is not None:
                        _merge_counts(response_counts, self._scan_response_message(message, redact=redact))
                    _merge_counts(response_counts, self._scan_text_field(choice, "text", redact=redact))
            _merge_counts(response_counts, self._scan_responses_api_output(response, redact=redact))
            if isinstance(response, dict):
                _merge_counts(response_counts, self._scan_response_content(response, redact=redact))
        except Exception as exc:  # noqa: BLE001  # engine failures are handled by the configured fail policy
            self._handle_engine_error(exc)
            return response
        if response_counts:
            self._record_guardrail_logging(data, response_counts)
            if self.action == "block":
                self._raise_block(response_counts)
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[Any, None]:
        if self.should_run_guardrail(data=request_data, event_type=GuardrailEventHooks.post_call) is not True:
            async for chunk in response:
                yield chunk
            return

        chunks = []
        async for chunk in response:
            chunks.append(chunk)

        stream_counts: dict[str, int] = {}
        try:
            stream_counts = self._scan_streaming_chunks(chunks, redact=self.action != "block")
        except Exception as exc:  # noqa: BLE001  # engine failures are handled by the configured fail policy
            self._handle_engine_error(exc)
            for chunk in chunks:
                yield chunk
            return

        if stream_counts:
            self._record_guardrail_logging(request_data, stream_counts)
            if self.action == "block":
                self._raise_block(stream_counts)

        for chunk in chunks:
            yield chunk

    @staticmethod
    def get_config_model() -> type | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.datafog import (
            DataFogGuardrailConfigModel,
        )

        return DataFogGuardrailConfigModel
