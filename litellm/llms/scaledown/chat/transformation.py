import json
import time
import uuid
from collections.abc import Mapping, Sequence
from functools import reduce
from typing import TYPE_CHECKING, Final, NoReturn

import httpx
from pydantic import JsonValue, TypeAdapter

from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse, Usage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
JSON_MESSAGES: Final = TypeAdapter(list[dict[str, JsonValue]])

DEFAULT_API_BASE: Final = "https://api.scaledown.xyz"

DOMAIN_MODELS: Final = frozenset({"classify", "extract", "summarize", "compress"})

DECISIONS_MODELS: Final = frozenset({"decisions"})

DECISIONS_UPSTREAM_MODEL: Final = "classify-1"

NATIVE_PATHS: Final[Mapping[str, str]] = {
    "classify": "/classify",
    "extract": "/extract",
    "summarize": "/summarization/abstractive",
    "compress": "/compress/raw/",
}

DECISION_QUESTION_TYPES: Final = frozenset({"choice", "noul", "score"})

EXTRA_BODY_KEYS: Final[Mapping[str, frozenset[str]]] = {
    "extract": frozenset({"threshold", "top_n"}),
    "summarize": frozenset(),
    "compress": frozenset({"compression_rate"}),
    "classify": frozenset({"labels"}),
    "decisions": frozenset({"questions"}),
}

# Set by the LiteLLM router and proxy on every call; handled by LiteLLM, never sent upstream.
ROUTER_PARAMS: Final = frozenset({"max_retries", "stream_options"})

SCORE_MIN_LEVELS: Final = 2
SCORE_MAX_LEVELS: Final = 10

MAX_ENTITIES: Final = 1000
MAX_REF_CHAIN: Final = 32
MAX_SCHEMA_DEPTH: Final = 32


class ScaleDownError(BaseLLMException):
    pass


def _reject(message: str, status_code: int = 400) -> NoReturn:
    """The single place this adapter raises its own errors."""
    raise ScaleDownError(status_code=status_code, message=message)


class ScaleDownChatConfig(BaseConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "scaledown"

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: BaseConfig signature
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: BaseConfig signature
        optional_params: dict[str, JsonValue],  # mutable-ok: BaseConfig signature
        litellm_params: dict[str, JsonValue],  # mutable-ok: BaseConfig signature
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, JsonValue]:  # mutable-ok: BaseConfig signature
        resolved_key: Final = api_key or get_secret_str("SCALEDOWN_API_KEY")
        if resolved_key is None:
            _reject("Missing ScaleDown API key. Set SCALEDOWN_API_KEY or pass api_key to the call.", 401)
        if not api_key and api_base is not None and _root(api_base) != _trusted_root():
            _reject(
                "A custom api_base needs its own api_key: the SCALEDOWN_API_KEY from the environment is only "
                "sent to the default host or the host in SCALEDOWN_API_BASE."
            )
        return {**headers, "x-api-key": resolved_key, "content-type": "application/json"}

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, JsonValue],  # mutable-ok: BaseConfig signature
        litellm_params: dict[str, JsonValue],  # mutable-ok: BaseConfig signature
        stream: bool | None = None,
    ) -> str:
        base: Final = _root(api_base or _trusted_root())
        operation: Final = _operation(model)
        if operation in DECISIONS_MODELS:
            return f"{base}/v1/scaledown"
        return f"{base}{NATIVE_PATHS[operation]}"

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: BaseConfig signature
        operation: Final = _operation(model)
        if operation == "extract":
            return ["response_format", "stream"]
        if operation == "summarize":
            return ["max_tokens", "max_completion_tokens", "stream"]
        return ["stream"]

    def map_openai_params(
        self,
        non_default_params: dict[str, JsonValue],  # mutable-ok: BaseConfig signature
        optional_params: dict[str, JsonValue],  # mutable-ok: BaseConfig signature
        model: str,
        drop_params: bool,
    ) -> dict[str, JsonValue]:  # mutable-ok: BaseConfig signature
        supported: Final = frozenset(self.get_supported_openai_params(model))
        unsupported: Final = sorted(key for key in non_default_params if key not in supported | ROUTER_PARAMS)
        if unsupported and not drop_params:
            _reject(
                f"ScaleDown model '{model}' does not support {unsupported}. "
                f"Supported: {sorted(supported)}. Set litellm.drop_params=True to ignore."
            )
        mapped: Final = {
            ("max_tokens" if key == "max_completion_tokens" else key): value
            for key, value in non_default_params.items()
            if key in supported
        }
        return {**optional_params, **mapped}

    def should_fake_stream(
        self, model: str | None, stream: bool | None, custom_llm_provider: str | None = None
    ) -> bool:
        return bool(stream)

    def transform_extra_body(
        self,
        extra_body: Mapping[str, object],
        request: Mapping[str, object],
        model: str,
        litellm_params: Mapping[str, object],
    ) -> Mapping[str, JsonValue]:
        # extra_body is merged after guardrails ran on the messages, so it may only add options.
        # Anything that carries prompt text or picks the model is refused.
        options: Final = JSON_OBJECT.validate_python(extra_body)
        operation: Final = _operation(model)
        allowed: Final = EXTRA_BODY_KEYS[operation]
        unexpected: Final = sorted(set(options) - allowed)
        if unexpected:
            allowance: Final = f"may only set {sorted(allowed)}" if allowed else "is not accepted"
            _reject(
                f"extra_body for scaledown/{operation} {allowance}, got {unexpected}. "
                "Text and instructions must be passed as messages."
            )
        if operation == "compress":
            return {"scaledown": {"rate": options["compression_rate"]}} if "compression_rate" in options else {}
        if "questions" in options:
            _validate_questions(options["questions"])
        if "labels" in options:
            _validate_labels(options["labels"])
        return dict(options)

    def sign_request(
        self,
        headers: dict[str, str],  # mutable-ok: BaseConfig signature
        optional_params: dict[str, JsonValue],  # mutable-ok: BaseConfig signature
        request_data: dict[str, JsonValue],  # mutable-ok: BaseConfig signature
        api_base: str,
        api_key: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        fake_stream: bool | None = None,
    ) -> tuple[dict[str, str], bytes | None]:  # mutable-ok: BaseConfig signature
        # Runs after extra_body is merged, so this is where a decisions body is known to be complete.
        if model is not None and _operation(model) in DECISIONS_MODELS:
            _require_complete_decisions(request_data)
        if model is not None and _operation(model) == "classify":
            _validate_labels(request_data.get("labels"))
        return headers, None

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: BaseConfig signature
        optional_params: dict[str, JsonValue],  # mutable-ok: BaseConfig signature
        litellm_params: dict[str, JsonValue],  # mutable-ok: BaseConfig signature
        headers: dict[str, str],  # mutable-ok: BaseConfig signature
    ) -> dict[str, JsonValue]:  # mutable-ok: BaseConfig signature
        checked_messages: Final = JSON_MESSAGES.validate_python(messages)
        operation: Final = _operation(model)
        _validate_text_input(operation, checked_messages, optional_params)
        if operation == "classify":
            return {
                "text": _last_user_text(operation, checked_messages),
                **({"labels": optional_params["labels"]} if "labels" in optional_params else {}),
            }
        if operation in DECISIONS_MODELS:
            return dict(_decisions_request(checked_messages, optional_params))
        if operation == "extract":
            return dict(_extract_request(checked_messages, optional_params))
        if operation == "summarize":
            return dict(_summarize_request(checked_messages, optional_params))
        return dict(_compress_request(checked_messages, optional_params))

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict[str, JsonValue],  # mutable-ok: BaseConfig signature
        messages: list[AllMessageValues],  # mutable-ok: BaseConfig signature
        optional_params: dict[str, JsonValue],  # mutable-ok: BaseConfig signature
        litellm_params: dict[str, JsonValue],  # mutable-ok: BaseConfig signature
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        try:
            raw: Final = JSON_OBJECT.validate_json(raw_response.content)
        except ValueError:
            _reject(
                f"ScaleDown returned a non-JSON response: {raw_response.text[:500]}",
                raw_response.status_code if raw_response.is_error else 502,
            )

        operation: Final = _operation(model)
        if operation in DECISIONS_MODELS:
            return _decisions_response(operation, raw, model_response)
        return _native_response(operation, raw, model_response, request_data.get("entities"))

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | httpx.Headers,  # mutable-ok: BaseConfig signature
    ) -> BaseLLMException:
        return ScaleDownError(status_code=status_code, message=error_message, headers=headers)


def _root(api_base: str) -> str:
    return api_base.rstrip("/").removesuffix("/v1")


def _trusted_root() -> str:
    return _root(get_secret_str("SCALEDOWN_API_BASE") or DEFAULT_API_BASE)


def _operation(model: str) -> str:
    operation: Final = model.split("/", 1)[1] if "/" in model else model
    if operation not in DOMAIN_MODELS | DECISIONS_MODELS:
        _reject(f"Unknown ScaleDown model '{model}'. Choose from {sorted(DOMAIN_MODELS | DECISIONS_MODELS)}.")
    return operation


def _text_of(message: Mapping[str, JsonValue]) -> str | None:
    content: Final = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: Final = [
            str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(parts) if parts else None
    return None


def _validate_text_input(
    operation: str, messages: Sequence[Mapping[str, JsonValue]], optional_params: Mapping[str, JsonValue]
) -> None:
    if any(key in optional_params for key in ("state", "document", "document_mime_type", "text")):
        _reject("ScaleDown accepts text through messages only. Native state and document inputs are not supported.")
    for message in messages:
        if operation in {"classify", "decisions", "extract"} and message.get("role") in {"system", "developer"}:
            _reject(
                f"ScaleDown {operation} does not support system messages. Put instructions in the field definitions."
            )
        if _has_non_text_content(message):
            _reject("ScaleDown currently supports text only. Image and document inputs are not supported.")


def _has_non_text_content(message: Mapping[str, JsonValue]) -> bool:
    content: Final = message.get("content")
    return isinstance(content, list) and any(
        not isinstance(part, dict) or part.get("type") != "text" for part in content
    )


def _last_user_text(operation: str, messages: Sequence[Mapping[str, JsonValue]]) -> str:
    message: Final = next((message for message in reversed(messages) if message.get("role") == "user"), None)
    text: Final = _text_of(message) if message is not None else None
    if not text:
        _reject(f"ScaleDown {operation} requires text in the last user message.")
    return text


def _validate_labels(labels: JsonValue) -> None:
    if not isinstance(labels, list) or not labels:
        _reject("ScaleDown classify requires a non-empty 'labels' list of objects with 'name' and 'rubric' strings.")
    for label in labels:
        if not isinstance(label, Mapping) or not all(
            isinstance(label.get(key), str) and label[key] for key in ("name", "rubric")
        ):
            _reject("Each classify label needs non-empty 'name' and 'rubric' strings.")


def _child(node: JsonValue, key: str) -> JsonValue:
    return node.get(key) if isinstance(node, Mapping) else None


def _extend_ref_path(path: tuple[str, ...], ref: str) -> tuple[str, ...]:
    return (*path, ref)


def _nullable_branch(schema: Mapping[str, JsonValue]) -> Mapping[str, JsonValue] | None:
    if "allOf" in schema:
        _reject("ScaleDown extraction does not support allOf. Supply the object's properties directly.")
    variants: Final = schema.get("anyOf", schema.get("oneOf"))
    if variants is None:
        return None
    if not isinstance(variants, list) or len(variants) != 2:
        _reject("ScaleDown extraction supports anyOf/oneOf only for one schema plus null.")
    branches: Final = tuple(branch for branch in variants if isinstance(branch, dict))
    non_null: Final = tuple(branch for branch in branches if branch.get("type") != "null")
    if len(branches) != 2 or len(non_null) != 1:
        _reject("ScaleDown extraction supports anyOf/oneOf only for one schema plus null.")
    return {
        **non_null[0],
        **{key: value for key, value in schema.items() if key not in {"anyOf", "oneOf"}},
    }


def _deref(
    prop: Mapping[str, JsonValue], root: Mapping[str, JsonValue], path: tuple[str, ...]
) -> tuple[Mapping[str, JsonValue], tuple[str, ...], bool]:
    """Follow nullable branches and local `$ref`s (`#/$defs/X`, `#/definitions/X`).

    Returns the schema, the refs followed so far, and whether a ref repeated (a cycle).
    The walk is a bounded loop, so a long chain of distinct refs cannot exhaust the stack.
    """
    current: Mapping[str, JsonValue] = prop  # rebind-ok: bounded walk along a ref chain
    followed: tuple[str, ...] = path  # rebind-ok: bounded walk along a ref chain
    for _ in range(MAX_REF_CHAIN):
        nullable: Mapping[str, JsonValue] | None = _nullable_branch(current)
        ref: JsonValue = current.get("$ref")
        if nullable is not None:
            current = nullable
            continue
        if not isinstance(ref, str) or not ref.startswith("#/"):
            return current, followed, False
        if ref in followed:
            return current, followed, True
        target: JsonValue = reduce(_child, ref[2:].split("/"), dict(root))
        if not isinstance(target, Mapping):
            return current, followed, False
        current, followed = target, _extend_ref_path(followed, ref)
    _reject(f"The response_format schema follows more than {MAX_REF_CHAIN} chained $refs or nullable branches.")


def _build_properties(
    properties: Mapping[str, JsonValue],
    root: Mapping[str, JsonValue],
    path: tuple[str, ...],
    budget: int,
    depth: int = 0,
) -> tuple[dict[str, JsonValue], int]:  # mutable-ok: serialized JSON entity map
    """Turn JSON-schema properties into the /extract entity map, spending one budget unit per entity.

    Each property becomes an entity whose value is its description (or its name when there is
    none). Nested objects stay nested, arrays of objects become a one-element list holding the
    item's entity map, local `$ref`s are followed, and a ref that points back into itself is cut
    off at that point instead of expanded.
    """
    if depth > MAX_SCHEMA_DEPTH:
        _reject(f"The response_format schema nests deeper than {MAX_SCHEMA_DEPTH} levels.")
    built: Final[dict[str, JsonValue]] = {}  # mutable-ok: filled once while threading the budget through
    remaining = budget  # rebind-ok: the budget is threaded through the loop
    for name, prop in properties.items():
        if remaining <= 0:
            _reject(
                f"The response_format schema expands to more than {MAX_ENTITIES} entities; "
                "remove recursive or heavily repeated $refs."
            )
        built[name], remaining = _build_entity(name, prop, root, path, remaining - 1, depth)
    return built, remaining


def _build_entity(
    name: str, prop: JsonValue, root: Mapping[str, JsonValue], path: tuple[str, ...], budget: int, depth: int
) -> tuple[JsonValue, int]:
    schema, followed, cyclic = _deref(prop if isinstance(prop, Mapping) else {}, root, path)
    leaf: Final = schema.get("description") or name
    nested: Final = schema.get("properties")
    if cyclic:
        return leaf, budget
    if isinstance(nested, Mapping):
        return _build_properties(nested, root, followed, budget, depth + 1)
    raw_items: Final = schema.get("items")
    schema_type: Final = schema.get("type")
    is_array: Final = schema_type == "array" or (isinstance(schema_type, list) and "array" in schema_type)
    if is_array and isinstance(raw_items, Mapping):
        items, item_path, item_cyclic = _deref(raw_items, root, followed)
        item_properties: Final = items.get("properties")
        if not item_cyclic and isinstance(item_properties, Mapping):
            built, remaining = _build_properties(item_properties, root, item_path, budget, depth + 1)
            return [built], remaining
    return leaf, budget


def _extract_request(
    messages: Sequence[Mapping[str, JsonValue]], optional_params: Mapping[str, JsonValue]
) -> Mapping[str, JsonValue]:
    response_format: Final = optional_params.get("response_format")
    json_schema: Final = response_format.get("json_schema") if isinstance(response_format, Mapping) else None
    if isinstance(json_schema, Mapping) and json_schema.get("strict") is True:
        _reject(
            "ScaleDown extract does not support strict JSON schemas. Use strict=false and validate the returned fields."
        )
    schema: Final = json_schema.get("schema") if isinstance(json_schema, Mapping) else None
    properties: Final = _deref(schema, schema, ())[0].get("properties") if isinstance(schema, Mapping) else None
    if not isinstance(schema, dict) or not isinstance(properties, dict) or not properties:
        _reject(
            "ScaleDown extract needs response_format={'type': 'json_schema', ...} with a non-empty "
            "'properties' map; each property becomes an entity and its description the extraction hint."
        )
    return {
        "text": _last_user_text("extract", messages),
        "entities": _build_properties(properties, schema, (), MAX_ENTITIES)[0],
        **{key: optional_params[key] for key in ("threshold", "top_n") if key in optional_params},
    }


def _summarize_request(
    messages: Sequence[Mapping[str, JsonValue]], optional_params: Mapping[str, JsonValue]
) -> Mapping[str, JsonValue]:
    instructions: Final = [
        text for message in messages if message.get("role") in {"system", "developer"} and (text := _text_of(message))
    ]
    return {
        "text": _last_user_text("summarize", messages),
        **({"instructions": "\n".join(instructions)} if instructions else {}),
        **({"max_tokens": optional_params["max_tokens"]} if "max_tokens" in optional_params else {}),
    }


def _compress_request(
    messages: Sequence[Mapping[str, JsonValue]], optional_params: Mapping[str, JsonValue]
) -> Mapping[str, JsonValue]:
    prompt: Final = _last_user_text("compress", messages)
    last_user_index: Final = max(i for i, m in enumerate(messages) if m.get("role") == "user" and _text_of(m))
    context: Final = "\n\n".join(text for i, m in enumerate(messages) if i != last_user_index and (text := _text_of(m)))
    return {
        "context": context,
        "prompt": prompt,
        "scaledown": {"rate": optional_params.get("compression_rate", "auto")},
    }


def _decisions_request(
    messages: Sequence[Mapping[str, JsonValue]], optional_params: Mapping[str, JsonValue]
) -> Mapping[str, JsonValue]:
    questions: Final = optional_params.get("questions")
    if questions is not None:
        _validate_questions(questions)
    return {
        "model": DECISIONS_UPSTREAM_MODEL,
        "state": {"text": _last_user_text("decisions", messages)},
        **({"questions": questions} if questions is not None else {}),
    }


def _require_complete_decisions(body: Mapping[str, JsonValue]) -> None:
    if not body.get("questions"):
        _reject(
            "ScaleDown decisions requires a non-empty 'questions' map, passed as an extra parameter, "
            'e.g. {"questions": {"category": {"type": "choice", "criteria": {"billing": "..."}}}}. '
            "Question types are 'choice', 'noul', and 'score'."
        )


def _validate_questions(questions: JsonValue) -> None:
    if not isinstance(questions, Mapping):
        _reject(f"'questions' must be a map of name to question, got {type(questions).__name__}.")
    for name, question in questions.items():
        if not isinstance(question, Mapping):
            _reject(f"Question '{name}' must be an object.")
        question_type = question.get("type")
        if not isinstance(question_type, str) or question_type not in DECISION_QUESTION_TYPES:
            _reject(f"Question '{name}' has type {question_type!r}; expected one of {sorted(DECISION_QUESTION_TYPES)}.")
        criteria = question.get("criteria")
        if question_type == "choice" and (not isinstance(criteria, Mapping) or not criteria):
            _reject(f"Choice question '{name}' needs a non-empty 'criteria' map of option key to description.")
        if question_type == "score" and (
            not isinstance(criteria, list) or not SCORE_MIN_LEVELS <= len(criteria) <= SCORE_MAX_LEVELS
        ):
            _reject(
                f"Score question '{name}' needs 'criteria' as an ordered list of "
                f"{SCORE_MIN_LEVELS} to {SCORE_MAX_LEVELS} level descriptions, lowest to highest."
            )


def _unwrap(value: JsonValue) -> JsonValue:
    wrapper_keys: Final = {"_value", "_span_anchor"}
    if isinstance(value, Mapping) and "_value" in value and set(value) <= wrapper_keys:
        return value["_value"]
    return value


def _clean_extraction(value: JsonValue, requested: JsonValue) -> JsonValue:
    """Reduce an extraction result to the fields the caller's schema asked for.

    The request's entity map says which keys are real fields, so provider-added keys such as
    `<field>_span_anchor` and `{"_value": ..., "_span_anchor": ...}` wrappers are dropped without
    touching a requested field that happens to share one of those names. The untouched payload
    stays on `_hidden_params["scaledown_response"]`.
    """
    if isinstance(requested, Mapping) and isinstance(value, Mapping):
        return {key: _clean_extraction(value[key], requested[key]) for key in requested if key in value}
    if isinstance(requested, list) and requested and isinstance(value, list):
        return [_clean_extraction(item, requested[0]) for item in value]
    return _unwrap(value)


def _extracted_fields(raw: Mapping[str, JsonValue], requested: JsonValue) -> JsonValue:
    entities: Final = raw.get("entities")
    matches: Final = (
        tuple(entity for entity in entities if isinstance(entity, Mapping)) if isinstance(entities, list) else ()
    )
    scalar_names: Final = (
        tuple(name for name, hint in requested.items() if isinstance(hint, str))
        if isinstance(requested, Mapping)
        else ()
    )
    scalars: Final = {
        name: next(entity["text"] for entity in matches if entity.get("type") == name and "text" in entity)
        for name in scalar_names
        if any(entity.get("type") == name and "text" in entity for entity in matches)
    }
    structured: Final = raw.get("structured_result")
    return _clean_extraction({**scalars, **(structured if isinstance(structured, Mapping) else {})}, requested)


def _native_response(
    operation: str, raw: JsonValue, model_response: ModelResponse, requested: JsonValue = None
) -> ModelResponse:
    """Wrap a native /extract, /summarization/abstractive or /compress/raw/ payload.

    Summarize and compress return the upstream payload as JSON on choices[0].message.content.
    Extract returns the fields requested by response_format without enforcing schema types.
    The native APIs report input tokens only; they return no output token
    count, so completion_tokens stays 0 because it is unmeasured, not because nothing was generated.
    """
    if not isinstance(raw, Mapping) or not raw:
        _reject(
            f"ScaleDown '{operation}' returned an empty or malformed response: {json.dumps(raw)[:500]}",
            500,
        )

    results: Final = raw.get("results")
    input_tokens: Final = raw.get("input_tokens") or (
        (results.get("original_prompt_tokens") if isinstance(results, Mapping) else None)
        or raw.get("original_prompt_tokens")
        if operation == "compress"
        else None
    )
    content: Final = _extracted_fields(raw, requested) if operation == "extract" else raw

    model_response.id = f"chatcmpl-{uuid.uuid4().hex}"  # rebind-ok: LiteLLM fills the response object it passes in
    model_response.created = int(time.time())  # rebind-ok: LiteLLM fills the response object it passes in
    model_response.model = f"scaledown/{operation}"  # rebind-ok: LiteLLM fills the response object it passes in
    model_response.object = "chat.completion"  # rebind-ok: LiteLLM fills the response object it passes in
    model_response.choices = [  # rebind-ok: LiteLLM fills the response object it passes in
        Choices(
            index=0,
            message=Message(role="assistant", content=json.dumps(content, separators=(",", ":"))),
            finish_reason="stop",
        )
    ]
    _set_usage(model_response, {"prompt_tokens": input_tokens or 0})
    model_response.hidden_params["scaledown_response"] = raw  # pyright: ignore[reportUnknownMemberType]  # Inherited hidden_params has untyped values. # rebind-ok: supplied response
    return model_response


def _decisions_response(operation: str, raw: JsonValue, model_response: ModelResponse) -> ModelResponse:
    answers: Final = raw.get("answers") if isinstance(raw, Mapping) else None
    if answers is None:
        _reject(f"ScaleDown decisions response contained no answers: {json.dumps(raw)[:500]}", 500)

    model_response.id = f"chatcmpl-{uuid.uuid4().hex}"  # rebind-ok: LiteLLM fills the response object it passes in
    model_response.created = int(time.time())  # rebind-ok: LiteLLM fills the response object it passes in
    model_response.model = f"scaledown/{operation}"  # rebind-ok: LiteLLM fills the response object it passes in
    model_response.object = "chat.completion"  # rebind-ok: LiteLLM fills the response object it passes in
    model_response.choices = [  # rebind-ok: LiteLLM fills the response object it passes in
        Choices(
            index=0,
            message=Message(role="assistant", content=json.dumps(answers, separators=(",", ":"))),
            finish_reason="stop",
        )
    ]

    usage: Final = raw.get("usage") if isinstance(raw, dict) else None
    counts: Final = usage if isinstance(usage, dict) else {}
    _set_usage(
        model_response,
        {
            "prompt_tokens": counts.get("input_tokens", 0),
            "completion_tokens": counts.get("output_tokens", 0),
        },
    )
    model_response.hidden_params["scaledown_response"] = raw  # pyright: ignore[reportUnknownMemberType]  # Inherited hidden_params has untyped values. # rebind-ok: supplied response
    return model_response


def _set_usage(model_response: ModelResponse, usage: Mapping[str, JsonValue]) -> None:
    prompt_tokens: Final = _token_count(usage.get("prompt_tokens"))
    completion_tokens: Final = _token_count(usage.get("completion_tokens"))
    total_tokens: Final = usage.get("total_tokens")
    model_response.usage = Usage(  # pyright: ignore[reportAttributeAccessIssue]  # ModelResponse stores usage as a dynamic Pydantic field. # rebind-ok: fills the supplied response
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=_token_count(total_tokens) if total_tokens is not None else prompt_tokens + completion_tokens,
    )


def _token_count(value: JsonValue) -> int:
    if value is None:
        return 0
    if isinstance(value, int) and value >= 0:
        return value
    _reject("ScaleDown returned an invalid token count.", 500)
