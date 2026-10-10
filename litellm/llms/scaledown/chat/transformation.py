import base64
import json
import time
import uuid
from collections.abc import Mapping, Sequence
from functools import reduce
from typing import TYPE_CHECKING, Final, NoReturn

import httpx

from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse, Usage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

DEFAULT_API_BASE: Final = "https://api.scaledown.xyz"

DOMAIN_MODELS: Final = frozenset({"extract", "summarize", "compress"})

DECISIONS_MODELS: Final = frozenset({"classify", "decisions"})

DECISIONS_UPSTREAM_MODEL: Final = "classify-1"

NATIVE_PATHS: Final[Mapping[str, str]] = {
    "extract": "/extract",
    "summarize": "/summarization/abstractive",
    "compress": "/compress/raw/",
}

DECISION_QUESTION_TYPES: Final = frozenset({"choice", "noul", "score"})

EXTRA_BODY_KEYS: Final[Mapping[str, frozenset[str]]] = {
    "extract": frozenset({"threshold", "top_n"}),
    "summarize": frozenset(),
    "compress": frozenset({"compression_rate"}),
    "classify": frozenset({"questions", "state"}),
    "decisions": frozenset({"questions", "state"}),
}

# Set by the LiteLLM router and proxy on every call; handled by LiteLLM, never sent upstream.
ROUTER_PARAMS: Final = frozenset({"max_retries", "stream_options"})

STATE_DOCUMENT_KEYS: Final = ("document", "document_mime_type")

SCORE_MIN_LEVELS: Final = 2
SCORE_MAX_LEVELS: Final = 10

MAX_ENTITIES: Final = 1000


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
        headers: dict,  # mutable-ok: BaseConfig signature
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: BaseConfig signature
        optional_params: dict,  # mutable-ok: BaseConfig signature
        litellm_params: dict,  # mutable-ok: BaseConfig signature
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: BaseConfig signature
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
        optional_params: dict,  # mutable-ok: BaseConfig signature
        litellm_params: dict,  # mutable-ok: BaseConfig signature
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
        non_default_params: dict,  # mutable-ok: BaseConfig signature
        optional_params: dict,  # mutable-ok: BaseConfig signature
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: BaseConfig signature
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
    ) -> Mapping[str, object]:
        # extra_body is merged after guardrails ran on the messages, so it may only add options.
        # Anything that carries prompt text or picks the model is refused.
        operation: Final = _operation(model)
        allowed: Final = EXTRA_BODY_KEYS[operation]
        unexpected: Final = sorted(set(extra_body) - allowed)
        if unexpected:
            allowance: Final = f"may only set {sorted(allowed)}" if allowed else "is not accepted"
            _reject(
                f"extra_body for scaledown/{operation} {allowance}, got {unexpected}. "
                "Text and instructions must be passed as messages."
            )
        if operation == "compress":
            return {"scaledown": {"rate": extra_body["compression_rate"]}} if "compression_rate" in extra_body else {}
        if operation not in DECISIONS_MODELS:
            return dict(extra_body)
        if "questions" in extra_body:
            _validate_questions(extra_body["questions"])
        current_state: Final = request.get("state")
        state: Final = {
            **(current_state if isinstance(current_state, Mapping) else {}),
            **_checked_state(extra_body.get("state")),
        }
        return {**{key: value for key, value in extra_body.items() if key != "state"}, "state": state}

    def sign_request(
        self,
        headers: dict,  # mutable-ok: BaseConfig signature
        optional_params: dict,  # mutable-ok: BaseConfig signature
        request_data: dict,  # mutable-ok: BaseConfig signature
        api_base: str,
        api_key: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        fake_stream: bool | None = None,
    ) -> tuple[dict, bytes | None]:  # mutable-ok: BaseConfig signature
        # Runs after extra_body is merged, so this is where a decisions body is known to be complete.
        if model is not None and _operation(model) in DECISIONS_MODELS:
            _require_complete_decisions(request_data)
        return super().sign_request(
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base=api_base,
            api_key=api_key,
            model=model,
            stream=stream,
            fake_stream=fake_stream,
        )

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: BaseConfig signature
        optional_params: dict,  # mutable-ok: BaseConfig signature
        litellm_params: dict,  # mutable-ok: BaseConfig signature
        headers: dict,  # mutable-ok: BaseConfig signature
    ) -> dict:  # mutable-ok: BaseConfig signature
        operation: Final = _operation(model)
        if operation in DECISIONS_MODELS:
            return dict(_decisions_request(messages, optional_params))
        if operation == "extract":
            return dict(_extract_request(messages, optional_params))
        if operation == "summarize":
            return dict(_summarize_request(messages, optional_params))
        return dict(_compress_request(messages, optional_params))

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict,  # mutable-ok: BaseConfig signature
        messages: list[AllMessageValues],  # mutable-ok: BaseConfig signature
        optional_params: dict,  # mutable-ok: BaseConfig signature
        litellm_params: dict,  # mutable-ok: BaseConfig signature
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        try:
            raw: Final = raw_response.json()
        except ValueError as exc:
            raise ScaleDownError(
                status_code=raw_response.status_code,
                message=f"ScaleDown returned a non-JSON response: {raw_response.text[:500]}",
            ) from exc

        operation: Final = _operation(model)
        if operation in DECISIONS_MODELS:
            return _decisions_response(operation, raw, model_response)
        return _native_response(operation, raw, model_response, request_data.get("entities"))

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,  # mutable-ok: BaseConfig signature
    ) -> BaseLLMException:
        return ScaleDownError(status_code=status_code, message=error_message, headers=headers)


def _root(api_base: str) -> str:
    return api_base.rstrip("/").removesuffix("/v1")


def _trusted_root() -> str:
    return _root(get_secret_str("SCALEDOWN_API_BASE") or DEFAULT_API_BASE)


def _operation(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def _text_of(message: AllMessageValues) -> str | None:
    content: Final = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: Final = [
            str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(parts) if parts else None
    return None


def _last_user_text(operation: str, messages: Sequence[AllMessageValues]) -> str:
    texts: Final = [text for message in messages if message.get("role") == "user" and (text := _text_of(message))]
    if not texts:
        _reject(f"ScaleDown {operation} requires the text in the last user message.")
    return texts[-1]


def _text_and_document(operation: str, messages: Sequence[AllMessageValues]) -> Mapping[str, str]:
    """The text and/or the single image or document in the last user message that has either."""
    user_messages: Final = [message for message in messages if message.get("role") == "user"]
    found: Final = next(
        (
            {**({"text": text} if text else {}), **_image_document(message)}
            for message in reversed(user_messages)
            if (text := _text_of(message)) or _image_document(message)
        ),
        {},
    )
    if not found:
        _reject(f"ScaleDown {operation} requires the text or an image in the last user message.")
    return found


def _child(node: object, key: str) -> object:
    return node.get(key) if isinstance(node, Mapping) else None


def _deref(
    prop: Mapping[str, object], root: Mapping[str, object], path: tuple[str, ...]
) -> tuple[Mapping[str, object], tuple[str, ...], bool]:
    """Follow local `$ref`s (`#/$defs/X`, `#/definitions/X`) to the schema they name.

    Returns the schema, the refs followed so far, and whether a ref repeated (a cycle).
    """
    ref: Final = prop.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return prop, path, False
    if ref in path:
        return prop, path, True
    target: Final = reduce(_child, ref[2:].split("/"), root)
    if not isinstance(target, Mapping):
        return prop, path, False
    return _deref(target, root, (*path, ref))


def _build_properties(
    properties: Mapping[str, object], root: Mapping[str, object], path: tuple[str, ...], budget: int
) -> tuple[Mapping[str, object], int]:
    """Turn JSON-schema properties into the /extract entity map, spending one budget unit per entity.

    Each property becomes an entity whose value is its description (or its name when there is
    none). Nested objects stay nested, arrays of objects become a one-element list holding the
    item's entity map, local `$ref`s are followed, and a ref that points back into itself is cut
    off at that point instead of expanded.
    """
    built: Final[dict[str, object]] = {}  # mutable-ok: filled once while threading the budget through
    remaining = budget  # rebind-ok: the budget is threaded through the loop
    for name, prop in properties.items():
        if remaining <= 0:
            _reject(
                f"The response_format schema expands to more than {MAX_ENTITIES} entities; "
                "remove recursive or heavily repeated $refs."
            )
        built[name], remaining = _build_entity(name, prop, root, path, remaining - 1)
    return built, remaining


def _build_entity(
    name: str, prop: object, root: Mapping[str, object], path: tuple[str, ...], budget: int
) -> tuple[object, int]:
    schema, followed, cyclic = _deref(prop, root, path) if isinstance(prop, Mapping) else ({}, path, False)
    leaf: Final = schema.get("description") or name
    nested: Final = schema.get("properties")
    if cyclic:
        return leaf, budget
    if isinstance(nested, Mapping):
        return _build_properties(nested, root, followed, budget)
    raw_items: Final = schema.get("items")
    if schema.get("type") == "array" and isinstance(raw_items, Mapping):
        items, item_path, item_cyclic = _deref(raw_items, root, followed)
        item_properties: Final = items.get("properties")
        if not item_cyclic and isinstance(item_properties, Mapping):
            built, remaining = _build_properties(item_properties, root, item_path, budget)
            return [built], remaining
    return leaf, budget


def _extract_request(
    messages: Sequence[AllMessageValues], optional_params: Mapping[str, object]
) -> Mapping[str, object]:
    response_format: Final = optional_params.get("response_format")
    json_schema: Final = response_format.get("json_schema") if isinstance(response_format, Mapping) else None
    schema: Final = json_schema.get("schema") if isinstance(json_schema, Mapping) else None
    properties: Final = _deref(schema, schema, ())[0].get("properties") if isinstance(schema, Mapping) else None
    if not isinstance(properties, Mapping) or not properties:
        _reject(
            "ScaleDown extract needs response_format={'type': 'json_schema', ...} with a non-empty "
            "'properties' map; each property becomes an entity and its description the extraction hint."
        )
    return {
        **_text_and_document("extract", messages),
        "entities": _build_properties(properties, schema, (), MAX_ENTITIES)[0],
        **{key: optional_params[key] for key in ("threshold", "top_n") if key in optional_params},
    }


def _summarize_request(
    messages: Sequence[AllMessageValues], optional_params: Mapping[str, object]
) -> Mapping[str, object]:
    instructions: Final = [
        text for message in messages if message.get("role") == "system" and (text := _text_of(message))
    ]
    return {
        **_text_and_document("summarize", messages),
        **({"instructions": "\n".join(instructions)} if instructions else {}),
        **({"max_tokens": optional_params["max_tokens"]} if "max_tokens" in optional_params else {}),
    }


def _compress_request(
    messages: Sequence[AllMessageValues], optional_params: Mapping[str, object]
) -> Mapping[str, object]:
    prompt: Final = _last_user_text("compress", messages)
    last_user_index: Final = max(i for i, m in enumerate(messages) if m.get("role") == "user" and _text_of(m))
    context: Final = "\n\n".join(text for i, m in enumerate(messages) if i != last_user_index and (text := _text_of(m)))
    return {
        "context": context,
        "prompt": prompt,
        "scaledown": {"rate": optional_params.get("compression_rate", "auto")},
    }


def _image_document(message: AllMessageValues) -> Mapping[str, str]:
    content: Final = message.get("content")
    if not isinstance(content, list):
        return {}
    urls: Final = [
        (part.get("image_url") or {}).get("url") if isinstance(part.get("image_url"), dict) else part.get("image_url")
        for part in content
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]
    if not urls:
        return {}
    if len(urls) > 1:
        _reject("ScaleDown takes one image or document per request; send the images in separate requests.")
    url: Final = urls[0]
    if not isinstance(url, str) or not url.startswith("data:") or ";base64," not in url:
        _reject("ScaleDown takes images as base64 data URLs (data:<mime>;base64,<data>).")
    header, data = url.split(";base64,", 1)
    try:
        base64.b64decode(data, validate=True)
    except ValueError:
        _reject("Image data URL is not valid base64.")
    return {"document": data, "document_mime_type": header[len("data:") :]}


def _checked_state(state: object) -> Mapping[str, object]:
    """A caller-supplied state may carry a document, never text.

    Text has to come from the messages, because that is what proxy guardrails inspect and mask.
    """
    if state is None:
        return {}
    if not isinstance(state, Mapping):
        _reject("'state' must be an object.")
    unexpected: Final = sorted(set(state) - set(STATE_DOCUMENT_KEYS))
    if unexpected:
        _reject(
            f"'state' may only carry {list(STATE_DOCUMENT_KEYS)}, got {unexpected}. "
            "Pass the text to decide on as the last user message."
        )
    return dict(state)


def _decisions_request(
    messages: Sequence[AllMessageValues], optional_params: Mapping[str, object]
) -> Mapping[str, object]:
    questions: Final = optional_params.get("questions")
    explicit_state: Final = {
        **_checked_state(optional_params.get("state")),
        **{key: optional_params[key] for key in STATE_DOCUMENT_KEYS if key in optional_params},
    }
    user_messages: Final = [message for message in messages if message.get("role") == "user"]
    from_messages: Final = next(
        (
            {**({"text": text} if text else {}), **_image_document(message)}
            for message in reversed(user_messages)
            if (text := _text_of(message)) or _image_document(message)
        ),
        {},
    )
    if questions is not None:
        _validate_questions(questions)
    return {
        "model": DECISIONS_UPSTREAM_MODEL,
        "state": {**from_messages, **explicit_state},
        **({"questions": questions} if questions is not None else {}),
    }


def _require_complete_decisions(body: Mapping[str, object]) -> None:
    state: Final = body.get("state")
    if not isinstance(state, Mapping) or not (state.get("text") or state.get("document")):
        _reject(
            "ScaleDown decisions requires the text to decide on as the last user message, "
            "or a base64 document (an image message, or state={'document': ..., 'document_mime_type': ...})."
        )
    if not body.get("questions"):
        _reject(
            "ScaleDown decisions requires a non-empty 'questions' map, passed as an extra parameter, "
            'e.g. {"questions": {"category": {"type": "choice", "criteria": {"billing": "..."}}}}. '
            "Question types are 'choice', 'noul', and 'score'."
        )


def _validate_questions(questions: object) -> None:
    if not isinstance(questions, Mapping):
        _reject(f"'questions' must be a map of name to question, got {type(questions).__name__}.")
    for name, question in questions.items():
        if not isinstance(question, Mapping):
            _reject(f"Question '{name}' must be an object.")
        question_type = question.get("type")
        if question_type not in DECISION_QUESTION_TYPES:
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


def _unwrap(value: object) -> object:
    wrapper_keys: Final = {"_value", "_span_anchor"}
    if isinstance(value, Mapping) and "_value" in value and set(value) <= wrapper_keys:
        return value["_value"]
    return value


def _clean_extraction(value: object, requested: object) -> object:
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


def _native_response(
    operation: str, raw: object, model_response: ModelResponse, requested: object = None
) -> ModelResponse:
    """Wrap a native /extract, /summarization/abstractive or /compress/raw/ payload.

    Summarize and compress return the upstream payload as JSON on choices[0].message.content.
    Extract returns the extracted fields, so the content validates against the caller's
    response_format. The native APIs report input tokens only; they return no output token
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
    content: Final = _clean_extraction(raw.get("structured_result") or {}, requested) if operation == "extract" else raw

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
    model_response._hidden_params["scaledown_response"] = raw  # rebind-ok: LiteLLM response object
    return model_response


def _decisions_response(operation: str, raw: object, model_response: ModelResponse) -> ModelResponse:
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

    usage: Final = raw.get("usage") or {}
    _set_usage(
        model_response,
        {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
        },
    )
    model_response._hidden_params["scaledown_response"] = raw  # rebind-ok: LiteLLM response object
    return model_response


def _set_usage(model_response: ModelResponse, usage: Mapping[str, object]) -> None:
    prompt_tokens: Final = int(usage.get("prompt_tokens") or 0)
    completion_tokens: Final = int(usage.get("completion_tokens") or 0)
    total_tokens: Final = usage.get("total_tokens")
    model_response.usage = Usage(  # rebind-ok: LiteLLM fills the response object it passes in
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=int(total_tokens) if total_tokens is not None else prompt_tokens + completion_tokens,
    )
