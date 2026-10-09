import json
import time
import uuid
import base64
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import httpx

from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse, Usage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

DEFAULT_API_BASE = "https://api.scaledown.xyz"

DOMAIN_MODELS = frozenset({"extract", "summarize", "compress"})

DECISIONS_MODELS = frozenset({"classify", "decisions"})

DECISIONS_UPSTREAM_MODEL = "classify-1"

NATIVE_PATHS = {
    "extract": "/extract",
    "summarize": "/summarization/abstractive",
    "compress": "/compress/raw/",
}

DECISION_QUESTION_TYPES = frozenset({"choice", "noul", "score"})

EXTRA_BODY_KEYS = {
    "extract": frozenset({"threshold", "top_n"}),
    "summarize": frozenset(),
    "compress": frozenset({"compression_rate"}),
    "classify": frozenset({"questions", "state"}),
    "decisions": frozenset({"questions", "state"}),
}

# Set by the LiteLLM router and proxy on every call; handled by LiteLLM, never sent upstream.
ROUTER_PARAMS = frozenset({"max_retries"})

STATE_DOCUMENT_KEYS = ("document", "document_mime_type")

SCORE_MIN_LEVELS = 2
SCORE_MAX_LEVELS = 10


class ScaleDownError(BaseLLMException):
    pass


class ScaleDownChatConfig(BaseConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "scaledown"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        resolved_key = api_key or get_secret_str("SCALEDOWN_API_KEY")
        if resolved_key is None:
            raise ScaleDownError(
                status_code=401,
                message="Missing ScaleDown API key. Set SCALEDOWN_API_KEY or pass api_key to the call.",
            )
        if not api_key and api_base is not None and _root(api_base) != _trusted_root():
            raise ScaleDownError(
                status_code=400,
                message=(
                    "A custom api_base needs its own api_key: the SCALEDOWN_API_KEY from the environment is only "
                    "sent to the default host or the host in SCALEDOWN_API_BASE."
                ),
            )
        return {**headers, "x-api-key": resolved_key, "content-type": "application/json"}

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        base = _root(api_base or _trusted_root())
        operation = _operation(model)
        if operation in DECISIONS_MODELS:
            return f"{base}/v1/scaledown"
        return f"{base}{NATIVE_PATHS[operation]}"

    def get_supported_openai_params(self, model: str) -> list[str]:
        # "stream" is accepted for every model and served as one chunk; see should_fake_stream.
        operation = _operation(model)
        if operation == "extract":
            return ["response_format", "stream"]
        if operation == "summarize":
            return ["max_tokens", "max_completion_tokens", "stream"]
        return ["stream"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported = set(self.get_supported_openai_params(model))
        unsupported = [key for key in non_default_params if key not in supported and key not in ROUTER_PARAMS]
        if unsupported and not drop_params:
            raise ScaleDownError(
                status_code=400,
                message=(
                    f"ScaleDown model '{model}' does not support {sorted(unsupported)}. "
                    f"Supported: {sorted(supported)}. Set litellm.drop_params=True to ignore."
                ),
            )
        mapped = {key: value for key, value in non_default_params.items() if key in supported}
        if "max_completion_tokens" in mapped:
            mapped["max_tokens"] = mapped.pop("max_completion_tokens")
        return {**optional_params, **mapped}

    def should_fake_stream(
        self, model: str | None, stream: bool | None, custom_llm_provider: str | None = None
    ) -> bool:
        # Every ScaleDown model answers in one shot, so a streaming request is served as a single chunk.
        return bool(stream)

    def transform_extra_body(
        self,
        extra_body: Mapping[str, object],
        request: Mapping[str, object],
        model: str,
        litellm_params: Mapping[str, object],
    ) -> Mapping[str, object]:
        # extra_body is merged after guardrails ran on the messages, so it may only add
        # options. Anything that carries prompt text or picks the model is refused.
        operation = _operation(model)
        allowed = EXTRA_BODY_KEYS[operation]
        unexpected = sorted(set(extra_body) - allowed)
        if unexpected:
            raise ScaleDownError(
                status_code=400,
                message=(
                    f"extra_body for scaledown/{operation} "
                    f"{'may only set ' + str(sorted(allowed)) if allowed else 'is not accepted'}, "
                    f"got {unexpected}. Text and instructions must be passed as messages."
                ),
            )
        merged = dict(extra_body)
        if operation == "compress" and "compression_rate" in merged:
            merged = {"scaledown": {"rate": merged["compression_rate"]}}
        if operation in DECISIONS_MODELS:
            state = dict(request.get("state") or {})
            state.update(_checked_state(merged.pop("state", None)))
            if "questions" in merged:
                _validate_questions(merged["questions"])
            merged["state"] = state
        return merged

    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        api_key: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        fake_stream: bool | None = None,
    ) -> tuple[dict, bytes | None]:
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
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        if _operation(model) in DECISIONS_MODELS:
            return _decisions_request(messages, optional_params)
        operation = _operation(model)
        if operation == "extract":
            return _extract_request(messages, optional_params)
        if operation == "summarize":
            return _summarize_request(messages, optional_params)
        return _compress_request(messages, optional_params)

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        try:
            raw = raw_response.json()
        except Exception as exc:
            raise ScaleDownError(
                status_code=raw_response.status_code,
                message=f"ScaleDown returned a non-JSON response: {raw_response.text[:500]}",
            ) from exc

        operation = _operation(model)
        if operation in DECISIONS_MODELS:
            return _decisions_response(operation, raw, model_response)
        return _native_response(operation, raw, model_response)

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return ScaleDownError(status_code=status_code, message=error_message, headers=headers)


def _root(api_base: str) -> str:
    base = api_base.rstrip("/")
    return base[: -len("/v1")] if base.endswith("/v1") else base


def _trusted_root() -> str:
    return _root(get_secret_str("SCALEDOWN_API_BASE") or DEFAULT_API_BASE)


def _operation(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def _text_of(message: AllMessageValues) -> str | None:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
        return "\n".join(parts) if parts else None
    return None


def _last_user_text_or_raise(operation: str, messages: list[AllMessageValues]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            text = _text_of(message)
            if text:
                return text
    raise ScaleDownError(status_code=400, message=f"ScaleDown {operation} requires the text in the last user message.")


def _schema_to_entities(properties: dict) -> dict:
    """Turn JSON-schema properties into the /extract entity map.

    Each property becomes an entity whose value is its description (or its name when
    there is none). Nested objects stay nested and arrays of objects become a
    one-element list holding the item's entity map.
    """
    entities: dict[str, Any] = {}
    for name, prop in properties.items():
        prop = prop if isinstance(prop, dict) else {}
        if isinstance(prop.get("properties"), dict):
            entities[name] = _schema_to_entities(prop["properties"])
        elif prop.get("type") == "array" and isinstance((prop.get("items") or {}).get("properties"), dict):
            entities[name] = [_schema_to_entities(prop["items"]["properties"])]
        else:
            entities[name] = prop.get("description") or name
    return entities


def _extract_request(messages: list[AllMessageValues], optional_params: dict) -> dict:
    schema = ((optional_params.get("response_format") or {}).get("json_schema") or {}).get("schema") or {}
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        raise ScaleDownError(
            status_code=400,
            message=(
                "ScaleDown extract needs response_format={'type': 'json_schema', ...} with a non-empty "
                "'properties' map; each property becomes an entity and its description the extraction hint."
            ),
        )
    body: dict[str, Any] = {
        "text": _last_user_text_or_raise("extract", messages),
        "entities": _schema_to_entities(properties),
    }
    body.update({key: optional_params[key] for key in ("threshold", "top_n") if key in optional_params})
    return body


def _summarize_request(messages: list[AllMessageValues], optional_params: dict) -> dict:
    body: dict[str, Any] = {"text": _last_user_text_or_raise("summarize", messages)}
    instructions = [text for message in messages if message.get("role") == "system" and (text := _text_of(message))]
    if instructions:
        body["instructions"] = "\n".join(instructions)
    if "max_tokens" in optional_params:
        body["max_tokens"] = optional_params["max_tokens"]
    return body


def _compress_request(messages: list[AllMessageValues], optional_params: dict) -> dict:
    prompt = _last_user_text_or_raise("compress", messages)
    last_user_index = max(i for i, m in enumerate(messages) if m.get("role") == "user" and _text_of(m))
    context = "\n\n".join(text for i, m in enumerate(messages) if i != last_user_index and (text := _text_of(m)))
    return {
        "context": context,
        "prompt": prompt,
        "scaledown": {"rate": optional_params.get("compression_rate", "auto")},
    }


def _image_document(message: AllMessageValues) -> dict[str, str]:
    content = message.get("content")
    if not isinstance(content, list):
        return {}
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "image_url":
            continue
        image = part.get("image_url")
        url = image.get("url") if isinstance(image, dict) else image
        if not isinstance(url, str) or not url.startswith("data:") or ";base64," not in url:
            raise ScaleDownError(
                status_code=400,
                message="ScaleDown decisions takes images as base64 data URLs (data:<mime>;base64,<data>).",
            )
        header, data = url.split(";base64,", 1)
        try:
            base64.b64decode(data, validate=True)
        except ValueError as exc:
            raise ScaleDownError(status_code=400, message="Image data URL is not valid base64.") from exc
        return {"document": data, "document_mime_type": header[len("data:") :]}
    return {}


def _checked_state(state: object) -> dict:
    """A caller-supplied state may carry a document, never text.

    Text has to come from the messages, because that is what proxy guardrails inspect and mask.
    """
    if state is None:
        return {}
    if not isinstance(state, dict):
        raise ScaleDownError(status_code=400, message="'state' must be an object.")
    unexpected = sorted(set(state) - set(STATE_DOCUMENT_KEYS))
    if unexpected:
        raise ScaleDownError(
            status_code=400,
            message=(
                f"'state' may only carry {list(STATE_DOCUMENT_KEYS)}, got {unexpected}. "
                "Pass the text to decide on as the last user message."
            ),
        )
    return dict(state)


def _decisions_request(messages: list[AllMessageValues], optional_params: dict) -> dict:
    params = dict(optional_params)
    questions = params.pop("questions", None)
    state = _checked_state(params.pop("state", None))
    state.update({key: params[key] for key in STATE_DOCUMENT_KEYS if key in params})

    for message in reversed(messages):
        if message.get("role") == "user":
            text = _text_of(message)
            if text:
                state = {"text": text, **state}
            for key, value in _image_document(message).items():
                state.setdefault(key, value)
            if text or "document" in state:
                break

    body: dict[str, Any] = {"model": DECISIONS_UPSTREAM_MODEL, "state": state}
    if questions is not None:
        _validate_questions(questions)
        body["questions"] = questions
    return body


def _require_complete_decisions(body: Mapping[str, Any]) -> None:
    state = body.get("state") or {}
    if not state.get("text") and not state.get("document"):
        raise ScaleDownError(
            status_code=400,
            message=(
                "ScaleDown decisions requires the text to decide on as the last user message, "
                "or a base64 document (an image message, or state={'document': ..., 'document_mime_type': ...})."
            ),
        )
    if not body.get("questions"):
        raise ScaleDownError(
            status_code=400,
            message=(
                "ScaleDown decisions requires a non-empty 'questions' map, passed as an extra parameter, "
                'e.g. {"questions": {"category": {"type": "choice", "criteria": {"billing": "..."}}}}. '
                "Question types are 'choice', 'noul', and 'score'."
            ),
        )


def _validate_questions(questions: object) -> None:
    if not isinstance(questions, dict):
        raise ScaleDownError(
            status_code=400,
            message=f"'questions' must be a map of name to question, got {type(questions).__name__}.",
        )
    for name, question in questions.items():
        if not isinstance(question, dict):
            raise ScaleDownError(status_code=400, message=f"Question '{name}' must be an object.")
        question_type = question.get("type")
        if question_type not in DECISION_QUESTION_TYPES:
            raise ScaleDownError(
                status_code=400,
                message=(
                    f"Question '{name}' has type {question_type!r}; expected one of {sorted(DECISION_QUESTION_TYPES)}."
                ),
            )
        if question_type == "choice":
            criteria = question.get("criteria")
            if not isinstance(criteria, dict) or not criteria:
                raise ScaleDownError(
                    status_code=400,
                    message=(
                        f"Choice question '{name}' needs a non-empty 'criteria' map of option key to description."
                    ),
                )
        elif question_type == "score":
            criteria = question.get("criteria")
            if not isinstance(criteria, list) or not (SCORE_MIN_LEVELS <= len(criteria) <= SCORE_MAX_LEVELS):
                raise ScaleDownError(
                    status_code=400,
                    message=(
                        f"Score question '{name}' needs 'criteria' as an ordered list of "
                        f"{SCORE_MIN_LEVELS} to {SCORE_MAX_LEVELS} level descriptions, "
                        f"lowest to highest."
                    ),
                )


def _native_response(operation: str, raw: dict, model_response: ModelResponse) -> ModelResponse:
    """Wrap a native /extract, /summarization/abstractive or /compress/raw/ payload.

    The upstream payload is returned as-is, as JSON on choices[0].message.content, so
    nested extraction values keep their upstream shape. The native APIs report input
    tokens only; they return no output token count, so completion_tokens stays 0
    because it is unmeasured, not because nothing was generated.
    """
    if not isinstance(raw, dict) or not raw:
        raise ScaleDownError(
            status_code=500,
            message=f"ScaleDown '{operation}' returned an empty or malformed response: {json.dumps(raw)[:500]}",
        )

    model_response.id = f"chatcmpl-{uuid.uuid4().hex}"
    model_response.created = int(time.time())
    model_response.model = f"scaledown/{operation}"
    model_response.object = "chat.completion"
    model_response.choices = [
        Choices(
            index=0,
            message=Message(role="assistant", content=json.dumps(raw, separators=(",", ":"))),
            finish_reason="stop",
        )
    ]
    input_tokens = raw.get("input_tokens")
    if input_tokens is None and operation == "compress":
        input_tokens = (raw.get("results") or {}).get("original_prompt_tokens") or raw.get("original_prompt_tokens")
    _set_usage(model_response, {"prompt_tokens": input_tokens or 0})
    model_response._hidden_params["scaledown_response"] = raw
    return model_response


def _decisions_response(operation: str, raw: dict, model_response: ModelResponse) -> ModelResponse:
    answers = raw.get("answers")
    if answers is None:
        raise ScaleDownError(
            status_code=500,
            message=f"ScaleDown decisions response contained no answers: {json.dumps(raw)[:500]}",
        )

    model_response.id = f"chatcmpl-{uuid.uuid4().hex}"
    model_response.created = int(time.time())
    model_response.model = f"scaledown/{operation}"
    model_response.object = "chat.completion"
    model_response.choices = [
        Choices(
            index=0,
            message=Message(role="assistant", content=json.dumps(answers, separators=(",", ":"))),
            finish_reason="stop",
        )
    ]

    usage = raw.get("usage") or {}
    _set_usage(
        model_response,
        {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
        },
    )
    model_response._hidden_params["scaledown_response"] = raw
    return model_response


def _set_usage(model_response: ModelResponse, usage: dict[str, Any]) -> None:
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = usage.get("total_tokens")
    model_response.usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=int(total_tokens) if total_tokens is not None else prompt_tokens + completion_tokens,
    )
