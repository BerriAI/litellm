import json
import time
import uuid
from typing import TYPE_CHECKING, Any

import httpx

from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse, Usage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

DEFAULT_API_BASE = "https://api.scaledown.xyz/v1"

DOMAIN_MODELS = frozenset({"extract", "summarize", "compress"})

DECISIONS_MODELS = frozenset({"classify", "decisions"})

DECISIONS_UPSTREAM_MODEL = "classify-1"

DECISION_QUESTION_TYPES = frozenset({"choice", "noul", "score"})

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
        base = (api_base or get_secret_str("SCALEDOWN_API_BASE") or DEFAULT_API_BASE).rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        if _operation(model) in DECISIONS_MODELS:
            return f"{base}/scaledown"
        return f"{base}/chat/completions"

    def get_supported_openai_params(self, model: str) -> list[str]:
        operation = _operation(model)
        if operation in DECISIONS_MODELS:
            return []
        if operation == "extract":
            return ["response_format", "max_tokens"]
        if operation == "summarize":
            return ["max_tokens"]
        return ["max_tokens", "temperature", "stream"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported = set(self.get_supported_openai_params(model))
        unsupported = [key for key in non_default_params if key not in supported]
        if unsupported and not drop_params:
            raise ScaleDownError(
                status_code=400,
                message=(
                    f"ScaleDown model '{model}' does not support {sorted(unsupported)}. "
                    f"Supported: {sorted(supported)}. Set litellm.drop_params=True to ignore."
                ),
            )
        return {
            **optional_params,
            **{key: value for key, value in non_default_params.items() if key in supported},
        }

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
        return {"model": _operation(model), "messages": messages, **optional_params}

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
        return _domain_response(operation, raw, model_response)

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return ScaleDownError(status_code=status_code, message=error_message, headers=headers)


def _operation(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def _last_user_text(messages: list[AllMessageValues]) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return str(message["content"])
    return None


def _decisions_request(messages: list[AllMessageValues], optional_params: dict) -> dict:
    params = dict(optional_params)
    questions = params.pop("questions", None)
    state = params.pop("state", None)

    if state is None:
        text = _last_user_text(messages)
        if not text:
            raise ScaleDownError(
                status_code=400,
                message=(
                    "ScaleDown decisions requires the text to decide on. Pass it as the last user "
                    "message, or pass state={...} via extra_body."
                ),
            )
        document_fields = {key: params[key] for key in ("document", "document_mime_type") if key in params}
        state = {"text": text, **document_fields}

    if not questions:
        raise ScaleDownError(
            status_code=400,
            message=(
                "ScaleDown decisions requires a non-empty 'questions' map, passed via extra_body, "
                'e.g. {"questions": {"category": {"type": "choice", "criteria": {"billing": "..."}}}}. '
                "Question types are 'choice', 'noul', and 'score'."
            ),
        )

    _validate_questions(questions)
    return {"model": DECISIONS_UPSTREAM_MODEL, "state": state, "questions": questions}


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


def _domain_response(operation: str, raw: dict, model_response: ModelResponse) -> ModelResponse:
    choices = raw.get("choices") or []
    if not choices:
        raise ScaleDownError(
            status_code=500,
            message=f"ScaleDown '{operation}' response contained no choices: {json.dumps(raw)[:500]}",
        )

    model_response.id = raw.get("id") or f"chatcmpl-{uuid.uuid4().hex}"
    model_response.created = raw.get("created") or int(time.time())
    model_response.model = f"scaledown/{operation}"
    model_response.object = "chat.completion"
    model_response.choices = [
        Choices(
            index=choice.get("index", index),
            message=Message(
                role=(choice.get("message") or {}).get("role") or "assistant",
                content=(choice.get("message") or {}).get("content"),
            ),
            finish_reason=choice.get("finish_reason") or "stop",
        )
        for index, choice in enumerate(choices)
    ]
    _set_usage(model_response, raw.get("usage") or {})
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
    if usage.get("cost") is not None:
        model_response._hidden_params["response_cost"] = usage["cost"]
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
