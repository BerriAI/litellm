from collections.abc import Mapping
from itertools import chain
from typing import Final

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import assert_never

from litellm.llms.base_llm.decisions.transformation import (
    BaseDecisionsConfig,
    decisions_text,
    parse_systemone_response,
    systemone_keys,
)
from litellm.types.decisions import (
    DecisionsBase64Audio,
    DecisionsBase64Image,
    DecisionsBase64Video,
    DecisionsIRBoxQuestion,
    DecisionsIRChoiceQuestion,
    DecisionsIRMessages,
    DecisionsIRPredicateQuestion,
    DecisionsIRQuestion,
    DecisionsIRRequest,
    DecisionsIRResponse,
    DecisionsIRScoreQuestion,
    DecisionsIRState,
    OpenAIDecisionContentPart,
    OpenAIDecisionInputImage,
    OpenAIDecisionInputMessage,
    OpenAIDecisionInputText,
    UnsupportedDecisionsRequest,
    systemone_choice_key,
)


class _GeminiUsage(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(default=0, ge=0)


class _GeminiResponse(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    answers: Mapping[str, object]
    usage: _GeminiUsage | None = None


def _instructions(question: DecisionsIRQuestion, default: str) -> str:
    return default if question.instructions is None else decisions_text(question.instructions)


def _question(question: DecisionsIRQuestion) -> Mapping[str, object]:
    match question:
        case DecisionsIRPredicateQuestion():
            return {
                "type": "noul",
                "instructions": _instructions(question, "Does the input match the true rather than false criteria?"),
                **(
                    {
                        "criteria": {
                            f"{key}_description": decisions_text(value) if value is not None else ""
                            for key, value in question.criteria.items()
                        }
                    }
                    if question.criteria is not None
                    else {}
                ),
            }
        case DecisionsIRChoiceQuestion():
            return {
                "type": "choice",
                "instructions": _instructions(question, "Choose the option that best matches the input."),
                "criteria": {
                    systemone_choice_key(option.value): decisions_text(option.description)
                    if option.description is not None
                    else ""
                    for option in question.choices
                },
            }
        case DecisionsIRScoreQuestion():
            return {
                "type": "score",
                "instructions": _instructions(question, "Rate the input against the ordered criteria."),
                "criteria": [
                    decisions_text(level.label)
                    if level.description is None
                    else f"{decisions_text(level.label)}: {level.description}"
                    for level in question.levels
                ],
            }
        case DecisionsIRBoxQuestion():
            return {}
        case _:
            assert_never(question)


def _media(
    media: str | DecisionsBase64Image | DecisionsBase64Audio | DecisionsBase64Video, kind: str
) -> Mapping[str, object]:
    if not isinstance(media, str):
        return {"type": kind, "data": media.base64, "mime_type": media.content_type}
    header, data = media.split(",", 1)
    return {"type": kind, "data": data, "mime_type": header.removeprefix("data:").split(";", 1)[0]}


def _part(part: OpenAIDecisionContentPart) -> Mapping[str, object]:
    match part:
        case OpenAIDecisionInputText():
            return {"type": "text", "text": part.text}
        case OpenAIDecisionInputImage():
            return _media(part.image_url, "image")
        case _:
            assert_never(part)


def _message_parts(message: OpenAIDecisionInputMessage) -> tuple[Mapping[str, object], ...]:
    if isinstance(message.content, str):
        return ({"type": "text", "text": message.content},)
    return tuple(_part(part) for part in message.content)


def _input_parts(input_: DecisionsIRState | DecisionsIRMessages) -> tuple[Mapping[str, object], ...]:
    match input_:
        case DecisionsIRState():
            return ({"type": "text", "text": decisions_text(input_.state)},) + tuple(
                _media(image, "image") for image in input_.images
            )
        case DecisionsIRMessages():
            return tuple(chain.from_iterable(_message_parts(message) for message in input_.messages))
        case _:
            assert_never(input_)


class GeminiDecisionsConfig(BaseDecisionsConfig):
    path = "/v1beta/decisions"
    api_key_env = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    api_base_env = ("GEMINI_API_BASE",)
    supports_images = True
    inlines_remote_images = True
    max_images = 16
    max_total_image_bytes = 20 * 1024 * 1024

    def get_default_api_base(self) -> str | None:
        return "https://generativelanguage.googleapis.com"

    def get_complete_url(self, api_base: str, model: str) -> str:
        return f"{api_base.rstrip('/').removesuffix('/v1beta')}/v1beta/decisions"

    def validate_client_connection(self, api_base: str | None, api_key: str | None) -> None:
        if api_base and not api_key:
            raise ValueError(
                "When overriding api_base for Gemini Decisions, supply an explicit api_key "
                "to prevent sending the shared Google key to a custom host"
            )

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None,
        litellm_params: Mapping[str, object],
    ) -> Mapping[str, str]:
        return {
            **{
                key: value
                for key, value in headers.items()
                if key.lower() not in {"authorization", "x-goog-api-key", "content-type"}
            },
            **({"x-goog-api-key": api_key} if api_key is not None else {}),
            "Content-Type": "application/json",
        }

    def transform_decisions_request(
        self,
        model: str,
        request: DecisionsIRRequest,
        custom_llm_provider: str,
    ) -> Mapping[str, object] | UnsupportedDecisionsRequest:
        if any(isinstance(question, DecisionsIRBoxQuestion) for question in request.questions):
            return UnsupportedDecisionsRequest("Gemini Decisions does not support box questions")
        parts: Final = (
            _input_parts(request.input)
            + tuple(_media(audio, "audio") for audio in request.audio)
            + tuple(_media(video, "video") for video in request.videos)
        )
        return {
            "model": model,
            "input": parts,
            "questions": {
                key: _question(question)
                for key, question in zip(systemone_keys(request.questions), request.questions, strict=True)
            },
        }

    def parse_response(self, payload: object, request: DecisionsIRRequest) -> DecisionsIRResponse:
        response: Final = _GeminiResponse.model_validate(payload)
        usage: Final = response.usage
        return parse_systemone_response(
            {
                **response.model_dump(exclude={"usage"}),
                **(
                    {
                        "usage": {
                            **usage.model_dump(),
                            "input_tokens": usage.total_input_tokens,
                            "output_tokens": usage.total_output_tokens,
                        }
                    }
                    if usage is not None
                    else {}
                ),
            },
            request,
        )
