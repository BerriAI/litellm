import warnings
from collections.abc import Mapping
from typing import Final

import httpx
from pydantic import TypeAdapter

from litellm.decisions.call import (
    DecisionsAudioClips,
    DecisionsCall,
    DecisionsVideoClips,
    OpenAIDecisionQuestions,
    OpenAIDecisionRequestFields,
    RequestFields,
    SystemOneQuestions,
    SystemOneRequestFields,
    asend,
    log_caller_request,
    prepare_call,
    reject_images_next_to_input,
    send,
)
from litellm.types.decisions import DecisionsJSON, DecisionsResponse, OpenAIDecisionInput, OpenAIDecisionResponse
from litellm.utils import client

_STATE_ADAPTER: Final[TypeAdapter[DecisionsJSON | None]] = TypeAdapter(DecisionsJSON | None)
_SYSTEMONE_QUESTIONS_ADAPTER: Final[TypeAdapter[SystemOneQuestions | None]] = TypeAdapter(SystemOneQuestions | None)
_OPENAI_QUESTIONS_ADAPTER: Final[TypeAdapter[OpenAIDecisionQuestions | None]] = TypeAdapter(
    OpenAIDecisionQuestions | None
)
_SYSTEMONE_FIELDS: Final = frozenset({"state", "images", "assets"})


def _request_fields(
    model: str,
    decision_input: OpenAIDecisionInput | None,
    questions: OpenAIDecisionQuestions | SystemOneQuestions | None,
    audio: DecisionsAudioClips | None,
    videos: DecisionsVideoClips | None,
    kwargs: Mapping[str, object],
) -> RequestFields:
    if "state" not in kwargs:
        reject_images_next_to_input(model=model, kwargs=kwargs)
        return OpenAIDecisionRequestFields(
            input=decision_input,
            questions=_OPENAI_QUESTIONS_ADAPTER.validate_python(questions),
            audio=audio,
            videos=videos,
        )
    warnings.warn(
        "litellm.decisions(state=...) is deprecated, call litellm.systemone for the System One request format",
        UserWarning,
        stacklevel=5,
    )
    return SystemOneRequestFields(
        state=_STATE_ADAPTER.validate_python(kwargs["state"]),
        questions=_SYSTEMONE_QUESTIONS_ADAPTER.validate_python(questions),
        images=kwargs.get("images"),
        assets=kwargs.get("assets"),
        audio=audio,
        videos=videos,
    )


def _prepare_decisions_call(
    *,
    model: str,
    decision_input: OpenAIDecisionInput | None,
    questions: OpenAIDecisionQuestions | SystemOneQuestions | None,
    safety_identifier: str | None,
    audio: DecisionsAudioClips | None,
    videos: DecisionsVideoClips | None,
    api_key: str | None,
    api_base: str | None,
    timeout: float | httpx.Timeout | None,
    custom_llm_provider: str | None,
    extra_headers: Mapping[str, str] | None,
    kwargs: Mapping[str, object],
) -> DecisionsCall:
    log_caller_request(
        kwargs,
        {
            "input": decision_input,
            "questions": questions,
            "audio": audio,
            "videos": videos,
            **{key: kwargs[key] for key in _SYSTEMONE_FIELDS if key in kwargs},
        },
    )
    return prepare_call(
        model=model,
        fields=_request_fields(model, decision_input, questions, audio, videos, kwargs),
        safety_identifier=safety_identifier,
        api_key=api_key,
        api_base=api_base,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        kwargs={key: value for key, value in kwargs.items() if key not in _SYSTEMONE_FIELDS},
    )


@client
async def adecisions(
    model: str,
    input: OpenAIDecisionInput | None = None,
    questions: OpenAIDecisionQuestions | SystemOneQuestions | None = None,
    safety_identifier: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    audio: DecisionsAudioClips | None = None,
    videos: DecisionsVideoClips | None = None,
    **kwargs: object,
) -> OpenAIDecisionResponse | DecisionsResponse:
    call: Final = _prepare_decisions_call(
        model=model,
        decision_input=input,
        questions=questions,
        safety_identifier=safety_identifier,
        audio=audio,
        videos=videos,
        api_key=api_key,
        api_base=api_base,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        kwargs=kwargs,
    )
    return await asend(call)


@client
def decisions(
    model: str,
    input: OpenAIDecisionInput | None = None,
    questions: OpenAIDecisionQuestions | SystemOneQuestions | None = None,
    safety_identifier: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    audio: DecisionsAudioClips | None = None,
    videos: DecisionsVideoClips | None = None,
    **kwargs: object,
) -> OpenAIDecisionResponse | DecisionsResponse:
    call: Final = _prepare_decisions_call(
        model=model,
        decision_input=input,
        questions=questions,
        safety_identifier=safety_identifier,
        audio=audio,
        videos=videos,
        api_key=api_key,
        api_base=api_base,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        kwargs=kwargs,
    )
    return send(call)


__all__ = ["adecisions", "decisions"]
