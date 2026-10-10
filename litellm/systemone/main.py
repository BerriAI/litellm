from collections.abc import Mapping
from typing import Final

import httpx

from litellm.decisions.call import (
    DecisionsCall,
    SystemOneQuestions,
    SystemOneRequestFields,
    asend,
    prepare_call,
    reject_other_format,
    send,
)
from litellm.types.decisions import DecisionsJSON, DecisionsResponse
from litellm.utils import client  # pyright: ignore[reportUnknownVariableType]  # client is an untyped decorator


def _prepare_systemone_call(
    *,
    model: str,
    state: DecisionsJSON | None,
    questions: SystemOneQuestions | None,
    safety_identifier: str | None,
    api_key: str | None,
    api_base: str | None,
    timeout: float | httpx.Timeout | None,
    custom_llm_provider: str | None,
    extra_headers: Mapping[str, str] | None,
    kwargs: Mapping[str, object],
) -> DecisionsCall:
    reject_other_format(model=model, kwargs=kwargs, field_name="input", other_function="decisions")
    return prepare_call(
        model=model,
        fields=SystemOneRequestFields(state=state, questions=questions),
        safety_identifier=safety_identifier,
        api_key=api_key,
        api_base=api_base,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        kwargs=kwargs,
    )


def _systemone_response(response: object) -> DecisionsResponse:
    assert isinstance(response, DecisionsResponse)
    return response


@client
async def asystemone(
    model: str,
    state: DecisionsJSON | None = None,
    questions: SystemOneQuestions | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    safety_identifier: str | None = None,
    **kwargs: object,  # kwargs-ok: litellm_params forwarded by @client
) -> DecisionsResponse:
    call: Final = _prepare_systemone_call(
        model=model,
        state=state,
        questions=questions,
        safety_identifier=safety_identifier,
        api_key=api_key,
        api_base=api_base,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        kwargs=kwargs,
    )
    return _systemone_response(await asend(call))


@client
def systemone(
    model: str,
    state: DecisionsJSON | None = None,
    questions: SystemOneQuestions | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    safety_identifier: str | None = None,
    **kwargs: object,  # kwargs-ok: litellm_params forwarded by @client
) -> DecisionsResponse:
    call: Final = _prepare_systemone_call(
        model=model,
        state=state,
        questions=questions,
        safety_identifier=safety_identifier,
        api_key=api_key,
        api_base=api_base,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        kwargs=kwargs,
    )
    return _systemone_response(send(call))


__all__ = ["asystemone", "systemone"]
