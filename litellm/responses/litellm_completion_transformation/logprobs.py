"""Chat ``choice.logprobs`` as Responses ``output_text.logprobs``, which share one token shape."""

from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.types.llms.base import OutputTextLogprob, OutputTextTopLogprob
from litellm.types.utils import ChatCompletionTokenLogprob, ChoiceLogprobs, TopLogprob

_CHOICE_LOGPROBS: Final = TypeAdapter(ChoiceLogprobs | None)


def _top_logprob(top: TopLogprob) -> OutputTextTopLogprob:
    return {"token": top.token, "logprob": top.logprob, "bytes": top.bytes}


def _token_logprob(token: ChatCompletionTokenLogprob) -> OutputTextLogprob:
    return {
        "token": token.token,
        "logprob": token.logprob,
        "bytes": token.bytes,
        "top_logprobs": [_top_logprob(top) for top in token.top_logprobs],
    }


def output_text_logprobs(choice_logprobs: object) -> list[OutputTextLogprob] | None:
    """The ``logprobs`` a Responses text part carries for a chat choice, ``None`` when the choice sent none."""
    try:
        validated: Final = _CHOICE_LOGPROBS.validate_python(choice_logprobs)
    except ValidationError:
        return None
    if validated is None or validated.content is None:
        return None
    return [_token_logprob(token) for token in validated.content]
