"""Chat ``choice.logprobs`` as Responses ``output_text.logprobs``, which share one token shape."""

from collections.abc import Mapping
from typing import Final

from pydantic import ValidationError

from litellm.types.llms.base import OutputTextLogprob, OutputTextTopLogprob
from litellm.types.utils import ChatCompletionTokenLogprob, ChoiceLogprobs, TopLogprob


def _top_logprob(top: TopLogprob) -> OutputTextTopLogprob:
    return {"token": top.token, "logprob": top.logprob, "bytes": top.bytes}


def _token_logprob(token: ChatCompletionTokenLogprob) -> OutputTextLogprob:
    return {
        "token": token.token,
        "logprob": token.logprob,
        "bytes": token.bytes,
        "top_logprobs": tuple(_top_logprob(top) for top in token.top_logprobs),
    }


def _choice_logprobs(choice_logprobs: object) -> ChoiceLogprobs | None:
    if isinstance(choice_logprobs, ChoiceLogprobs):
        return choice_logprobs
    if not isinstance(choice_logprobs, Mapping):
        return None
    try:
        return ChoiceLogprobs.model_validate(choice_logprobs)
    except ValidationError:
        return None


def output_text_logprobs(choice_logprobs: object) -> tuple[OutputTextLogprob, ...] | None:
    """The ``logprobs`` a Responses text part carries for a chat choice, ``None`` when the choice sent none."""
    validated: Final = _choice_logprobs(choice_logprobs)
    if validated is None or validated.content is None:
        return None
    return tuple(_token_logprob(token) for token in validated.content)
