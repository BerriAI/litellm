import re
from collections.abc import Generator, Iterable, Mapping
from collections.abc import Set as AbstractSet
from contextlib import contextmanager
from types import MappingProxyType
from typing import Final

import openai

from litellm.types.llms.openai import ResponsesAPIRequestParams

_RESPONSES_REQUEST_FIELDS: Final = (
    ResponsesAPIRequestParams.__required_keys__ | ResponsesAPIRequestParams.__optional_keys__
)
_RESPONSES_FIELD_BY_BRIDGED_FIELD: Final[Mapping[str, str]] = MappingProxyType(
    {
        "max_tokens": "max_output_tokens",
        "max_completion_tokens": "max_output_tokens",
        "reasoning_effort": "reasoning.effort",
        "reasoning_summary": "reasoning.summary",
        "response_format": "text.format",
        "tool_choice": "tool_choice",
        "tools": "tools",
        "verbosity": "text.verbosity",
        "web_search_options": "tools",
    }
)
_FIELDS_ONLY_THE_BRIDGE_SENDS: Final = frozenset({"stream_options"})
_PARAM_PATH_SEPARATOR: Final = re.compile(r"[.\[]")
_FIRST_MESSAGE: Final = re.compile(r"messages\.?\[0\]")


def responses_error_param(
    chat_completion_param: object, caller_fields: AbstractSet[str], request: Mapping[str, object]
) -> str | None:
    if not isinstance(chat_completion_param, str):
        return None
    root: Final = _PARAM_PATH_SEPARATOR.split(chat_completion_param, maxsplit=1)[0]
    if root == "messages":
        return _message_field(chat_completion_param, request)
    responses_field: Final = _RESPONSES_FIELD_BY_BRIDGED_FIELD.get(root)
    if responses_field is not None:
        return responses_field if _caller_sent(responses_field, request) else None
    if root in _FIELDS_ONLY_THE_BRIDGE_SENDS or root not in caller_fields:
        return None
    return chat_completion_param


def _caller_sent(responses_field: str, request: Mapping[str, object]) -> bool:
    return request.get(_PARAM_PATH_SEPARATOR.split(responses_field, maxsplit=1)[0]) is not None


def _message_field(chat_completion_param: str, request: Mapping[str, object]) -> str | None:
    if request.get("previous_response_id"):
        return None
    if request.get("instructions") and _FIRST_MESSAGE.match(chat_completion_param):
        return "instructions"
    return "input"


@contextmanager
def error_param_named_as_a_responses_field(
    caller_kwargs: Iterable[str], request: Mapping[str, object]
) -> Generator[None]:
    caller_fields: Final = _RESPONSES_REQUEST_FIELDS.union(caller_kwargs)
    try:
        yield
    except openai.APIError as error:
        error.param = responses_error_param(error.param, caller_fields, request)
        raise
