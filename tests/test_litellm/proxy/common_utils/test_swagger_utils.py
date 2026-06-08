import inspect

from litellm.exceptions import RateLimitError
from litellm.proxy.common_utils.swagger_utils import (
    ERROR_RESPONSES,
    _exception_description,
)


class _ChildWithoutDoc(RateLimitError):
    pass


def test_exception_description_dedents_multiline_docstring():
    description = _exception_description(RateLimitError)

    assert RateLimitError.__doc__ is not None
    assert RateLimitError.__doc__.startswith("\n    ")
    assert description == inspect.cleandoc(RateLimitError.__doc__)
    assert "\n    " not in description


def test_exception_description_falls_back_to_name_when_no_own_doc():
    assert _ChildWithoutDoc.__doc__ is None
    assert _exception_description(_ChildWithoutDoc) == "_ChildWithoutDoc"


def test_rate_limit_error_response_description_is_dedented():
    rate_limit_response = ERROR_RESPONSES[429]

    description = rate_limit_response["description"]
    assert description.startswith("Unified rate-limit error.")
    assert "\n    " not in description
