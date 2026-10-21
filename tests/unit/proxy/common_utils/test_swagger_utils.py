import inspect

from litellm.exceptions import RateLimitError
from litellm.proxy.common_utils.swagger_utils import ERROR_RESPONSES, _error_description


class _ChildWithoutDoc(RateLimitError):
    pass


def test_error_response_descriptions_carry_no_docstring_indentation():
    assert ERROR_RESPONSES[429]["description"] == inspect.cleandoc(RateLimitError.__doc__ or "")
    for response in ERROR_RESPONSES.values():
        assert response["description"] == inspect.cleandoc(response["description"])


def test_error_description_falls_back_to_the_class_name_without_an_own_docstring():
    assert _error_description(_ChildWithoutDoc) == "_ChildWithoutDoc"
