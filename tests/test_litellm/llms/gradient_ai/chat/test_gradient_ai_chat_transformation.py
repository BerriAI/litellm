import pytest

import litellm
from litellm.utils import UnsupportedParamsError


def test_gradient_ai_honors_global_drop_params():
    original = litellm.drop_params
    litellm.drop_params = True
    try:
        result = litellm.get_optional_params(model="llama3", custom_llm_provider="gradient_ai", user="abc")
        assert "user" not in result
    finally:
        litellm.drop_params = original


def test_gradient_ai_raises_for_unsupported_param_without_drop_params():
    original = litellm.drop_params
    litellm.drop_params = False
    try:
        with pytest.raises(UnsupportedParamsError):
            litellm.get_optional_params(model="llama3", custom_llm_provider="gradient_ai", user="abc")
    finally:
        litellm.drop_params = original
