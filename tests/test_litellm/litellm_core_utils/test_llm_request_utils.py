"""
Tests for litellm_core_utils.llm_request_utils module.

Covers _ensure_extra_body_is_safe, including the fix for top-level kwargs
(e.g. `strict=True`) that callers like LangChain pass to litellm.completion()
but that are not valid at the OpenAI top-level request body.
"""

from litellm.litellm_core_utils.llm_request_utils import _ensure_extra_body_is_safe
from litellm.utils import get_optional_params


# ---------------------------------------------------------------------------
# _ensure_extra_body_is_safe
# ---------------------------------------------------------------------------


def test_ensure_extra_body_is_safe_strips_strict():
    """strict is not a valid top-level OpenAI param and must be removed."""
    result = _ensure_extra_body_is_safe({"strict": True, "some_custom_key": "value"})
    assert isinstance(result, dict)
    body: dict = result
    assert "strict" not in body
    assert body.get("some_custom_key") == "value"


def test_ensure_extra_body_is_safe_none_passthrough():
    assert _ensure_extra_body_is_safe(None) is None


def test_ensure_extra_body_is_safe_non_dict_passthrough():
    sentinel = object()
    assert _ensure_extra_body_is_safe(sentinel) is sentinel  # type: ignore[arg-type]


def test_ensure_extra_body_is_safe_empty_dict():
    assert _ensure_extra_body_is_safe({}) == {}


def test_ensure_extra_body_is_safe_preserves_valid_keys():
    body = {"some_provider_param": 42, "another": "hello"}
    result = _ensure_extra_body_is_safe(body)
    assert result == {"some_provider_param": 42, "another": "hello"}


def test_ensure_extra_body_is_safe_strict_false_also_stripped():
    """strict=False is equally invalid at the top level."""
    result = _ensure_extra_body_is_safe({"strict": False})
    assert isinstance(result, dict)
    body: dict = result
    assert "strict" not in body


# ---------------------------------------------------------------------------
# get_optional_params — end-to-end: strict must not reach extra_body for OpenAI
# ---------------------------------------------------------------------------


def test_get_optional_params_openai_strict_not_in_extra_body():
    """
    Regression test for: litellm.BadRequestError: OpenAIException - Unrecognized
    request argument supplied: strict

    When strict=True is passed as a top-level kwarg (LangChain-style),
    it must NOT appear in extra_body for OpenAI models.
    """
    params = get_optional_params(
        model="gpt-4o",
        custom_llm_provider="openai",
        strict=True,
        temperature=0.7,
    )
    extra_body: dict = params.get("extra_body", {})
    assert "strict" not in extra_body, (
        "'strict' must not be forwarded in extra_body for OpenAI — "
        "it causes a 400 'Unrecognized request argument' error."
    )


def test_get_optional_params_openai_strict_false_not_in_extra_body():
    params = get_optional_params(
        model="gpt-4o",
        custom_llm_provider="openai",
        strict=False,
    )
    extra_body: dict = params.get("extra_body", {})
    assert "strict" not in extra_body


def test_get_optional_params_openai_strict_does_not_block_other_extra_body_keys():
    """Unrecognized provider-extension keys other than strict still flow through."""
    params = get_optional_params(
        model="gpt-4o",
        custom_llm_provider="openai",
        strict=True,
        my_custom_provider_extension="foo",
    )
    extra_body: dict = params.get("extra_body", {})
    assert "strict" not in extra_body
    assert extra_body.get("my_custom_provider_extension") == "foo"
