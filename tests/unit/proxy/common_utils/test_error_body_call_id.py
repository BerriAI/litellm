import pytest

from litellm.proxy._types import ConfigGeneralSettings
from litellm.proxy.common_utils.error_body_call_id import error_body_call_id, with_call_id


@pytest.mark.parametrize(
    "general_settings, call_id, expected",
    [
        ({"include_call_id_in_error_body": True}, "call-1", "call-1"),
        ({"include_call_id_in_error_body": True}, None, None),
        ({"include_call_id_in_error_body": True}, "", None),
        ({"include_call_id_in_error_body": False}, "call-1", None),
        ({"include_call_id_in_error_body": "true"}, "call-1", None),
        ({}, "call-1", None),
    ],
)
def test_only_the_boolean_opt_in_with_a_real_id_yields_a_body_call_id(general_settings, call_id, expected):
    """The setting is off by default and only a literal true turns it on; without an id
    there is nothing to copy, so the body must never get a fabricated one."""
    assert error_body_call_id(general_settings, call_id) == expected


def test_with_call_id_appends_the_key_without_touching_the_input():
    error = {"message": "bad input", "type": "invalid_request_error", "param": None, "code": "400"}

    assert with_call_id(error, "call-1") == {**error, "litellm_call_id": "call-1"}
    assert with_call_id(error, None) == error
    assert "litellm_call_id" not in error


def test_the_setting_name_is_a_config_general_settings_field():
    """The yaml key the docs name and the key the runtime reads must be the same field."""
    assert ConfigGeneralSettings.model_validate({"include_call_id_in_error_body": True}).include_call_id_in_error_body
    assert ConfigGeneralSettings.model_validate({}).include_call_id_in_error_body is None
