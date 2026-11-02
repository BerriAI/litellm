from typing import Final

from litellm.proxy.common_utils.validation_error_body import public_validation_errors

_PASSWORD: Final = "hunter2-Sup3rSecret!"


def test_public_validation_errors_drops_input_ctx_and_url():
    errors: Final = (
        {
            "type": "missing",
            "loc": ("body", "user_id"),
            "msg": "Field required",
            "input": {"invitation_link": "abc", "password": _PASSWORD},
            "url": "https://errors.pydantic.dev/2/v/missing",
        },
        {
            "type": "value_error",
            "loc": ("body", "password"),
            "msg": "Value error, password cannot be set here",
            "input": _PASSWORD,
            "ctx": {"error": ValueError(_PASSWORD)},
        },
    )

    public: Final = public_validation_errors(errors)

    assert public == (
        {"type": "missing", "loc": ("body", "user_id"), "msg": "Field required"},
        {"type": "value_error", "loc": ("body", "password"), "msg": "Value error, password cannot be set here"},
    )
    assert _PASSWORD not in repr(public)


def test_public_validation_errors_keeps_type_loc_and_msg_verbatim_in_order():
    errors: Final = (
        {"type": "int_parsing", "loc": ("body", "litellm_params", "rpm"), "msg": "Input should be a valid integer"},
        {"type": "extra_forbidden", "loc": ("body", "users", 0, "user_emial"), "msg": "Extra inputs are not permitted"},
        {"type": "too_short", "loc": ("body", "users"), "msg": "List should have at least 1 item"},
    )

    assert public_validation_errors(errors) == errors


def test_public_validation_errors_empty_in_empty_out():
    assert public_validation_errors(()) == ()
