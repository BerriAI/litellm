from typing import TYPE_CHECKING, Final

import pytest

from litellm.rust_bridge._native import AuthCoreHandle

if TYPE_CHECKING:
    from litellm.rust_bridge._native import (
        _AuthCustomAuthResult,  # pyright: ignore[reportPrivateUsage]  # the stub's step shapes
        _AuthOAuth2Steps,  # pyright: ignore[reportPrivateUsage]  # see above
        _AuthRequestFacts,  # pyright: ignore[reportPrivateUsage]  # see above
    )

FACTS: Final["_AuthRequestFacts"] = {"route": "/chat/completions", "token": "tok-1", "is_llm_api_route": True}


def _oauth2_after_introspection(active: bool) -> "_AuthOAuth2Steps":
    call: Final = AuthCoreHandle().oauth2("tok-1", {"endpoint_configured": True, "requires_active": True})
    assert call.start() == {"kind": "introspect", "token": "tok-1"}
    return call.answer_introspection({"active": active, "user_id": "user-1", "role": None, "team_id": 7})


@pytest.mark.parametrize(
    ("active", "outcome"),
    [
        (True, {"kind": "identity", "token": "tok-1", "user_id": "user-1", "role": None, "team_id": 7}),
        (False, {"kind": "inactive_token"}),
    ],
)
def test_oauth2_reads_python_s_introspection_and_returns_the_verdict(active: bool, outcome: object) -> None:
    assert _oauth2_after_introspection(active) == {"kind": "done", "outcome": outcome}


@pytest.mark.parametrize(
    "result",
    [
        {"kind": "identity", "user_id": "user-1", "team_id": "team-1", "hashed_token": "hashed-1"},
        {"kind": "replaced_key", "token": "sk-new"},
        {"kind": "no_identity"},
    ],
)
def test_custom_auth_returns_what_python_resolved(result: "_AuthCustomAuthResult") -> None:
    call: Final = AuthCoreHandle().custom(FACTS)
    assert call.start() == {"kind": "custom_auth"}
    assert call.answer_custom_auth(result) == {"kind": "done", "outcome": result}


def test_an_answer_before_the_call_starts_is_a_value_error() -> None:
    call: Final = AuthCoreHandle().custom(FACTS)
    with pytest.raises(ValueError, match="has not started"):
        call.answer_custom_auth({"kind": "no_identity"})
