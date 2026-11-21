from dataclasses import replace
from typing import Final

import pytest

from litellm.proxy.mcp_server.events.cursors import CursorBinding, OpenedCursor, arguments_digest, open_cursor, seal_cursor
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok
from litellm.types.mcp_server.events import EventsErrorCode

NOW: Final = 1_800_000_000
BINDING: Final = CursorBinding(server_id="srv", event="build.failed", args_hash=arguments_digest({"repo": "a"}), caller="key:1")


@pytest.fixture(autouse=True)
def salt_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "synthetic-events-cursor-key")


def test_a_sealed_cursor_reopens_to_the_upstream_cursor_and_hides_it() -> None:
    sealed: Final = seal_cursor(BINDING, "upstream-42", NOW)
    assert isinstance(sealed, Ok) and sealed.ok is not None
    assert "upstream-42" not in sealed.ok
    assert open_cursor(sealed.ok, BINDING, NOW + 10) == Ok(OpenedCursor(upstream="upstream-42"))


def test_a_null_cursor_stays_null_in_both_directions() -> None:
    assert seal_cursor(BINDING, None, NOW) == Ok(None)
    assert open_cursor(None, BINDING, NOW) == Ok(OpenedCursor(upstream=None))


@pytest.mark.parametrize(
    "other",
    (
        replace(BINDING, server_id="other-server"),
        replace(BINDING, event="deploy.done"),
        replace(BINDING, args_hash=arguments_digest({"repo": "b"})),
        replace(BINDING, caller="key:2"),
    ),
)
def test_a_cursor_presented_for_another_subscription_is_rejected(other: CursorBinding) -> None:
    sealed: Final = seal_cursor(BINDING, "upstream-42", NOW)
    assert isinstance(sealed, Ok) and sealed.ok is not None
    opened: Final = open_cursor(sealed.ok, other, NOW)
    assert isinstance(opened, Error) and opened.error.code is EventsErrorCode.INVALID_PARAMS


def test_an_expired_cursor_restarts_from_now_instead_of_failing() -> None:
    sealed: Final = seal_cursor(BINDING, "upstream-42", NOW)
    assert isinstance(sealed, Ok) and sealed.ok is not None
    assert open_cursor(sealed.ok, BINDING, NOW + 401 * 24 * 3600) == Ok(OpenedCursor(upstream=None, expired=True))


@pytest.mark.parametrize("token", ("not-a-cursor", "mcp_state_v1.AAAA", ""))
def test_a_forged_cursor_is_invalid_params(token: str) -> None:
    opened: Final = open_cursor(token, BINDING, NOW)
    assert isinstance(opened, Error) and opened.error.code is EventsErrorCode.INVALID_PARAMS


def test_without_a_salt_key_cursors_fail_with_the_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_SALT_KEY")
    sealed: Final = seal_cursor(BINDING, "upstream-42", NOW)
    assert isinstance(sealed, Error) and sealed.error.code is EventsErrorCode.INTERNAL_ERROR
    assert "LITELLM_SALT_KEY" in sealed.error.message


@pytest.mark.parametrize(
    ("left", "right", "same"),
    (({"a": 1, "b": 2}, {"b": 2, "a": 1}, True), ({"a": 1}, {"a": "1"}, False), ({}, {"a": None}, False)),
)
def test_argument_digests_compare_canonical_json(left: dict[str, object], right: dict[str, object], same: bool) -> None:
    assert (arguments_digest(left) == arguments_digest(right)) is same
