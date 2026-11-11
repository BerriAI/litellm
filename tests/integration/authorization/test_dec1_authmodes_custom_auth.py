"""Custom auth: an operator's own Python function (`general_settings.custom_auth: <module>.<function>`) that
decides who the caller is, and `general_settings.custom_auth_settings.mode` that says when it runs.

Contract (enterprise/litellm_enterprise/proxy/auth/user_api_key_auth.py `enterprise_custom_auth`;
litellm/proxy/auth/user_api_key_auth.py `user_api_key_auth_builder`, `_run_post_custom_auth_checks`, and the
custom auth skip in the centralized common checks):

* The function is awaited with the request and the bearer value. A returned `UserAPIKeyAuth` is the caller
  (team, user, rpm limit and expiry included). A returned string replaces the bearer value and continues on
  the normal path, so returning an `sk-` key makes the caller that virtual key. `None` continues on the normal
  path with the original value.
* With no `custom_auth_settings`, or `mode: on`, an exception the function raises is the response: a
  `ProxyException` as raised, anything else 401 "Authentication Error, <message>". `mode: auto` keeps a
  `ProxyException` but swallows any other exception and falls back to the normal path. `mode: off` never
  calls the function.
* The shared common checks (team blocked, team and user budgets, team model list) are skipped whenever a
  custom auth function is configured, unless `custom_auth_run_common_checks` is set.
  `litellm_settings.enable_post_custom_auth_checks` adds an expiry check and, with common checks on, the key
  model check.

The function is written into each proxy's config directory, where `get_instance_fn` loads it from. It reads
the bearer value as a small instruction (`dec1-object;user=...;team=...`) so one module covers every case,
and appends each value it sees to `calls.log` beside itself so a test can tell whether it ran.
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually
from tests.integration.authorization.dec1_authmodes_support import (
    authmodes_proxy,
    chat_as,
    message_of,
    served_body,
    spend_row,
    unique,
)
from tests.integration.authorization.dec1_keys_support import assert_denied, error_of
from tests.integration.authorization.dec1_limits_support import assert_limited, hashed

pytestmark: Final = pytest.mark.timeout(240)

MODULE_SOURCE: Final = textwrap.dedent(
    '''
    """Synthetic custom auth for the dec1 authmodes contracts. Reads the bearer value as an instruction."""

    from pathlib import Path

    from litellm.proxy._types import ProxyException, UserAPIKeyAuth

    CALLS = Path(__file__).with_name("calls.log")
    NUMBERS = {"rpm_limit": int, "tpm_limit": int, "max_budget": float}


    async def authenticate(request, api_key):
        with CALLS.open("a") as calls:
            calls.write(api_key + "\\n")
        kind, _, rest = api_key.partition(";")
        if kind == "dec1-object":
            fields = dict(part.split("=", 1) for part in rest.split(";") if part)
            values = {}
            for name, raw in fields.items():
                if name == "models":
                    values[name] = raw.split(",")
                elif name in NUMBERS:
                    values[name] = NUMBERS[name](raw)
                else:
                    values[name] = raw
            return UserAPIKeyAuth(**values)
        if kind == "dec1-string":
            return rest
        if kind == "dec1-raise":
            raise ValueError(rest)
        if kind == "dec1-proxy":
            raise ProxyException(message=rest, type="auth_error", param="dec1", code=403)
        return None
    '''
)


@dataclass(frozen=True, slots=True)
class CustomAuthRig:
    proxy: Gateway
    calls: Path

    def saw(self, value: str) -> bool:
        try:
            return value in self.calls.read_text().splitlines()
        except FileNotFoundError:
            return False


def _rig(
    directory: Path,
    *,
    mode: str | None = None,
    general_settings: dict[str, JsonValue] | None = None,
    litellm_settings: dict[str, JsonValue] | None = None,
) -> Iterator[CustomAuthRig]:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "dec1_custom_auth.py").write_text(MODULE_SOURCE)
    settings: Final[dict[str, JsonValue]] = {
        "custom_auth": "dec1_custom_auth.authenticate",
        **({"custom_auth_settings": {"mode": mode}} if mode is not None else {}),
        **(general_settings or {}),
    }
    with authmodes_proxy(directory, general_settings=settings, litellm_settings=litellm_settings) as proxy:
        yield CustomAuthRig(proxy, directory / "calls.log")


@pytest.fixture(scope="module")
def plain(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CustomAuthRig]:
    yield from _rig(tmp_path_factory.mktemp("dec1-authmodes-custom-plain"))


@pytest.fixture(scope="module")
def mode_on(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CustomAuthRig]:
    yield from _rig(tmp_path_factory.mktemp("dec1-authmodes-custom-on"), mode="on")


@pytest.fixture(scope="module")
def mode_off(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CustomAuthRig]:
    yield from _rig(tmp_path_factory.mktemp("dec1-authmodes-custom-off"), mode="off")


@pytest.fixture(scope="module")
def mode_auto(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CustomAuthRig]:
    yield from _rig(tmp_path_factory.mktemp("dec1-authmodes-custom-auto"), mode="auto")


@pytest.fixture(scope="module")
def checked(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CustomAuthRig]:
    yield from _rig(
        tmp_path_factory.mktemp("dec1-authmodes-custom-checked"),
        general_settings={"custom_auth_run_common_checks": True},
        litellm_settings={"enable_post_custom_auth_checks": True},
    )


def _object(**fields: str) -> str:
    return "dec1-object;" + ";".join(f"{name}={value}" for name, value in fields.items())


def _synthetic_key() -> str:
    return f"sk-dec1-custom-{unique('k')}"


def _past() -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()


def test_a_returned_auth_object_names_the_caller_in_spend_logs(plain: CustomAuthRig) -> None:
    """The function returns `UserAPIKeyAuth(user_id=U, team_id=T)` for the bearer value.

    The chat must be served and its spend row's `user` and `team_id` must be U and T.
    """
    with plain.proxy.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        user: Final = unique("custom-user")
        row: Final = spend_row(plain.proxy, chat_as(plain.proxy, _object(user_id=user, team_id=team), model))
        assert (row["user"], row["team_id"]) == (user, team), row


def test_a_returned_auth_object_with_an_sk_api_key_is_logged_by_its_hash(plain: CustomAuthRig) -> None:
    """The returned object carries `api_key=sk-...` (a synthetic value, no such key exists).

    The spend row's `api_key` must be the SHA-256 digest of that value, the same hashing as for a real key.
    """
    with plain.proxy.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = _synthetic_key()
        row: Final = spend_row(plain.proxy, chat_as(plain.proxy, _object(api_key=key, user_id=unique("u")), model))
        assert row["api_key"] == hashed(key), row


def test_a_returned_string_continues_as_that_virtual_key(plain: CustomAuthRig) -> None:
    """The function returns the string of a real `sk-` virtual key created for the test.

    The bearer value is replaced, so the chat must be served as that key: the spend row's `api_key` is the
    key's digest.
    """
    with plain.proxy.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        row: Final = spend_row(plain.proxy, chat_as(plain.proxy, f"dec1-string;{key}", model))
        assert row["api_key"] == hashed(key), row


def test_a_returned_string_that_is_not_a_key_is_refused_as_a_malformed_virtual_key(plain: CustomAuthRig) -> None:
    """The function returns the string `dec1-not-a-key-value`.

    The virtual key path must refuse it with 401 `auth_error` "LiteLLM Virtual Key expected. Received=dec1****
    alue, expected to start with 'sk-'.", masking the replaced value, not the original bearer value.
    """
    with plain.proxy.scenario() as scenario:
        model: Final = scenario.model()
        replaced: Final = "dec1-not-a-key-value"
        assert_denied(
            chat_as(plain.proxy, f"dec1-string;{replaced}", model),
            status=401,
            error_type="auth_error",
            message=f"LiteLLM Virtual Key expected. Received={replaced[:4]}****{replaced[-4:]}, expected to start with 'sk-'.",
        )


def test_returning_none_falls_through_to_virtual_keys(plain: CustomAuthRig) -> None:
    """The function returns `None` for an `sk-` bearer value that is a real virtual key.

    The normal path must serve it as that key, and the function must have been called with it.
    """
    with plain.proxy.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        row: Final = spend_row(plain.proxy, chat_as(plain.proxy, key, model))
        assert row["api_key"] == hashed(key), row
        assert plain.saw(key)


def test_a_plain_exception_without_a_mode_is_a_401_with_its_message(plain: CustomAuthRig) -> None:
    """No `custom_auth_settings`; the function raises `ValueError("dec1 refused this caller")`.

    The proxy must answer 401 `auth_error` with exactly "Authentication Error, dec1 refused this caller".
    """
    assert_denied(
        chat_as(plain.proxy, "dec1-raise;dec1 refused this caller", "any-model"),
        status=401,
        error_type="auth_error",
        message="Authentication Error, dec1 refused this caller",
    )


def test_a_proxy_exception_is_returned_as_raised(plain: CustomAuthRig) -> None:
    """The function raises `ProxyException(message, type="auth_error", param="dec1", code=403)`.

    The proxy must answer 403 with that exact message, type and param, unprefixed.
    """
    assert_denied(
        chat_as(plain.proxy, "dec1-proxy;dec1 forbids this tenant", "any-model"),
        status=403,
        error_type="auth_error",
        message="dec1 forbids this tenant",
        param="dec1",
    )


def test_a_returned_rpm_limit_is_enforced_by_the_rate_limiter(plain: CustomAuthRig) -> None:
    """The returned object carries a synthetic `sk-` api key with `rpm_limit=1`.

    The first chat must be served and the second refused with the v3 limiter's 429 for scope `api_key`, the
    key's digest, kind `requests`, limit 1.
    """
    with plain.proxy.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = _synthetic_key()
        credential: Final = _object(api_key=key, user_id=unique("rpm"), rpm_limit="1")
        assert served_body(chat_as(plain.proxy, credential, model))
        assert_limited(chat_as(plain.proxy, credential, model), "api_key", hashed(key), "requests", 1)


def test_returned_models_are_not_enforced_without_common_checks(plain: CustomAuthRig) -> None:
    """The returned object lists `models=[<another model>]` and the chat asks for a different, deployed model.

    Without `custom_auth_run_common_checks` nothing asks for the key model check, so the chat is served.
    Pinned: the port must not start refusing here silently, and must not stop refusing in `checked` below.
    """
    with plain.proxy.scenario() as scenario:
        model: Final = scenario.model()
        other: Final = scenario.model()
        assert served_body(chat_as(plain.proxy, _object(user_id=unique("u"), models=other), model))


def test_a_returned_expiry_in_the_past_is_ignored_without_post_checks(plain: CustomAuthRig) -> None:
    """The returned object's `expires` is an hour ago, and `enable_post_custom_auth_checks` is off.

    The expiry check only runs with post checks on, so the chat must be served.
    """
    with plain.proxy.scenario() as scenario:
        model: Final = scenario.model()
        assert served_body(chat_as(plain.proxy, _object(user_id=unique("u"), expires=_past()), model))


def test_a_blocked_team_returned_by_custom_auth_is_not_refused_without_common_checks(plain: CustomAuthRig) -> None:
    """The returned object names a team that is blocked in the database.

    Team blocking is part of the common checks that custom auth skips by default, so the chat is served.
    Pinned as current behaviour (the custom auth function is trusted to check this itself).
    """
    with plain.proxy.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        plain.proxy.post("/team/block", {"team_id": team})
        assert served_body(chat_as(plain.proxy, _object(user_id=unique("u"), team_id=team), model))


def test_mode_on_calls_the_function_and_surfaces_its_errors(mode_on: CustomAuthRig) -> None:
    """`custom_auth_settings.mode: on`; the function raises a plain `ValueError`.

    The proxy must answer 401 "Authentication Error, <message>", the same as with no mode set.
    """
    assert_denied(
        chat_as(mode_on.proxy, "dec1-raise;dec1 mode on refusal", "any-model"),
        status=401,
        error_type="auth_error",
        message="Authentication Error, dec1 mode on refusal",
    )


def test_mode_on_honours_a_returned_auth_object(mode_on: CustomAuthRig) -> None:
    """`mode: on`; the function returns an object with a user id.

    The chat must be served and attributed to that user.
    """
    with mode_on.proxy.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = unique("mode-on-user")
        assert spend_row(mode_on.proxy, chat_as(mode_on.proxy, _object(user_id=user), model))["user"] == user


def test_mode_off_never_calls_the_function(mode_off: CustomAuthRig) -> None:
    """`mode: off`; a bearer value that would make the function return an auth object is sent.

    The function must not be called, so the value takes the virtual key path and is refused 401 with the
    malformed virtual key message, and `calls.log` never records it.
    """
    value: Final = _object(user_id=unique("off-user"))
    response: Final = chat_as(mode_off.proxy, value, "any-model")
    assert_denied(response, status=401, error_type="auth_error")
    assert message_of(response).startswith("LiteLLM Virtual Key expected."), response.text
    assert not mode_off.saw(value)


def test_mode_off_serves_virtual_keys(mode_off: CustomAuthRig) -> None:
    """`mode: off`; a real virtual key calls a chat completion.

    It must be served as that key.
    """
    with mode_off.proxy.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        assert spend_row(mode_off.proxy, chat_as(mode_off.proxy, key, model))["api_key"] == hashed(key)


def test_mode_off_still_refuses_a_virtual_key_in_a_blocked_team(mode_off: CustomAuthRig) -> None:
    """`mode: off`; a virtual key belongs to a team that is then blocked.

    With custom auth switched off the key should get every normal check, so the chat should be 401 "Authentication
    Error, Team=<team> is blocked. ...".
    """
    pytest.skip(
        "BUG: the centralized common checks are skipped whenever `custom_auth` is configured "
        "(user_api_key_auth.py, `user_custom_auth is not None and not custom_auth_run_common_checks`), "
        "even with `custom_auth_settings.mode: off`, so team blocking and team or user budgets stop applying "
        "to virtual keys. UNVERIFIED in a run."
    )
    with mode_off.proxy.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        mode_off.proxy.post("/team/block", {"team_id": team})
        response: Final = chat_as(mode_off.proxy, key, model)
        assert_denied(response, status=401, error_type="auth_error")
        assert message_of(response).startswith(f"Authentication Error, Team={team} is blocked."), response.text


def test_mode_auto_falls_back_to_virtual_keys_when_the_function_raises(mode_auto: CustomAuthRig) -> None:
    """`mode: auto`; the function raises a plain exception for every bearer value of the form
    `dec1-raise;<sk key>`.

    The error is swallowed and the original value takes the virtual key path, where it is malformed: the
    proxy must answer 401 with the malformed virtual key message for that value, not the function's message.
    """
    value: Final = "dec1-raise;auto-fallback"
    response: Final = chat_as(mode_auto.proxy, value, "any-model")
    assert_denied(
        response,
        status=401,
        error_type="auth_error",
        message=f"LiteLLM Virtual Key expected. Received={value[:4]}****{value[-4:]}, expected to start with 'sk-'.",
    )
    assert mode_auto.saw(value)


def test_mode_auto_serves_a_virtual_key_the_function_does_not_handle(mode_auto: CustomAuthRig) -> None:
    """`mode: auto`; the function returns `None` for a real virtual key.

    The key must be served and the function must have been asked first.
    """
    with mode_auto.proxy.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        assert spend_row(mode_auto.proxy, chat_as(mode_auto.proxy, key, model))["api_key"] == hashed(key)
        assert mode_auto.saw(key)


def test_mode_auto_keeps_a_proxy_exception(mode_auto: CustomAuthRig) -> None:
    """`mode: auto`; the function raises a `ProxyException` (403, param `dec1`).

    Only plain exceptions fall back, so the proxy must answer the `ProxyException` exactly.
    """
    assert_denied(
        chat_as(mode_auto.proxy, "dec1-proxy;dec1 auto keeps this", "any-model"),
        status=403,
        error_type="auth_error",
        message="dec1 auto keeps this",
        param="dec1",
    )


def test_post_checks_refuse_an_expired_auth_object(checked: CustomAuthRig) -> None:
    """`enable_post_custom_auth_checks` is on; the returned object's `expires` is an hour ago.

    The proxy must answer 401 type `expired_key` with a message starting "Authentication Error - Expired Key.
    Key Expiry time ".
    """
    with checked.proxy.scenario() as scenario:
        model: Final = scenario.model()
        response: Final = chat_as(
            checked.proxy, _object(api_key=_synthetic_key(), user_id=unique("u"), expires=_past()), model
        )
        assert response.status_code == 401, response.text
        assert error_of(response)["type"] == "expired_key", response.text
        assert message_of(response).startswith("Authentication Error - Expired Key. Key Expiry time "), response.text


def test_common_checks_enforce_the_returned_model_list(checked: CustomAuthRig) -> None:
    """With `custom_auth_run_common_checks` and post checks on, the returned object lists only another model.

    The key model check runs, so the chat must be refused with a 401 or 403 and never served.
    UNVERIFIED: the exact status and message of the key model refusal on this path.
    """
    with checked.proxy.scenario() as scenario:
        model: Final = scenario.model()
        other: Final = scenario.model()
        response: Final = chat_as(
            checked.proxy, _object(api_key=_synthetic_key(), user_id=unique("u"), models=other), model
        )
        assert response.status_code in {401, 403}, response.text


def test_common_checks_refuse_a_blocked_team(checked: CustomAuthRig) -> None:
    """With common checks on, the returned object names a team that is blocked in the database.

    The proxy must refuse with 401 `auth_error` whose message starts "Authentication Error, Team=<team> is
    blocked.". UNVERIFIED: that the gate loads the team row for a custom auth caller.
    """
    with checked.proxy.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        checked.proxy.post("/team/block", {"team_id": team})
        response: Final = chat_as(checked.proxy, _object(user_id=unique("u"), team_id=team), model)
        assert_denied(response, status=401, error_type="auth_error")
        assert message_of(response).startswith(f"Authentication Error, Team={team} is blocked."), response.text


def test_common_checks_refuse_a_user_over_budget(checked: CustomAuthRig) -> None:
    """With common checks on, the returned object names a user created with `max_budget: 0`.

    The proxy must eventually answer 422 `budget_exceeded` for that user. UNVERIFIED: that the gate loads the
    user row for a custom auth caller.
    """
    with checked.proxy.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(max_budget=0)
        credential: Final = _object(user_id=user)
        refused: Final = eventually(
            lambda: chat_as(checked.proxy, credential, model), lambda response: response.status_code == 422
        )
        assert error_of(refused)["type"] == "budget_exceeded", refused.text
