import base64
import copy
import json
from collections.abc import Mapping
from typing import Final

import pytest

import litellm
from litellm.exceptions import CallerCredentialAuthenticationError
from litellm.llms.chatgpt.common_utils import CHATGPT_API_BASE, ChatGPTAuthError
from litellm.llms.chatgpt.subscription_login import (
    attach_chatgpt_subscription_login,
    names_chatgpt_subscription_login,
)
from litellm.litellm_core_utils.clientside_credential_handler import get_dynamic_litellm_params
from litellm.types.proxy.litellm_pre_call_utils import RedactedDict
from litellm.types.utils import CredentialItem


def _jwt(claims: Mapping[str, object]) -> str:
    def segment(value: Mapping[str, object]) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()

    return f"{segment({'alg': 'RS256'})}.{segment(claims)}.fake-signature"


TEAM_A_LOGIN: Final = "team-a-chatgpt"
TEAM_B_LOGIN: Final = "team-b-chatgpt"
TEAM_A_TOKEN: Final = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-team-a"}, "exp": 4102444800})
TEAM_B_TOKEN: Final = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-team-b"}, "exp": 4102444800})
CLIENT_TOKEN: Final = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-client"}})


def _subscription(name: str, access_token: str, owner_team_id: str) -> CredentialItem:
    return CredentialItem(
        credential_name=name,
        credential_values={
            "chatgpt_access_token": access_token,
            "chatgpt_refresh_token": f"refresh-{name}",
            "chatgpt_id_token": access_token,
            "chatgpt_account_id": f"acct-{owner_team_id}",
            "chatgpt_expires_at": "4102444800",
        },
        credential_info={"custom_llm_provider": "chatgpt", "chatgpt_owner_team_id": owner_team_id},
    )


ORDINARY: Final = CredentialItem(
    credential_name="plain-openai", credential_values={"api_key": "sk-plain-example"}, credential_info={}
)


@pytest.fixture(autouse=True)
def credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            _subscription(TEAM_A_LOGIN, TEAM_A_TOKEN, "team-a"),
            _subscription(TEAM_B_LOGIN, TEAM_B_TOKEN, "team-b"),
            ORDINARY,
        ],
    )


def _owned(*logins: tuple[str, str]) -> dict[str, object]:
    return {"chatgpt_subscription_logins": RedactedDict(dict(logins)), "raw_headers": {}}


def _request(**overrides: object) -> dict[str, object]:
    return {
        "model": "chatgpt/gpt-5.5",
        "litellm_credential_name": TEAM_A_LOGIN,
        "chatgpt_auth_file": "/host/chatgpt/auth.json",
        "secret_fields": _owned((TEAM_A_LOGIN, TEAM_A_TOKEN)),
        **overrides,
    }


@pytest.mark.parametrize(
    ("model", "overrides"),
    [
        ("chatgpt/gpt-5.5", {}),
        (None, {}),
        ("gpt-5.5", {"model": "gpt-5.5", "custom_llm_provider": "chatgpt"}),
    ],
    ids=["model_argument", "model_kwarg", "custom_llm_provider"],
)
def test_attach_sends_the_owned_login_to_chatgpt_instead_of_the_host_auth_file(
    model: str | None, overrides: dict[str, object]
) -> None:
    kwargs: Final = _request(**overrides)

    attach_chatgpt_subscription_login(kwargs, model)

    assert kwargs["api_key"] == TEAM_A_TOKEN
    assert kwargs["api_base"] == CHATGPT_API_BASE
    assert "chatgpt_auth_file" not in kwargs


@pytest.mark.parametrize(
    "secret_fields",
    [
        _owned((TEAM_B_LOGIN, TEAM_B_TOKEN)),
        _owned(),
        _owned((TEAM_A_LOGIN, "")),
        {"raw_headers": {}},
        None,
    ],
    ids=["another_teams_login", "no_logins", "empty_token", "no_login_field", "no_secret_fields"],
)
def test_a_login_the_caller_does_not_own_fails_closed_with_a_401(secret_fields: dict[str, object] | None) -> None:
    kwargs: Final = _request(secret_fields=secret_fields) if secret_fields is not None else _request()
    if secret_fields is None:
        kwargs.pop("secret_fields")
    before: Final = copy.deepcopy(kwargs)

    with pytest.raises(CallerCredentialAuthenticationError) as raised:
        attach_chatgpt_subscription_login(kwargs, "chatgpt/gpt-5.5")

    assert raised.value.status_code == 401
    assert TEAM_A_LOGIN in str(raised.value)
    assert TEAM_A_TOKEN not in str(raised.value)
    assert TEAM_B_TOKEN not in str(raised.value)
    assert kwargs == before


def test_a_chatgpt_deployment_naming_a_missing_credential_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "credential_list", [])
    kwargs: Final = _request()

    with pytest.raises(CallerCredentialAuthenticationError) as raised:
        attach_chatgpt_subscription_login(kwargs, "chatgpt/gpt-5.5")

    assert raised.value.status_code == 401
    assert TEAM_A_TOKEN not in str(raised.value)
    assert "api_key" not in kwargs
    assert kwargs["chatgpt_auth_file"] == "/host/chatgpt/auth.json"


@pytest.mark.parametrize(
    ("kwargs", "model"),
    [
        (
            {
                "model": "openai/gpt-5.4-mini",
                "litellm_credential_name": "deleted-credential",
                "secret_fields": _owned(),
            },
            "openai/gpt-5.4-mini",
        ),
        (
            {"model": "chatgpt/gpt-5.5", "litellm_credential_name": "plain-openai", "chatgpt_auth_file": "/a.json"},
            "chatgpt/gpt-5.5",
        ),
        (
            {
                "model": "chatgpt/gpt-5.5",
                "litellm_credential_name": "plain-openai",
                "secret_fields": _owned(("plain-openai", TEAM_A_TOKEN)),
            },
            "chatgpt/gpt-5.5",
        ),
        ({"model": "chatgpt/gpt-5.5", "secret_fields": _owned((TEAM_A_LOGIN, TEAM_A_TOKEN))}, "chatgpt/gpt-5.5"),
        ({"model": "chatgpt/gpt-5.5", "litellm_credential_name": ""}, "chatgpt/gpt-5.5"),
    ],
    ids=[
        "missing_credential_on_another_provider",
        "ordinary_credential",
        "ordinary_credential_with_a_same_named_login",
        "no_credential_name",
        "empty_credential_name",
    ],
)
def test_requests_that_name_no_subscription_login_are_left_untouched(
    kwargs: dict[str, object], model: str
) -> None:
    before: Final = copy.deepcopy(kwargs)

    attach_chatgpt_subscription_login(kwargs, model)

    assert kwargs == before


@pytest.mark.parametrize(
    ("model", "overrides"),
    [
        ("openai/gpt-5.4-mini", {"model": "openai/gpt-5.4-mini"}),
        ("gpt-5.4-mini", {"model": "gpt-5.4-mini", "custom_llm_provider": "openai"}),
    ],
)
def test_a_subscription_login_is_never_sent_to_another_provider(model: str, overrides: dict[str, object]) -> None:
    kwargs: Final = _request(**overrides)

    with pytest.raises(CallerCredentialAuthenticationError) as raised:
        attach_chatgpt_subscription_login(kwargs, model)

    assert raised.value.status_code == 401
    assert TEAM_A_TOKEN not in str(raised.value)
    assert "api_key" not in kwargs


def _login_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {CLIENT_TOKEN}",
        "chatgpt-account-id": "acct-client",
        "originator": "codex_cli_rs",
    }


def _header_names(headers: object) -> set[str]:
    assert isinstance(headers, Mapping)
    return {str(name).lower() for name in headers}


@pytest.mark.parametrize("scoped_as_list", [False, True])
def test_attach_strips_a_forwarded_chatgpt_login_from_every_header_carrier(scoped_as_list: bool) -> None:
    scoped: Final = {"custom_llm_provider": "chatgpt", "extra_headers": _login_headers()}
    kwargs: Final = _request(
        provider_specific_header=[scoped, {"custom_llm_provider": "openai", "extra_headers": _login_headers()}]
        if scoped_as_list
        else scoped,
        extra_headers={"AUTHORIZATION": f"Bearer {CLIENT_TOKEN}", "ChatGPT-Account-Id": "acct-client", "x-trace": "1"},
        headers={"authorization": f"Bearer {CLIENT_TOKEN}", "x-trace": "1"},
        optional_params={"extra_headers": _login_headers(), "temperature": 0.2},
        litellm_params={"extra_headers": _login_headers(), "metadata": {"tag": "kept"}},
    )

    attach_chatgpt_subscription_login(kwargs, "chatgpt/gpt-5.5")

    entries: Final = kwargs["provider_specific_header"] if scoped_as_list else [kwargs["provider_specific_header"]]
    assert isinstance(entries, list)
    expected_providers: Final = ["chatgpt", "openai"] if scoped_as_list else ["chatgpt"]
    assert [entry["custom_llm_provider"] for entry in entries] == expected_providers
    assert all(_header_names(entry["extra_headers"]) == {"originator"} for entry in entries)
    assert _header_names(kwargs["extra_headers"]) == {"x-trace"}
    assert _header_names(kwargs["headers"]) == {"x-trace"}
    assert kwargs["optional_params"] == {"extra_headers": {"originator": "codex_cli_rs"}, "temperature": 0.2}
    assert kwargs["litellm_params"] == {"extra_headers": {"originator": "codex_cli_rs"}, "metadata": {"tag": "kept"}}
    assert kwargs["api_key"] == TEAM_A_TOKEN
    assert CLIENT_TOKEN not in json.dumps({key: value for key, value in kwargs.items() if key != "secret_fields"})


@pytest.mark.parametrize(
    "overrides",
    [
        {"api_base": "https://attacker.example/backend-api/codex"},
        {"base_url": "https://attacker.example/backend-api/codex"},
        {"api_base": "http://chatgpt.com/backend-api/codex"},
        {"api_base": "https://chatgpt.com.attacker.example/backend-api/codex"},
        {"base_url": CHATGPT_API_BASE, "api_base": "https://attacker.example/backend-api/codex"},
        {"base_url": "https://attacker.example/backend-api/codex", "api_base": CHATGPT_API_BASE},
    ],
)
def test_a_client_base_off_chatgpt_is_refused_before_the_login_is_attached(overrides: dict[str, object]) -> None:
    kwargs: Final = _request(**overrides)
    before: Final = copy.deepcopy(kwargs)

    with pytest.raises(ChatGPTAuthError) as raised:
        attach_chatgpt_subscription_login(kwargs, "chatgpt/gpt-5.5")

    assert raised.value.status_code == 400
    assert TEAM_A_TOKEN not in str(raised.value)
    assert kwargs == before


@pytest.mark.parametrize("field", ["api_base", "base_url"])
def test_a_client_base_on_chatgpt_is_kept_as_api_base_and_base_url_is_removed(field: str) -> None:
    client_base: Final = "https://chatgpt.com:443/backend-api/codex"
    kwargs: Final = _request(**{field: client_base})

    attach_chatgpt_subscription_login(kwargs, "chatgpt/gpt-5.5")

    assert kwargs["api_base"] == client_base
    assert "base_url" not in kwargs
    assert kwargs["api_key"] == TEAM_A_TOKEN


def test_a_caller_api_base_override_on_a_subscription_deployment_never_sends_the_login_off_chatgpt() -> None:
    redirected: Final = get_dynamic_litellm_params(
        litellm_params={"model": "chatgpt/gpt-5.5", "litellm_credential_name": TEAM_A_LOGIN},
        request_kwargs={"api_base": "https://attacker.example/backend-api/codex"},
    )
    kwargs: Final = {**redirected, "secret_fields": _owned((TEAM_A_LOGIN, TEAM_A_TOKEN))}

    with pytest.raises(ChatGPTAuthError):
        attach_chatgpt_subscription_login(kwargs, "chatgpt/gpt-5.5")

    assert "api_key" not in kwargs


@pytest.mark.parametrize(
    ("credential_name", "expected"),
    [(TEAM_A_LOGIN, True), ("plain-openai", False), ("deleted-credential", False), ("", False), (None, False)],
)
def test_names_chatgpt_subscription_login(credential_name: object, expected: bool) -> None:
    assert names_chatgpt_subscription_login(credential_name) is expected


def test_attach_is_repeatable_on_the_same_request() -> None:
    kwargs: Final = _request()

    attach_chatgpt_subscription_login(kwargs, "chatgpt/gpt-5.5")
    first: Final = dict(kwargs)
    attach_chatgpt_subscription_login(kwargs, "chatgpt/gpt-5.5")

    assert kwargs == first
    assert kwargs["api_key"] == TEAM_A_TOKEN
