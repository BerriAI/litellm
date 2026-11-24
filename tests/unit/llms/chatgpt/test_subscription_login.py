import base64
import copy
import json
from collections.abc import Callable, Mapping
from typing import Final

import pytest

import litellm
from litellm.exceptions import CallerCredentialAuthenticationError
from litellm.llms.chatgpt.authenticator import TOKEN_EXPIRY_SKEW_SECONDS
from litellm.llms.chatgpt.common_utils import CHATGPT_API_BASE, ChatGPTAuthError
from litellm.llms.chatgpt.subscription_login import (
    aattach_chatgpt_subscription_login,
    attach_chatgpt_subscription_login,
    chatgpt_login_probe_params,
    chatgpt_subscription_owner,
    names_chatgpt_subscription_login,
)
from litellm.litellm_core_utils.clientside_credential_handler import get_dynamic_litellm_params
from litellm.types.proxy.litellm_pre_call_utils import RedactedDict
from litellm.types.utils import CredentialItem


def _jwt(claims: Mapping[str, object]) -> str:
    def segment(value: Mapping[str, object]) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()

    return f"{segment({'alg': 'RS256'})}.{segment(claims)}.fake-signature"


EXPIRES_AT: Final = 4102444800
TEAM_A_LOGIN: Final = "team-a-chatgpt"
TEAM_B_LOGIN: Final = "team-b-chatgpt"
KEY_K_LOGIN: Final = "key-k-chatgpt"
TEAM_A_TOKEN: Final = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-team-a"}, "exp": EXPIRES_AT})
TEAM_B_TOKEN: Final = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-team-b"}, "exp": EXPIRES_AT})
KEY_K_TOKEN: Final = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-key-k"}, "exp": EXPIRES_AT})
CLIENT_TOKEN: Final = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-client"}})
REFRESHED_TEAM_A_TOKEN: Final = _jwt(
    {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-team-a"}, "exp": 4102444800, "jti": "refreshed"}
)


def _subscription(name: str, access_token: str, **owner: str) -> CredentialItem:
    return CredentialItem(
        credential_name=name,
        credential_values={
            "chatgpt_access_token": access_token,
            "chatgpt_refresh_token": f"refresh-{name}",
            "chatgpt_id_token": access_token,
            "chatgpt_account_id": f"acct-{name}",
            "chatgpt_expires_at": str(EXPIRES_AT),
        },
        credential_info={"custom_llm_provider": "chatgpt", **owner},
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
            _subscription(TEAM_A_LOGIN, TEAM_A_TOKEN, chatgpt_owner_team_id="team-a"),
            _subscription(TEAM_B_LOGIN, TEAM_B_TOKEN, chatgpt_owner_team_id="team-b"),
            _subscription(KEY_K_LOGIN, KEY_K_TOKEN, chatgpt_owner_key_hash="hash-k"),
            ORDINARY,
        ],
    )


def _caller(team_id: str | None, key_hash: str | None) -> dict[str, object]:
    return {
        "raw_headers": {},
        "chatgpt_subscription_owner": RedactedDict(dict(chatgpt_subscription_owner(team_id, key_hash))),
    }


def _at(now: float) -> Callable[[], float]:
    return lambda: now


def _request(**overrides: object) -> dict[str, object]:
    return {
        "model": "chatgpt/gpt-5.5",
        "litellm_credential_name": TEAM_A_LOGIN,
        "chatgpt_auth_file": "/host/chatgpt/auth.json",
        "secret_fields": _caller("team-a", None),
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
        _caller("team-b", None),
        _caller(None, "hash-k"),
        _caller(None, None),
        {"raw_headers": {}},
        None,
    ],
    ids=["another_teams_owner", "another_keys_owner", "empty_owner", "no_owner_field", "no_secret_fields"],
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
                "secret_fields": _caller("team-a", None),
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
                "secret_fields": _caller("team-a", None),
            },
            "chatgpt/gpt-5.5",
        ),
        ({"model": "chatgpt/gpt-5.5", "secret_fields": _caller("team-a", None)}, "chatgpt/gpt-5.5"),
        ({"model": "chatgpt/gpt-5.5", "litellm_credential_name": ""}, "chatgpt/gpt-5.5"),
    ],
    ids=[
        "missing_credential_on_another_provider",
        "ordinary_credential",
        "ordinary_credential_named_by_a_login_owner",
        "no_credential_name",
        "empty_credential_name",
    ],
)
def test_requests_that_name_no_subscription_login_are_left_untouched(kwargs: dict[str, object], model: str) -> None:
    before: Final = copy.deepcopy(kwargs)

    attach_chatgpt_subscription_login(kwargs, model)

    assert kwargs == before


@pytest.mark.parametrize(
    ("model", "overrides"),
    [
        ("openai/gpt-5.4-mini", {"model": "openai/gpt-5.4-mini"}),
        ("gpt-5.4-mini", {"model": "gpt-5.4-mini", "custom_llm_provider": "openai"}),
        ("chatgpt/gpt-5.5", {"custom_llm_provider": "gigachat", "gigachat_auth_url": "https://auth.example.com"}),
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
    kwargs: Final = {**redirected, "secret_fields": _caller("team-a", None)}

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


class _Refresh:
    def __init__(self, access_token: str | None) -> None:
        self.access_token = access_token
        self.calls: list[tuple[str, str | None, str | None]] = []

    async def __call__(self, credential_name: str, team_id: str | None, key_hash: str | None) -> str | None:
        self.calls.append((credential_name, team_id, key_hash))
        return self.access_token


async def test_async_attach_refreshes_the_callers_stale_login_once_and_sends_it() -> None:
    refresh: Final = _Refresh(REFRESHED_TEAM_A_TOKEN)
    kwargs: Final = _request(secret_fields=_caller("team-a", "hash-unbound"))

    await aattach_chatgpt_subscription_login(kwargs, "chatgpt/gpt-5.5", refresh=refresh, clock=_at(EXPIRES_AT))

    assert refresh.calls == [(TEAM_A_LOGIN, "team-a", "hash-unbound")]
    assert kwargs["api_key"] == REFRESHED_TEAM_A_TOKEN
    assert kwargs["api_base"] == CHATGPT_API_BASE
    assert "chatgpt_auth_file" not in kwargs


async def test_async_attach_sends_a_fresh_login_without_refreshing_it() -> None:
    refresh: Final = _Refresh(REFRESHED_TEAM_A_TOKEN)
    kwargs: Final = _request(secret_fields=_caller("team-a", None))

    await aattach_chatgpt_subscription_login(kwargs, "chatgpt/gpt-5.5", refresh=refresh)

    assert refresh.calls == []
    assert kwargs["api_key"] == TEAM_A_TOKEN


async def test_async_attach_fails_a_login_that_could_not_be_refreshed_closed_with_a_401() -> None:
    refresh: Final = _Refresh(None)
    kwargs: Final = _request(secret_fields=_caller("team-a", None))
    before: Final = copy.deepcopy(kwargs)

    with pytest.raises(CallerCredentialAuthenticationError) as raised:
        await aattach_chatgpt_subscription_login(kwargs, "chatgpt/gpt-5.5", refresh=refresh, clock=_at(EXPIRES_AT))

    assert raised.value.status_code == 401
    assert refresh.calls == [(TEAM_A_LOGIN, "team-a", None)]
    assert kwargs == before


@pytest.mark.parametrize("sync", [True, False], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("credential_name", "owner", "token"),
    [
        (TEAM_A_LOGIN, ("team-a", "hash-unbound"), TEAM_A_TOKEN),
        (KEY_K_LOGIN, ("team-unbound", "hash-k"), KEY_K_TOKEN),
    ],
    ids=["team_owner", "key_owner"],
)
async def test_attach_binds_the_owners_fresh_token_from_the_named_credential_itself(
    sync: bool, credential_name: str, owner: tuple[str | None, str | None], token: str
) -> None:
    refresh: Final = _Refresh(REFRESHED_TEAM_A_TOKEN)
    kwargs: Final = _request(litellm_credential_name=credential_name, secret_fields=_caller(*owner))
    clock: Final = _at(EXPIRES_AT - TOKEN_EXPIRY_SKEW_SECONDS - 1)

    if sync:
        attach_chatgpt_subscription_login(kwargs, "chatgpt/gpt-5.5", clock=clock)
    else:
        await aattach_chatgpt_subscription_login(kwargs, "chatgpt/gpt-5.5", refresh=refresh, clock=clock)

    assert kwargs["api_key"] == token
    assert refresh.calls == []


@pytest.mark.parametrize("owner", [("team-b", None), (None, "hash-k")], ids=["another_team", "another_key"])
async def test_async_attach_refuses_a_caller_that_does_not_own_the_named_login_with_a_401(
    owner: tuple[str | None, str | None],
) -> None:
    refresh: Final = _Refresh(None)
    kwargs: Final = _request(secret_fields=_caller(*owner))
    before: Final = copy.deepcopy(kwargs)

    with pytest.raises(CallerCredentialAuthenticationError) as raised:
        await aattach_chatgpt_subscription_login(kwargs, "chatgpt/gpt-5.5", refresh=refresh)

    assert raised.value.status_code == 401
    assert refresh.calls == [(TEAM_A_LOGIN, *owner)]
    assert TEAM_A_TOKEN not in str(raised.value)
    assert kwargs == before


@pytest.mark.parametrize(
    ("now", "sent", "refreshed"),
    [
        (EXPIRES_AT - TOKEN_EXPIRY_SKEW_SECONDS - 1, TEAM_A_TOKEN, False),
        (EXPIRES_AT - TOKEN_EXPIRY_SKEW_SECONDS, REFRESHED_TEAM_A_TOKEN, True),
        (EXPIRES_AT + 1, REFRESHED_TEAM_A_TOKEN, True),
    ],
    ids=["fresh", "inside_the_expiry_skew", "expired"],
)
async def test_async_attach_sends_the_stored_token_only_while_it_is_fresh_and_refreshes_it_after(
    now: float, sent: str, refreshed: bool
) -> None:
    refresh: Final = _Refresh(REFRESHED_TEAM_A_TOKEN)
    kwargs: Final = _request()

    await aattach_chatgpt_subscription_login(kwargs, "chatgpt/gpt-5.5", refresh=refresh, clock=_at(now))

    assert kwargs["api_key"] == sent
    assert refresh.calls == ([(TEAM_A_LOGIN, "team-a", None)] if refreshed else [])


@pytest.mark.parametrize("now", [EXPIRES_AT - TOKEN_EXPIRY_SKEW_SECONDS, EXPIRES_AT + 1], ids=["skew", "expired"])
def test_sync_attach_fails_an_owned_login_whose_stored_token_is_stale_closed_with_a_401(now: float) -> None:
    kwargs: Final = _request()
    before: Final = copy.deepcopy(kwargs)

    with pytest.raises(CallerCredentialAuthenticationError) as raised:
        attach_chatgpt_subscription_login(kwargs, "chatgpt/gpt-5.5", clock=_at(now))

    assert raised.value.status_code == 401
    assert TEAM_A_TOKEN not in str(raised.value)
    assert kwargs == before


@pytest.mark.parametrize(
    ("kwargs", "model"),
    [
        (
            {"model": "openai/gpt-5.4-mini", "litellm_credential_name": "plain-openai"},
            "openai/gpt-5.4-mini",
        ),
        ({"model": "chatgpt/gpt-5.5", "litellm_credential_name": "plain-openai"}, "chatgpt/gpt-5.5"),
        ({"model": "openai/gpt-5.4-mini", "litellm_credential_name": "deleted-credential"}, "openai/gpt-5.4-mini"),
        ({"model": "chatgpt/gpt-5.5"}, "chatgpt/gpt-5.5"),
    ],
    ids=["another_provider", "ordinary_credential_on_chatgpt", "missing_credential_on_another_provider", "no_name"],
)
async def test_async_attach_never_refreshes_for_a_call_that_names_no_subscription_login(
    kwargs: dict[str, object], model: str
) -> None:
    refresh: Final = _Refresh(REFRESHED_TEAM_A_TOKEN)
    request: Final = {**kwargs, "secret_fields": _caller("team-a", "hash-k")}
    before: Final = copy.deepcopy(request)

    await aattach_chatgpt_subscription_login(request, model, refresh=refresh)

    assert refresh.calls == []
    assert request == before


@pytest.mark.parametrize(
    ("secret_fields", "model"),
    [
        ({"raw_headers": {}}, "chatgpt/gpt-5.5"),
        (_caller("team-a", None), "openai/gpt-5.4-mini"),
    ],
    ids=["no_owner_identity", "another_provider"],
)
async def test_async_attach_fails_a_login_the_caller_cannot_use_closed_without_refreshing_it(
    secret_fields: dict[str, object], model: str
) -> None:
    refresh: Final = _Refresh(REFRESHED_TEAM_A_TOKEN)
    kwargs: Final = _request(model=model, secret_fields=secret_fields)
    before: Final = copy.deepcopy(kwargs)

    with pytest.raises(CallerCredentialAuthenticationError) as raised:
        await aattach_chatgpt_subscription_login(kwargs, model, refresh=refresh)

    assert raised.value.status_code == 401
    assert TEAM_A_LOGIN in str(raised.value)
    assert refresh.calls == []
    assert kwargs == before


@pytest.mark.parametrize(
    "credential_name",
    [TEAM_A_LOGIN, "signed-in-on-another-replica"],
    ids=["held_here_under_another_owner", "not_loaded_here_yet"],
)
async def test_async_attach_asks_the_proxy_for_a_login_this_replica_does_not_show_the_caller_owning(
    credential_name: str,
) -> None:
    refresh: Final = _Refresh(REFRESHED_TEAM_A_TOKEN)
    kwargs: Final = _request(litellm_credential_name=credential_name, secret_fields=_caller(None, "hash-regenerated"))

    await aattach_chatgpt_subscription_login(kwargs, "chatgpt/gpt-5.5", refresh=refresh)

    assert refresh.calls == [(credential_name, None, "hash-regenerated")]
    assert kwargs["api_key"] == REFRESHED_TEAM_A_TOKEN


@pytest.mark.parametrize(
    "probing_owners",
    [None, (("team-a", "hash-unbound"),), (("team-c", "hash-k"), ("team-a", None))],
    ids=["unrestricted", "the_owning_team", "one_prober_owns_it"],
)
async def test_a_probe_of_a_team_owned_login_runs_as_that_team(
    probing_owners: tuple[tuple[str | None, str | None], ...] | None,
) -> None:
    refresh: Final = _Refresh(REFRESHED_TEAM_A_TOKEN)
    params: Final = {"model": "chatgpt/gpt-5.5", "litellm_credential_name": TEAM_A_LOGIN}
    probe: Final = dict(chatgpt_login_probe_params(params, probing_owners))

    await aattach_chatgpt_subscription_login(probe, "chatgpt/gpt-5.5", refresh=refresh, clock=_at(EXPIRES_AT))

    assert refresh.calls == [(TEAM_A_LOGIN, "team-a", None)]
    assert probe["api_key"] == REFRESHED_TEAM_A_TOKEN


async def test_a_probe_by_a_caller_that_does_not_own_the_login_gets_no_owner_and_fails_closed() -> None:
    refresh: Final = _Refresh(REFRESHED_TEAM_A_TOKEN)
    params: Final = {"model": "chatgpt/gpt-5.5", "litellm_credential_name": TEAM_A_LOGIN}
    probe: Final = dict(chatgpt_login_probe_params(params, (("team-b", "hash-k"), ("team-c", None))))

    assert probe == params
    with pytest.raises(CallerCredentialAuthenticationError):
        await aattach_chatgpt_subscription_login(probe, "chatgpt/gpt-5.5", refresh=refresh)
    assert refresh.calls == []


@pytest.mark.parametrize(
    "params",
    [
        {"model": "openai/gpt-5.4-mini", "litellm_credential_name": "plain-openai"},
        {"model": "chatgpt/gpt-5.5", "litellm_credential_name": "deleted-credential"},
        {"model": "chatgpt/gpt-5.5", "secret_fields": {"raw_headers": {}}},
    ],
    ids=["ordinary_credential", "missing_credential", "no_credential_name"],
)
def test_a_probe_that_names_no_subscription_login_gets_no_owner(params: dict[str, object]) -> None:
    assert chatgpt_login_probe_params(params) == params


@pytest.mark.parametrize(
    "owner_info",
    [{}, {"chatgpt_owner_team_id": "team-a", "chatgpt_owner_key_hash": "hash-k"}],
    ids=["no_owner", "two_owners"],
)
def test_a_probe_of_a_login_without_exactly_one_owner_gets_no_owner(
    monkeypatch: pytest.MonkeyPatch, owner_info: dict[str, str]
) -> None:
    login: Final = _subscription(TEAM_A_LOGIN, TEAM_A_TOKEN, chatgpt_owner_team_id="team-a").model_copy(
        update={"credential_info": {"custom_llm_provider": "chatgpt", **owner_info}}
    )
    monkeypatch.setattr(litellm, "credential_list", [login])
    params: Final = {"model": "chatgpt/gpt-5.5", "litellm_credential_name": TEAM_A_LOGIN}

    assert chatgpt_login_probe_params(params) == params
