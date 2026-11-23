from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.exceptions import BadRequestError, CallerCredentialAuthenticationError
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.llms.chatgpt.common_utils import CHATGPT_ACCOUNT_ID_HEADER, CHATGPT_API_BASE, chatgpt_client_api_base
from litellm.types.proxy.litellm_pre_call_utils import RedactedDict
from litellm.types.utils import CredentialItem, chatgpt_subscription_login_litellm_params

CHATGPT_SUBSCRIPTION_LOGINS_SECRET_FIELD: Final = "chatgpt_subscription_logins"
CHATGPT_SUBSCRIPTION_OWNER_SECRET_FIELD: Final = "chatgpt_subscription_owner"
CHATGPT_OWNER_TEAM_ID_KEY: Final = "chatgpt_owner_team_id"
CHATGPT_OWNER_KEY_HASH_KEY: Final = "chatgpt_owner_key_hash"
CHATGPT_PROVIDER: Final = "chatgpt"
_LOGIN_HEADER_NAMES: Final = frozenset({"authorization", CHATGPT_ACCOUNT_ID_HEADER.lower()})
_STR_KEYED: Final = TypeAdapter(Mapping[str, object])
_ENTRIES: Final = TypeAdapter(tuple[object, ...])
_NOTHING: Final[Mapping[str, object]] = MappingProxyType({})

ChatGPTLoginRefresh: TypeAlias = Callable[[str, str | None, str | None], Awaitable[str | None]]


class _CredentialInfoView(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    credential_info: Mapping[str, object]


def is_chatgpt_subscription_login(credential: CredentialItem) -> bool:
    return any(key in credential.credential_values for key in chatgpt_subscription_login_litellm_params)


def _non_empty_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def chatgpt_credential_info(credential: CredentialItem | None) -> Mapping[str, object]:
    if credential is None:
        return _NOTHING
    try:
        return _CredentialInfoView.model_validate(credential).credential_info
    except ValidationError:
        return _NOTHING


def chatgpt_login_owner(credential: CredentialItem) -> tuple[str | None, str | None]:
    info: Final = chatgpt_credential_info(credential)
    team_id: Final = _non_empty_str(info.get(CHATGPT_OWNER_TEAM_ID_KEY))
    key_hash: Final = _non_empty_str(info.get(CHATGPT_OWNER_KEY_HASH_KEY))
    if (team_id is None) == (key_hash is None):
        return None, None
    return team_id, key_hash


def owns_chatgpt_login(credential: CredentialItem, team_id: str | None, key_hash: str | None) -> bool:
    if not is_chatgpt_subscription_login(credential):
        return False
    owner_team_id, owner_key_hash = chatgpt_login_owner(credential)
    return (owner_team_id is not None and owner_team_id == team_id) or (
        owner_key_hash is not None and owner_key_hash == key_hash
    )


def chatgpt_subscription_owner(team_id: str | None, key_hash: str | None) -> Mapping[str, str]:
    return MappingProxyType(
        {
            **({CHATGPT_OWNER_TEAM_ID_KEY: team_id} if team_id else {}),
            **({CHATGPT_OWNER_KEY_HASH_KEY: key_hash} if key_hash else {}),
        }
    )


def chatgpt_login_probe_params(
    probe_params: Mapping[str, object], probing_owners: tuple[tuple[str | None, str | None], ...] | None = None
) -> Mapping[str, object]:
    credential_name: Final = probe_params.get("litellm_credential_name")
    credential: Final = (
        CredentialAccessor.find_credential(credential_name) if isinstance(credential_name, str) else None
    )
    if credential is None or not is_chatgpt_subscription_login(credential):
        return probe_params
    owner: Final = chatgpt_login_owner(credential)
    if owner == (None, None) or (
        probing_owners is not None and not any(owns_chatgpt_login(credential, *prober) for prober in probing_owners)
    ):
        return probe_params
    return MappingProxyType(
        {
            **probe_params,
            "secret_fields": {
                **(_str_keyed(probe_params.get("secret_fields")) or _NOTHING),
                CHATGPT_SUBSCRIPTION_OWNER_SECRET_FIELD: RedactedDict(chatgpt_subscription_owner(*owner)),
            },
        }
    )


def names_chatgpt_subscription_login(credential_name: object) -> bool:
    if not isinstance(credential_name, str) or not credential_name:
        return False
    credential: Final = CredentialAccessor.find_credential(credential_name)
    return credential is not None and is_chatgpt_subscription_login(credential)


def locked_chatgpt_credential_name(deployment_params: object) -> str | None:
    params: Final = _str_keyed(deployment_params)
    credential_name: Final = params.get("litellm_credential_name") if params is not None else None
    if params is None or not isinstance(credential_name, str) or not credential_name:
        return None
    if names_chatgpt_subscription_login(credential_name) or _is_chatgpt_call(params, params.get("model")):
        return credential_name
    return None


def reject_chatgpt_credential_override(
    deployment_params: object, request_kwargs: Mapping[str, object], model: str
) -> None:
    locked_name: Final = locked_chatgpt_credential_name(deployment_params)
    if (
        locked_name is None
        or "litellm_credential_name" not in request_kwargs
        or request_kwargs["litellm_credential_name"] == locked_name
    ):
        return
    raise BadRequestError(
        message="litellm_credential_name cannot be overridden on a deployment that uses a ChatGPT subscription login",
        model=model,
        llm_provider="",
    )


def _str_keyed(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return _STR_KEYED.validate_python(value)
    except ValidationError:
        return None


def _is_chatgpt_call(kwargs: Mapping[str, object], model: object) -> bool:
    match kwargs.get("custom_llm_provider"):
        case None | "":
            return any(
                isinstance(candidate, str) and candidate.startswith(f"{CHATGPT_PROVIDER}/")
                for candidate in (model, kwargs.get("model"))
            )
        case provider:
            return provider == CHATGPT_PROVIDER


def _login_unavailable(credential_name: str) -> CallerCredentialAuthenticationError:
    return CallerCredentialAuthenticationError(
        message=(
            f"ChatGPT subscription login '{credential_name}' does not exist or is not available to this team or key. "
            "An admin can sign in to ChatGPT for this team or key under LLM Credentials in the LiteLLM UI"
        ),
        llm_provider=CHATGPT_PROVIDER,
        model="",
    )


def _secret_field(kwargs: Mapping[str, object], field: str) -> Mapping[str, object]:
    secret_fields: Final = _str_keyed(kwargs.get("secret_fields"))
    return (_str_keyed(secret_fields.get(field)) if secret_fields is not None else None) or _NOTHING


@dataclass(frozen=True, slots=True)
class _NamedLogin:
    credential_name: str
    credential: CredentialItem | None


def _owned_access_token(kwargs: Mapping[str, object], login: _NamedLogin) -> str | None:
    if login.credential is None:
        return None
    return _non_empty_str(_secret_field(kwargs, CHATGPT_SUBSCRIPTION_LOGINS_SECRET_FIELD).get(login.credential_name))


def _caller_owner(kwargs: Mapping[str, object]) -> tuple[str | None, str | None] | None:
    owner: Final = _secret_field(kwargs, CHATGPT_SUBSCRIPTION_OWNER_SECRET_FIELD)
    team_id: Final = _non_empty_str(owner.get(CHATGPT_OWNER_TEAM_ID_KEY))
    key_hash: Final = _non_empty_str(owner.get(CHATGPT_OWNER_KEY_HASH_KEY))
    return None if team_id is None and key_hash is None else (team_id, key_hash)


def _without_login_headers(headers: Mapping[str, object]) -> Mapping[str, object]:
    return {key: value for key, value in headers.items() if key.lower() not in _LOGIN_HEADER_NAMES}


def _scoped_header_without_login(entry: object) -> object:
    scoped: Final = _str_keyed(entry)
    extra_headers: Final = _str_keyed(scoped.get("extra_headers")) if scoped is not None else None
    if scoped is None or extra_headers is None:
        return entry
    return {**scoped, "extra_headers": _without_login_headers(extra_headers)}


def _provider_specific_header_without_login(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return [_scoped_header_without_login(entry) for entry in _ENTRIES.validate_python(value)]
    return _scoped_header_without_login(value)


def _nested_extra_headers_without_login(params: object) -> object:
    mapping: Final = _str_keyed(params)
    extra_headers: Final = _str_keyed(mapping.get("extra_headers")) if mapping is not None else None
    if mapping is None or extra_headers is None:
        return params
    return {**mapping, "extra_headers": _without_login_headers(extra_headers)}


def _header_mapping_without_login(value: object) -> object:
    headers: Final = _str_keyed(value)
    return value if headers is None else _without_login_headers(headers)


def _subscription_api_base(kwargs: Mapping[str, object]) -> str:
    validated: Final = tuple(
        chatgpt_client_api_base(value)
        for value in (kwargs.get("base_url"), kwargs.get("api_base"))
        if isinstance(value, str) and value
    )
    return validated[0] if validated else CHATGPT_API_BASE


_HEADER_CARRIERS: Final[tuple[tuple[str, Callable[[object], object]], ...]] = (
    ("provider_specific_header", _provider_specific_header_without_login),
    ("extra_headers", _header_mapping_without_login),
    ("headers", _header_mapping_without_login),
    ("optional_params", _nested_extra_headers_without_login),
    ("litellm_params", _nested_extra_headers_without_login),
)


def _request_updates(kwargs: Mapping[str, object], access_token: str) -> Mapping[str, object]:
    return MappingProxyType(
        {
            **{key: rebuild(kwargs[key]) for key, rebuild in _HEADER_CARRIERS if key in kwargs},
            "api_key": access_token,
            "api_base": _subscription_api_base(kwargs),
        }
    )


def _named_login(kwargs: Mapping[str, object], model: object) -> _NamedLogin | None:
    credential_name: Final = kwargs.get("litellm_credential_name")
    if not isinstance(credential_name, str) or not credential_name:
        return None
    credential: Final = CredentialAccessor.find_credential(credential_name)
    if credential is None and not _is_chatgpt_call(kwargs, model):
        return None
    if credential is not None and not is_chatgpt_subscription_login(credential):
        return None
    return _NamedLogin(credential_name=credential_name, credential=credential)


def _bind_login(
    kwargs: dict[str, object],  # mutable-ok: kwargs is the request mutation channel
    model: object,
    credential_name: str,
    access_token: str | None,
) -> None:
    if access_token is None or not _is_chatgpt_call(kwargs, model):
        raise _login_unavailable(credential_name)
    updates: Final = _request_updates(kwargs, access_token)
    kwargs.pop("chatgpt_auth_file", None)
    kwargs.pop("base_url", None)
    kwargs.update(updates)


def attach_chatgpt_subscription_login(
    kwargs: dict[str, object],  # mutable-ok: kwargs is the request mutation channel
    model: object,
) -> None:
    login: Final = _named_login(kwargs, model)
    if login is None:
        return
    _bind_login(kwargs, model, login.credential_name, _owned_access_token(kwargs, login))


async def _refresh_through_proxy(credential_name: str, team_id: str | None, key_hash: str | None) -> str | None:
    from litellm.proxy.credential_endpoints.chatgpt_subscription_logins import (
        owned_chatgpt_subscription_access_token,
    )

    return await owned_chatgpt_subscription_access_token(credential_name, team_id, key_hash)


async def _refreshed_access_token(
    kwargs: Mapping[str, object], model: object, credential_name: str, refresh: ChatGPTLoginRefresh
) -> str | None:
    owner: Final = _caller_owner(kwargs)
    if owner is None or not _is_chatgpt_call(kwargs, model):
        return None
    return await refresh(credential_name, *owner)


async def aattach_chatgpt_subscription_login(
    kwargs: dict[str, object],  # mutable-ok: kwargs is the request mutation channel
    model: object,
    refresh: ChatGPTLoginRefresh = _refresh_through_proxy,
) -> None:
    login: Final = _named_login(kwargs, model)
    if login is None:
        return
    access_token: Final = _owned_access_token(kwargs, login) or await _refreshed_access_token(
        kwargs, model, login.credential_name, refresh
    )
    _bind_login(kwargs, model, login.credential_name, access_token)
