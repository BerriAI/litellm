from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.exceptions import CallerCredentialAuthenticationError
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.llms.chatgpt.common_utils import CHATGPT_ACCOUNT_ID_HEADER, CHATGPT_API_BASE, chatgpt_client_api_base
from litellm.types.utils import CredentialItem, chatgpt_subscription_login_litellm_params

CHATGPT_SUBSCRIPTION_LOGINS_SECRET_FIELD: Final = "chatgpt_subscription_logins"
CHATGPT_OWNER_TEAM_ID_KEY: Final = "chatgpt_owner_team_id"
CHATGPT_OWNER_KEY_HASH_KEY: Final = "chatgpt_owner_key_hash"
CHATGPT_PROVIDER: Final = "chatgpt"
_LOGIN_HEADER_NAMES: Final = frozenset({"authorization", CHATGPT_ACCOUNT_ID_HEADER.lower()})
_STR_KEYED: Final = TypeAdapter(Mapping[str, object])
_ENTRIES: Final = TypeAdapter(tuple[object, ...])


def is_chatgpt_subscription_login(credential: CredentialItem) -> bool:
    return any(key in credential.credential_values for key in chatgpt_subscription_login_litellm_params)


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


def _str_keyed(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return _STR_KEYED.validate_python(value)
    except ValidationError:
        return None


def _is_chatgpt_call(kwargs: Mapping[str, object], model: object) -> bool:
    return kwargs.get("custom_llm_provider") == CHATGPT_PROVIDER or any(
        isinstance(candidate, str) and candidate.startswith(f"{CHATGPT_PROVIDER}/")
        for candidate in (model, kwargs.get("model"))
    )


def _login_unavailable(credential_name: str) -> CallerCredentialAuthenticationError:
    return CallerCredentialAuthenticationError(
        message=(
            f"ChatGPT subscription login '{credential_name}' is not available to this key. "
            "An admin can sign in to ChatGPT for this team or key under LLM Credentials in the LiteLLM UI"
        ),
        llm_provider=CHATGPT_PROVIDER,
        model="",
    )


def _owned_access_token(kwargs: Mapping[str, object], credential_name: str) -> str | None:
    secret_fields: Final = _str_keyed(kwargs.get("secret_fields"))
    logins: Final = _str_keyed(
        secret_fields.get(CHATGPT_SUBSCRIPTION_LOGINS_SECRET_FIELD) if secret_fields is not None else None
    )
    token: Final = logins.get(credential_name) if logins is not None else None
    return token if isinstance(token, str) and token else None


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


def attach_chatgpt_subscription_login(
    kwargs: dict[str, object],  # mutable-ok: kwargs is the request mutation channel
    model: object,
) -> None:
    credential_name: Final = kwargs.get("litellm_credential_name")
    if not isinstance(credential_name, str) or not credential_name:
        return
    credential: Final = CredentialAccessor.find_credential(credential_name)
    if credential is None and not _is_chatgpt_call(kwargs, model):
        return
    if credential is not None and not is_chatgpt_subscription_login(credential):
        return
    access_token: Final = _owned_access_token(kwargs, credential_name)
    if credential is None or access_token is None or not _is_chatgpt_call(kwargs, model):
        raise _login_unavailable(credential_name)
    updates: Final = _request_updates(kwargs, access_token)
    kwargs.pop("chatgpt_auth_file", None)
    kwargs.pop("base_url", None)
    kwargs.update(updates)
