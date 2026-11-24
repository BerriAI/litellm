from collections.abc import Mapping, Sequence
from itertools import chain, groupby
from types import MappingProxyType
from typing import Final, TypeVar

from pydantic import TypeAdapter

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.types.llms.anthropic import ANTHROPIC_USER_PROFILE_ID_HEADER
from litellm.types.utils import LlmProviders, ProviderSpecificHeader
from litellm.utils import supports_anthropic_user_profile_id

_PROVIDER_HEADERS: Final[TypeAdapter[ProviderSpecificHeader | Sequence[ProviderSpecificHeader] | None]] = TypeAdapter(
    ProviderSpecificHeader | Sequence[ProviderSpecificHeader] | None
)

_HeaderValue = TypeVar("_HeaderValue")
_CASE_INSENSITIVE_OVERRIDES: Final = frozenset({ANTHROPIC_USER_PROFILE_ID_HEADER})


def override_headers(
    headers: Mapping[str, _HeaderValue], overrides: Mapping[str, _HeaderValue]
) -> dict[str, _HeaderValue]:  # mutable-ok: legacy callers pass the merged headers on as a dict
    overridden: Final = frozenset(name.lower() for name in overrides) & _CASE_INSENSITIVE_OVERRIDES
    return {**{name: value for name, value in headers.items() if name.lower() not in overridden}, **overrides}


ANTHROPIC_USER_PROFILE_ID_PROVIDERS: Final = frozenset(
    (
        LlmProviders.ANTHROPIC.value,
        LlmProviders.AZURE_AI.value,
        LlmProviders.BEDROCK.value,
        LlmProviders.BEDROCK_MANTLE.value,
        LlmProviders.VERTEX_AI.value,
    )
)


def _gates_on_user_profile_id_support(scoped_header: ProviderSpecificHeader) -> bool:
    return bool(
        scoped_header.get("user_profile_id_models_only") or scoped_header.get("non_user_profile_id_models_only")
    )


def deployment_base_model(litellm_params: Mapping[str, object], model_info: object) -> str | None:
    base_model: Final = (as_str_mapping(model_info) or {}).get("base_model") or litellm_params.get("base_model")
    return base_model if isinstance(base_model, str) else None


def takes_anthropic_user_profile_id(
    model: str | None, custom_llm_provider: str | None, base_model: str | None = None
) -> bool:
    return any(
        supports_anthropic_user_profile_id(candidate, custom_llm_provider)
        for candidate in (base_model, model)
        if candidate
    )


def _applies_to_call(
    scoped_header: ProviderSpecificHeader,
    custom_llm_provider: str,
    native_anthropic_messages: bool,
    takes_user_profile_id: bool,
) -> bool:
    if scoped_header.get("native_anthropic_messages_only") and not native_anthropic_messages:
        return False
    if scoped_header.get("user_profile_id_models_only") and not takes_user_profile_id:
        return False
    if scoped_header.get("non_user_profile_id_models_only") and takes_user_profile_id:
        return False
    scoped_providers: Final = scoped_header.get("custom_llm_provider", "").split(",")
    return custom_llm_provider in (provider.strip() for provider in scoped_providers)


def _header_name(header: tuple[str, str]) -> str:
    return header[0]


def _text_values_by_name(headers: Mapping[str, object]) -> Mapping[str, tuple[str, ...]]:
    named: Final = sorted(
        ((name.lower(), value if isinstance(value, str) else "") for name, value in headers.items()), key=_header_name
    )
    return MappingProxyType(
        {name: tuple(value for _, value in values) for name, values in groupby(named, key=_header_name)}
    )


def _sent_values(values: tuple[str, ...], prefix: str) -> tuple[str, ...]:
    return values[:1] if values and values[0].startswith(prefix) else values


def _sends(
    values_by_name: Mapping[str, tuple[str, ...]], name: str, prefix: str, not_configured: frozenset[str]
) -> bool:
    return any(
        value.startswith(prefix) and value not in not_configured
        for value in _sent_values(values_by_name.get(name, ()), prefix)
    )


def _configures_a_listed_credential(
    scoped_header: ProviderSpecificHeader,
    sent_values: Mapping[str, tuple[str, ...]],
    contributed_values: frozenset[str],
) -> bool:
    credential_headers: Final = frozenset(
        name.lower() for name in scoped_header.get("configured_credential_headers", ())
    )
    if not credential_headers:
        return True
    prefix: Final = scoped_header.get("configured_credential_prefix", "")
    not_configured: Final = frozenset(scoped_header.get("caller_credentials", ())) | contributed_values
    configured: Final = any(_sends(sent_values, name, prefix, not_configured) for name in credential_headers)
    return configured != scoped_header.get("without_configured_credential", False)


def _contributed_headers(scoped_header: ProviderSpecificHeader, configured_names: frozenset[str]) -> Mapping[str, str]:
    headers: Final[Mapping[str, str]] = scoped_header.get("extra_headers", {})
    if not scoped_header.get("yields_to_configured"):
        return headers
    return {name: value for name, value in headers.items() if name.lower() not in configured_names}


_PROVIDER_HEADER: Final = TypeAdapter(ProviderSpecificHeader)


def _without_credential_conditions(scoped_header: ProviderSpecificHeader) -> ProviderSpecificHeader:
    return _PROVIDER_HEADER.validate_python({**scoped_header, "configured_credential_headers": ()})


def where_only_the_deployment_credential_authenticates(
    provider_specific_header: object,
) -> tuple[ProviderSpecificHeader, ...]:
    validated: Final = _PROVIDER_HEADERS.validate_python(provider_specific_header)
    scoped_headers: Final = (validated,) if isinstance(validated, dict) else validated or ()
    return tuple(
        _without_credential_conditions(scoped_header)
        for scoped_header in scoped_headers
        if not scoped_header.get("without_configured_credential")
    )


class ProviderSpecificHeaderUtils:
    @staticmethod
    def validated_provider_specific_headers(
        provider_specific_header: object, custom_llm_provider: str | None
    ) -> Mapping[str, str]:
        return ProviderSpecificHeaderUtils.get_provider_specific_headers(
            _PROVIDER_HEADERS.validate_python(provider_specific_header), custom_llm_provider
        )

    @staticmethod
    def with_provider_specific_headers(
        headers: Mapping[str, object],
        provider_specific_header: ProviderSpecificHeader | Sequence[ProviderSpecificHeader] | None,
        custom_llm_provider: str | None,
        model: str | None,
        base_model: object,
        model_info: object,
    ) -> dict[str, object]:  # mutable-ok: completion passes its request headers on as a dict
        return override_headers(
            headers,
            ProviderSpecificHeaderUtils.get_provider_specific_headers(
                provider_specific_header,
                custom_llm_provider,
                model=model,
                configured_headers=headers,
                base_model=deployment_base_model(
                    {"base_model": base_model if isinstance(base_model, str) else None}, model_info
                ),
            ),
        )

    @staticmethod
    def forwarded_authorization(
        provider_specific_header: object, custom_llm_provider: str | None, model: str | None, base_model: str | None
    ) -> str | None:
        headers: Final = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            _PROVIDER_HEADERS.validate_python(provider_specific_header),
            custom_llm_provider,
            model=model,
            base_model=base_model,
        )
        return next((value for key, value in headers.items() if key.lower() == "authorization"), None)

    @staticmethod
    def get_provider_specific_headers(
        provider_specific_header: ProviderSpecificHeader | Sequence[ProviderSpecificHeader] | None,
        custom_llm_provider: str | None,
        native_anthropic_messages: bool = False,
        model: str | None = None,
        configured_headers: Mapping[str, object] | None = None,
        base_model: str | None = None,
    ) -> Mapping[str, str]:
        """
        Get the provider specific headers for the given custom llm provider.

        Accepts either a single ProviderSpecificHeader or a sequence of them. Each entry
        carries its own comma-separated provider list, so headers that are safe for several
        providers and headers that are safe for exactly one can travel on the same request
        without sharing a scope. Entries whose provider list does not contain
        `custom_llm_provider` contribute nothing, and entries marked
        `native_anthropic_messages_only` contribute only to a provider's native Messages route.
        Entries marked `user_profile_id_models_only` contribute only when the model cost map says `model` or the
        deployment's `base_model` takes the Anthropic user profile id, entries marked
        `non_user_profile_id_models_only` only when neither does, and entries marked `yields_to_configured` never
        replace a header already in `configured_headers`, and
        entries listing `configured_credential_headers` contribute only when a value under one of those names, in any
        letter case, is sent that starts with `configured_credential_prefix`, is not one of the entry's
        `caller_credentials`, and is not a value another applicable entry contributes. The values sent are
        `configured_headers` after the other applicable entries replace the ones of the same name, and only the first
        is sent when it starts with that prefix, since the OAuth handling then drops the rest. Entries also marked `without_configured_credential` contribute only when no such value is
        sent.

        Returns:
            Dict: The provider specific headers for the given custom llm provider
        """
        if provider_specific_header is None or custom_llm_provider is None:
            return {}

        scoped_headers: Final = (
            (provider_specific_header,) if isinstance(provider_specific_header, dict) else provider_specific_header
        )

        configured: Final[Mapping[str, object]] = configured_headers or {}
        configured_names: Final = frozenset(name.lower() for name in configured)
        takes_user_profile_id: Final = any(map(_gates_on_user_profile_id_support, scoped_headers)) and (
            takes_anthropic_user_profile_id(model, custom_llm_provider, base_model)
        )
        applicable: Final = tuple(
            scoped_header
            for scoped_header in scoped_headers
            if _applies_to_call(scoped_header, custom_llm_provider, native_anthropic_messages, takes_user_profile_id)
        )
        replacing: Final = MappingProxyType(
            dict(
                chain.from_iterable(
                    _contributed_headers(scoped_header, configured_names).items()
                    for scoped_header in applicable
                    if not scoped_header.get("configured_credential_headers")
                )
            )
        )
        sent_values: Final = _text_values_by_name({**configured, **replacing})
        contributed_values: Final = frozenset(replacing.values())
        return dict(
            chain.from_iterable(
                _contributed_headers(scoped_header, configured_names).items()
                for scoped_header in applicable
                if _configures_a_listed_credential(scoped_header, sent_values, contributed_values)
            )
        )
