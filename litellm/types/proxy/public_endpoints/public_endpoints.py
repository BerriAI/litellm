from collections.abc import Mapping
from typing import Any, Final, Literal

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self


class PublicModelHubInfo(BaseModel):
    docs_title: str
    custom_docs_description: str | None
    litellm_version: str
    # Supports both old format (Dict[str, str]) and new format (Dict[str, Dict[str, Any]])
    # New format: { "displayName": { "url": "...", "index": 0 } }
    # Old format: { "displayName": "url" } (for backward compatibility)
    useful_links: dict[str, str | dict[str, Any]] | None


class ProviderCredentialField(BaseModel):
    key: str
    label: str
    placeholder: str | None = None
    tooltip: str | None = None
    required: bool = False
    field_type: Literal["text", "password", "select", "upload", "textarea"] = "text"
    options: list[str] | None = None
    default_value: str | None = None


class ProviderCredentialVariant(BaseModel):
    """One selectable auth shape for a provider, e.g. 'api_key' or 'workload identity
    federation, Keycloak'. ``field_keys`` names entries in the parent
    ``ProviderCredentialVariants.field_definitions`` to mount when this variant is active;
    ``fixed_values`` are litellm_params values the variant implies (e.g. a discriminator like
    ``anthropic_identity_source: keycloak``) and are submitted without a form field for them.
    ``optional_field_keys`` relaxes a globally-required field for this variant alone, for a value
    only obtainable after the credential exists (the federation rule id an operator can only read
    off the Anthropic Console once the generated JWKS is registered)."""

    id: str
    label: str
    field_keys: tuple[str, ...]
    optional_field_keys: tuple[str, ...] = ()
    fixed_values: Mapping[str, str] = Field(default_factory=dict)


def _validate_variant(variant: "ProviderCredentialVariant", defined_keys: frozenset[str]) -> None:
    """Each variant may only reference declared fields, may only relax fields it actually mounts, and a
    fixed value may not also be a field the form would mount, or the form and the payload would
    disagree about who owns that key."""
    unresolved: Final = tuple(key for key in variant.field_keys if key not in defined_keys)
    if unresolved:
        raise ValueError(f"variant {variant.id!r} references undefined field_keys: {unresolved}")
    overlap: Final = sorted(frozenset(variant.field_keys) & frozenset(variant.fixed_values))
    if overlap:
        raise ValueError(f"variant {variant.id!r} has fixed_values overlapping field_keys: {overlap}")
    unmounted: Final = tuple(key for key in variant.optional_field_keys if key not in frozenset(variant.field_keys))
    if unmounted:
        raise ValueError(f"variant {variant.id!r} relaxes optional_field_keys it does not mount: {unmounted}")


class ProviderCredentialVariants(BaseModel):
    """Declares selectable auth variants for a provider whose credential shape branches (e.g.
    API key vs. one of several workload-identity-federation identity sources), so the field
    list itself depends on a choice the form makes, not just which fields are shown.

    ``field_definitions`` is the full pool of fields any variant may reference by key;
    a UI mounts only the active variant's ``field_keys``, keeping every other field both
    unmounted and unsubmitted. ``default_variant`` seeds the selector on a fresh form.
    """

    selector_label: str
    default_variant: str
    field_definitions: tuple[ProviderCredentialField, ...]
    variants: tuple[ProviderCredentialVariant, ...]

    @model_validator(mode="after")
    def _validate_structure(self) -> Self:
        variant_ids: Final = tuple(variant.id for variant in self.variants)
        if len(variant_ids) != len(frozenset(variant_ids)):
            raise ValueError(f"credential_variants.variants ids must be unique, got {variant_ids}")
        if self.default_variant not in variant_ids:
            raise ValueError(
                f"credential_variants.default_variant {self.default_variant!r} is not one of {variant_ids}"
            )
        defined_keys: Final = frozenset(field.key for field in self.field_definitions)
        for variant in self.variants:
            _validate_variant(variant, defined_keys)
        return self


class ProviderCreateInfo(BaseModel):
    provider: str
    provider_display_name: str
    litellm_provider: str
    credential_fields: list[ProviderCredentialField]
    default_model_placeholder: str | None = None
    credential_variants: ProviderCredentialVariants | None = None


class AgentCredentialField(BaseModel):
    key: str
    label: str
    placeholder: str | None = None
    tooltip: str | None = None
    required: bool = False
    field_type: Literal["text", "password", "select", "upload", "textarea"] = "text"
    options: list[str] | None = None
    default_value: str | None = None
    include_in_litellm_params: bool | None = None
    validation_pattern: str | None = None
    validation_message: str | None = None


class AgentCreateInfo(BaseModel):
    agent_type: str
    agent_type_display_name: str
    description: str | None = None
    logo_url: str | None = None
    credential_fields: list[AgentCredentialField]
    litellm_params_template: dict[str, str] | None = None
    model_template: str | None = None


class EndpointProvider(BaseModel):
    slug: str
    display_name: str


class SupportedEndpoint(BaseModel):
    key: str
    label: str
    endpoint: str
    providers: list[EndpointProvider]


class SupportedEndpointsResponse(BaseModel):
    endpoints: list[SupportedEndpoint]


class ComplexityScorerDefaults(BaseModel):
    """The complexity router's shipped heuristic scorer defaults.

    The dashboard prefills its Advanced scoring controls from these rather than keeping its own copy, so
    a recalibration of the defaults cannot leave the form reporting numbers the router no longer uses.
    """

    tier_boundaries: Mapping[str, float]
    token_thresholds: Mapping[str, int]
    dimension_weights: Mapping[str, float]
