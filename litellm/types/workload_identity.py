"""litellm_params keys that configure workload identity federation.

The kwargs funnel (``litellm_core_utils.get_litellm_params``) and the request-body ban list
(``types.utils.all_litellm_params``) both derive from these sets, so they live in a module with
no litellm imports that either side can reach without a cycle. Every key here rides the funnel
into ``litellm_params`` and is banned from request bodies, which also covers
``anthropic_disable_workload_identity_federation``: the proxy sets it when a client redirects
``api_base`` so a federated deployment stops minting for a base the caller chose, and a caller
must not be able to set it in either direction.
"""

from typing import Final

ANTHROPIC_WIF_KWARGS_KEYS: Final = frozenset(
    {
        "anthropic_federation_rule_id",
        "anthropic_organization_id",
        "anthropic_service_account_id",
        "anthropic_federation_workspace_id",
        "anthropic_identity_token_file",
        "anthropic_identity_token",
        "anthropic_identity_source",
        "anthropic_issuer_url",
        "anthropic_issuer_subject",
        "anthropic_issuer_audience",
        "anthropic_issuer_ttl_seconds",
        "anthropic_issuer_signing_key_ref",
        "anthropic_keycloak_token_url",
        "anthropic_keycloak_client_id",
        "anthropic_keycloak_auth_method",
        "anthropic_keycloak_client_secret_ref",
        "anthropic_keycloak_scope",
        "anthropic_disable_workload_identity_federation",
    }
)

OPENAI_WIF_KWARGS_KEYS: Final = frozenset(
    {
        "openai_identity_provider_id",
        "openai_service_account_id",
        "openai_identity_token_file",
    }
)

WIF_SECRET_BEARING_KEYS: Final = frozenset(
    {
        "anthropic_identity_token",
        "anthropic_identity_token_file",
        "anthropic_issuer_signing_key_ref",
        "anthropic_keycloak_client_secret_ref",
        "openai_identity_token_file",
    }
)
"""The federation keys whose value is a credential, or the path or reference that reaches one.

The rest of the sets above name a rule, an organization, a workspace, or a URL: an operator has
to be able to read those back to tell what a deployment federates as. These carry the secret
itself, so no surface displays them to anyone. Splitting the sensitivity out here keeps the
callers that redact them from having to know which provider a field belongs to.
"""
