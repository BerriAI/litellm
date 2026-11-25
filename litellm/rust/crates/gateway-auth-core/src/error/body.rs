/// Why the proxy does not accept a request body. The text is what the caller reads after
/// `Authentication Error, `.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Rejection {
    #[error("Rejected Request: model_list is not allowed in the request body.")]
    ModelList,
    #[error(
        "Rejected Request: {param} is not allowed in request body. Clientside passthrough requires explicit admin \
opt-in via either `general_settings.allow_client_side_credentials = true` (proxy-wide) or \
`configurable_clientside_auth_params` on the deployment in your proxy config.yaml. Relevant Issue: \
https://huntr.com/bounties/4001e1a2-7b7a-4776-a3ae-e6692ec3d997{allowlist}"
    )]
    Override { param: String, allowlist: Allowlist },
    #[error(
        "Rejected Request: {param} is a server-owned {kind} parameter and cannot be set in a request body. A proxy \
admin configures it on the deployment or on a stored credential."
    )]
    ServerOwned {
        param: String,
        kind: ServerOwnedKind,
    },
    #[error(
        "Rejected Request: api_key holds an oidc/ secret reference, which only a deployment or credential a proxy \
admin configures can carry."
    )]
    ApiKeyOidcReference,
    #[error(
        "Rejected Request: {param} holds a secret reference, which only a deployment or credential a proxy admin \
configures can carry."
    )]
    SecretReference { param: String },
    #[error(
        "Rejected Request: litellm_credential_name='{named}' names a credential configured for workload identity \
federation or OAuth token exchange, or holding an oidc/ api_key secret reference ({field}), which a request body \
cannot choose. A proxy admin attaches it to a deployment."
    )]
    FederatedCredential { named: String, field: String },
    #[error(
        "Rejected Request: litellm_credential_name='{named}' names a credential holding an oidc/ api_key secret \
reference, which only a deployment a proxy admin configures can attach."
    )]
    CredentialApiKeyOidcReference { named: String },
    #[error(
        "Rejected Request: the request names more than {limit} fallback models, too many to check against \
configurable_clientside_auth_params."
    )]
    TooManyFallbackModels { limit: usize },
    #[error("Rejected Request: fallback nesting exceeds the allowed validation depth.")]
    FallbackNestingTooDeep,
    #[error(
        "Rejected Request: URL-valued fallback destination '{value}' is not allowed. Configure custom endpoints with \
api_base instead, or add the destination host to `provider_url_destination_allowed_hosts` in litellm_settings."
    )]
    UrlFallbackDestination { value: String },
    #[error(
        "Rejected Request: {param} nests deeper than the proxy reads a request body, so it cannot be checked for \
secret references."
    )]
    TooDeep { param: String },
}

/// The settings that admit single values of an override, named in its rejection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::Display)]
pub enum Allowlist {
    #[strum(serialize = "")]
    None,
    #[strum(
        serialize = " To allow specific values only, list them in `general_settings.allowed_client_side_s3_buckets` or in the team's `metadata.allowed_s3_buckets`."
    )]
    Buckets,
    #[strum(
        serialize = " To allow specific values only, list them in `general_settings.allowed_client_side_s3_endpoint_urls`."
    )]
    EndpointUrls,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::Display)]
pub enum ServerOwnedKind {
    #[strum(serialize = "GitHub Copilot authentication")]
    GitHubCopilot,
    #[strum(serialize = "workload identity federation or OAuth token exchange")]
    Federation,
}
