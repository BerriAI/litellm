use crate::{Error, ErrorDetail};

pub const AZURE_OPENAI_API_KEY_ENV: &str = "AZURE_OPENAI_API_KEY";
pub const AZURE_API_KEY_ENV: &str = "AZURE_API_KEY";
pub const AZURE_API_BASE_ENV: &str = "AZURE_API_BASE";

/// The params Python's `get_azure_ad_token` reads from a call to mint an Entra ID token.
pub const ENTRA_ID_PARAMS: &[&str] = &[
    "azure_ad_token",
    "tenant_id",
    "client_id",
    "client_secret",
    "azure_username",
    "azure_password",
    "azure_scope",
];

/// Set by a host whose call names an `azure_ad_token_provider`: [`CALLABLE_TOKEN_PROVIDER`]
/// when Python would call it, any other string when Python only counts it as present.
pub const CALLER_TOKEN_PROVIDER_PARAM: &str = "azure_ad_token_provider";
pub const CALLABLE_TOKEN_PROVIDER: &str = "callable";
pub const TOKEN_REFRESH_PARAM: &str = "enable_azure_ad_token_refresh";

pub const AZURE_AD_TOKEN_ENV: &str = "AZURE_AD_TOKEN";
pub const AZURE_TENANT_ID_ENV: &str = "AZURE_TENANT_ID";
pub const AZURE_CLIENT_ID_ENV: &str = "AZURE_CLIENT_ID";
pub const AZURE_CLIENT_SECRET_ENV: &str = "AZURE_CLIENT_SECRET";
pub const AZURE_USERNAME_ENV: &str = "AZURE_USERNAME";
pub const AZURE_PASSWORD_ENV: &str = "AZURE_PASSWORD";
pub const AZURE_SCOPE_ENV: &str = "AZURE_SCOPE";

/// Python's `BaseAzureLLM._base_validate_azure_environment` key chain after the call's key and
/// `litellm.api_key` / `litellm.azure_key`: the first value Python finds truthy, so an empty one
/// falls through and a blank one is kept as sent.
pub fn resolve_azure_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    api_key
        .filter(|key| !key.is_empty())
        .map(str::to_string)
        .or_else(|| {
            [AZURE_OPENAI_API_KEY_ENV, AZURE_API_KEY_ENV]
                .iter()
                .find_map(|name| env_lookup(name).filter(|key| !key.is_empty()))
        })
}

/// Python's `api_base or get_secret_str("AZURE_API_BASE")`: the call's base unless empty, else the
/// environment's as it is.
pub fn resolve_azure_api_base(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    api_base
        .filter(|base| !base.is_empty())
        .map(str::to_string)
        .or_else(|| env_lookup(AZURE_API_BASE_ENV))
        .ok_or(Error::InvalidRequest(ErrorDetail::MissingAzureApiBase))
}
