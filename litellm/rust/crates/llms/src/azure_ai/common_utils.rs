use crate::Error;
use litellm_core_utils::settings::resolve_non_empty;

pub const AZURE_OPENAI_API_KEY_ENV: &str = "AZURE_OPENAI_API_KEY";
pub const AZURE_API_KEY_ENV: &str = "AZURE_API_KEY";
pub const AZURE_API_BASE_ENV: &str = "AZURE_API_BASE";

/// The params Python's `get_azure_ad_token` reads from a call to mint an Entra ID token.
pub const ENTRA_ID_PARAMS: &[&str] = &[
    "azure_ad_token",
    "tenant_id",
    "client_id",
    "client_secret",
    "azure_scope",
];

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

pub fn resolve_azure_api_base(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    resolve_non_empty(api_base, env_lookup, &[AZURE_API_BASE_ENV])
        .ok_or_else(|| Error::from(litellm_auth::Error::MissingApiBase { provider: "Azure", guidance: "Set `api_base` or the AZURE_API_BASE environment variable. Expected format: https://<resource-name>.services.ai.azure.com/anthropic" }))
}
