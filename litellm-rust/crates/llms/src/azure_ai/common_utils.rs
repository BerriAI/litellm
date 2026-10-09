use crate::Error;
use litellm_core_utils::settings::resolve_non_empty;

pub const AZURE_API_KEY_ENV: &str = "AZURE_API_KEY";
pub const AZURE_API_BASE_ENV: &str = "AZURE_API_BASE";
pub const MISSING_AZURE_API_KEY: litellm_auth::Error = litellm_auth::Error::MissingApiKey {
    provider: "Azure",
    environment_variable: AZURE_API_KEY_ENV,
};

pub fn get_azure_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    resolve_non_empty(api_key, env_lookup, &[AZURE_API_KEY_ENV])
}

pub fn resolve_azure_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    get_azure_api_key(api_key, env_lookup).ok_or_else(|| Error::from(MISSING_AZURE_API_KEY))
}

pub fn resolve_azure_api_base(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    resolve_non_empty(api_base, env_lookup, &[AZURE_API_BASE_ENV])
        .ok_or_else(|| Error::from(litellm_auth::Error::MissingApiBase { provider: "Azure", guidance: "Set `api_base` or the AZURE_API_BASE environment variable. Expected format: https://<resource-name>.services.ai.azure.com/anthropic" }))
}
