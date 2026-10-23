use crate::Error;
use litellm_layered::{Layered, non_blank, read};

pub const AZURE_API_KEY_ENV: &str = "AZURE_API_KEY";
pub const AZURE_API_BASE_ENV: &str = "AZURE_API_BASE";

#[derive(Layered)]
#[expect(dead_code, reason = "only the derived partial is read")]
struct AzureAiKeys {
    #[layer(litellm_param = "api_key", env = "AZURE_API_KEY", read = read::text, secret, setting = "API Key")]
    api_key: String,
    #[layer(litellm_param = "api_base", env = "AZURE_API_BASE", read = read::text, setting = "API Base")]
    api_base: String,
}

impl AzureAiKeysPartial {
    fn env(env: &dyn Fn(&str) -> Option<String>) -> Self {
        Self::from_env(&env).unwrap_or_default()
    }
}

pub fn resolve_azure_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    non_blank(api_key)
        .or(AzureAiKeysPartial::env(env_lookup).api_key)
        .ok_or_else(|| {
            Error::from(litellm_auth::Error::MissingApiKey {
                provider: "Azure",
                environment_variable: AzureAiKeys::API_KEY.env[0],
            })
        })
}

pub fn resolve_azure_api_base(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    non_blank(api_base)
        .or(AzureAiKeysPartial::env(env_lookup).api_base)
        .ok_or_else(|| Error::from(litellm_auth::Error::MissingApiBase { provider: "Azure", guidance: "Set `api_base` or the AZURE_API_BASE environment variable. Expected format: https://<resource-name>.services.ai.azure.com/anthropic" }))
}
