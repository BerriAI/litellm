use crate::Error;

const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
const ANTHROPIC_API_BASE_ENV: &str = "ANTHROPIC_API_BASE";
const DEFAULT_ANTHROPIC_API_BASE: &str = "https://api.anthropic.com";
const MESSAGES_PATH_SUFFIX: &str = "/v1/messages";

pub fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

pub fn resolve_anthropic_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    non_empty(api_key)
        .map(str::to_string)
        .or_else(|| env_lookup(ANTHROPIC_API_KEY_ENV).filter(|value| !value.trim().is_empty()))
        .ok_or_else(|| {
            Error::Auth(
                "Missing Anthropic API Key - Set `api_key` or the ANTHROPIC_API_KEY \
                 environment variable"
                    .to_string(),
            )
        })
}

pub fn complete_anthropic_url(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    let api_base = non_empty(api_base)
        .map(str::to_string)
        .or_else(|| env_lookup(ANTHROPIC_API_BASE_ENV).filter(|value| !value.trim().is_empty()))
        .unwrap_or_else(|| DEFAULT_ANTHROPIC_API_BASE.to_string());

    let api_base = api_base.trim_end_matches('/');
    if api_base.ends_with(MESSAGES_PATH_SUFFIX) {
        return api_base.to_string();
    }
    format!("{api_base}{MESSAGES_PATH_SUFFIX}")
}
