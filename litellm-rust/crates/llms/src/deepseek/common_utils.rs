use litellm_core_utils::settings::resolve_non_empty;

pub const DEEPSEEK_API_KEY_ENV: &str = "DEEPSEEK_API_KEY";
pub const DEEPSEEK_ANTHROPIC_API_BASE_ENV: &str = "DEEPSEEK_ANTHROPIC_API_BASE";
pub const DEEPSEEK_API_BASE_ENV: &str = "DEEPSEEK_API_BASE";
pub const DEEPSEEK_ANTHROPIC_DEFAULT_API_BASE: &str = "https://api.deepseek.com/anthropic";

pub fn resolve_deepseek_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    resolve_non_empty(api_key, env_lookup, &[DEEPSEEK_API_KEY_ENV])
}

pub fn resolve_deepseek_anthropic_api_base(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    resolve_non_empty(
        api_base,
        env_lookup,
        &[DEEPSEEK_ANTHROPIC_API_BASE_ENV, DEEPSEEK_API_BASE_ENV],
    )
    .unwrap_or_else(|| DEEPSEEK_ANTHROPIC_DEFAULT_API_BASE.to_string())
}
