use litellm_core_utils::settings::resolve_non_empty;

pub const MINIMAX_API_KEY_ENV: &str = "MINIMAX_API_KEY";
pub const MINIMAX_API_BASE_ENV: &str = "MINIMAX_API_BASE";
pub const MINIMAX_ANTHROPIC_API_BASE: &str = "https://api.minimax.io/anthropic";

pub fn resolve_minimax_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    resolve_non_empty(api_key, env_lookup, &[MINIMAX_API_KEY_ENV])
}

pub fn resolve_minimax_anthropic_api_base(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    resolve_non_empty(api_base, env_lookup, &[MINIMAX_API_BASE_ENV])
        .unwrap_or_else(|| MINIMAX_ANTHROPIC_API_BASE.to_string())
}
