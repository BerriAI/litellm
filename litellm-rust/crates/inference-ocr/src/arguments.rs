use super::provider_config::resolve_provider_config;

pub fn is_supported_request(model: &str, custom_llm_provider: Option<&str>) -> bool {
    resolve_provider_config(model, custom_llm_provider).is_ok()
}
