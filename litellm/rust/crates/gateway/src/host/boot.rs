use litellm_core_utils::get_llm_provider_logic::get_custom_llm_provider;
use litellm_router_core::{BuildAsk, Provided, RegistryHost};
use serde_json::Value;

/// What the gateway knows about deployments while it boots: the provider each one's model
/// resolves to. The config load already resolved their `os.environ/` references.
pub struct NativeBootHost;

impl RegistryHost for NativeBootHost {
    async fn provider(&self, ask: &BuildAsk<'_>) -> Result<Provided, litellm_router_core::Error> {
        let param = |name: &str| {
            ask.entry
                .get("litellm_params")
                .and_then(|params| params.get(name))
                .and_then(Value::as_str)
        };
        let custom = param("custom_llm_provider");
        let resolved = get_custom_llm_provider(param("model").unwrap_or_default(), custom)
            .map(|resolved| resolved.custom_llm_provider)
            .or(custom);
        Ok(Provided::Provider(resolved.map(str::to_owned)))
    }
}
