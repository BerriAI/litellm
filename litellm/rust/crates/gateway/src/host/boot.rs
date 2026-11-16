use litellm_config::ENV_REFERENCE_PREFIX;
use litellm_core_utils::{
    get_llm_provider_logic::get_custom_llm_provider, serde_compat::parse_str_bool, settings::Lookup,
};
use litellm_router_core::{BuildAsk, Provided, RegistryHost, SecretRef};
use serde_json::Value;

/// What the gateway knows about deployments while it boots: the environment their secret
/// references name, and the provider each one's model resolves to.
pub struct NativeBootHost<'a, L> {
    pub env: &'a L,
}

impl<L: Lookup + Sync> RegistryHost for NativeBootHost<'_, L> {
    async fn secrets(
        &self,
        _index: usize,
        refs: &[SecretRef],
    ) -> Result<Vec<Value>, litellm_router_core::Error> {
        Ok(refs
            .iter()
            .map(|secret| environment_value(&secret.name, self.env))
            .collect())
    }

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

/// `get_secret` on the environment: a `true`/`false` word reads as a bool, an unset name as
/// `None`.
pub fn environment_value(reference: &str, env: &impl Lookup) -> Value {
    let name = reference
        .strip_prefix(ENV_REFERENCE_PREFIX)
        .unwrap_or(reference);
    env.get(name).map_or(Value::Null, |value| {
        parse_str_bool(&value).map_or(Value::String(value), Value::Bool)
    })
}
