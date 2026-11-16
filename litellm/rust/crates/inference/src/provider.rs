use litellm_core_utils::get_llm_provider_logic::{
    CustomLlmProvider, LlmProviders, get_custom_llm_provider,
};

use litellm_core_utils::decline::Decline;

use crate::error::Error;

const NO_NATIVE_CONFIG: Decline = Decline::new(
    "llms.provider",
    "a provider with no native config for this route",
);

/// The decline for a provider the Rust path has no config for on `route`.
pub fn no_native_config(route: &'static str, provider: &str) -> Decline {
    NO_NATIVE_CONFIG.with_detail(format!("{route} on provider `{provider}`"))
}

#[derive(Debug)]
pub struct ResolvedProvider<'a> {
    pub model: &'a str,
    pub provider: LlmProviders,
}

pub fn resolve_llm_provider<'a>(
    model: &'a str,
    custom_llm_provider: Option<&'a str>,
    route: &'static str,
) -> Result<ResolvedProvider<'a>, Error> {
    let CustomLlmProvider {
        model,
        custom_llm_provider,
    } = get_custom_llm_provider(model, custom_llm_provider)
        .or_else(|| {
            custom_llm_provider.map(|provider| CustomLlmProvider {
                model,
                custom_llm_provider: provider,
            })
        })
        .ok_or(Error::UnresolvedProvider { route })?;
    let provider = custom_llm_provider
        .parse()
        .map_err(|_| Error::InvalidProvider(custom_llm_provider.to_string()))?;
    Ok(ResolvedProvider { model, provider })
}
