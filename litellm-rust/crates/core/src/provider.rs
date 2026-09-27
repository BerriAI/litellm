pub use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use litellm_core_utils::get_llm_provider_logic::{CustomLlmProvider, get_custom_llm_provider};

use crate::error::RouteError as Error;

#[derive(Debug)]
pub(crate) struct ResolvedProvider<'a> {
    pub(crate) model: &'a str,
    pub(crate) provider: LlmProviders,
}

pub(crate) fn resolve_llm_provider<'a>(
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
        .ok_or_else(|| {
            Error::InvalidProvider(format!(
                "unable to resolve custom_llm_provider for {route} request"
            ))
        })?;
    let provider = custom_llm_provider
        .parse()
        .map_err(|_| Error::InvalidProvider(custom_llm_provider.to_string()))?;
    Ok(ResolvedProvider { model, provider })
}
