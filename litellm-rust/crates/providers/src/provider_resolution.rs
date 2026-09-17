#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CustomLlmProvider<'a> {
    pub model: &'a str,
    pub custom_llm_provider: &'a str,
}

pub fn get_custom_llm_provider<'a>(
    model: &'a str,
    custom_llm_provider: Option<&'a str>,
) -> Option<CustomLlmProvider<'a>> {
    if let Some(custom_llm_provider) = custom_llm_provider.filter(|provider| !provider.is_empty()) {
        return Some(CustomLlmProvider {
            model: strip_custom_llm_provider_prefix(model, custom_llm_provider),
            custom_llm_provider,
        });
    }

    let (custom_llm_provider, model) = model.split_once('/')?;
    if custom_llm_provider.is_empty() || model.is_empty() {
        return None;
    }
    Some(CustomLlmProvider {
        model,
        custom_llm_provider,
    })
}

fn strip_custom_llm_provider_prefix<'a>(model: &'a str, custom_llm_provider: &str) -> &'a str {
    model
        .strip_prefix(custom_llm_provider)
        .and_then(|model| model.strip_prefix('/'))
        .unwrap_or(model)
}
