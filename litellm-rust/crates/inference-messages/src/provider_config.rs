use litellm_inference::provider::{ResolvedProvider, resolve_llm_provider};

use crate::{
    Error,
    common_utils::{MessagesProvider, messages_provider},
};

pub(crate) fn resolve_provider_config<'a>(
    model: &'a str,
    custom_llm_provider: Option<&'a str>,
) -> Result<(ResolvedProvider<'a>, MessagesProvider), Error> {
    let provider = resolve_llm_provider(model, custom_llm_provider, "messages")?;
    let config = messages_provider(provider.provider, provider.model)
        .ok_or_else(|| Error::InvalidProvider(<&str>::from(provider.provider).to_string()))?;
    Ok((provider, config))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::resolve_provider_config;

    #[rstest]
    #[case::anthropic_prefix("anthropic/claude-test", None, "claude-test", "anthropic")]
    #[case::azure_ai_prefix("azure_ai/claude-test", None, "claude-test", "azure_ai")]
    #[case::explicit_provider("claude-test", Some("anthropic"), "claude-test", "anthropic")]
    #[case::vertex_claude("vertex_ai/claude-sonnet-4-5", None, "claude-sonnet-4-5", "vertex_ai")]
    fn resolver_returns_the_messages_provider_configuration(
        #[case] model: &str,
        #[case] custom_llm_provider: Option<&str>,
        #[case] resolved_model: &str,
        #[case] expected_provider: &str,
    ) {
        let (provider, config) =
            resolve_provider_config(model, custom_llm_provider).expect("provider resolves");
        assert_eq!(
            (
                provider.model,
                <&'static str>::from(provider.provider),
                config.as_str()
            ),
            (resolved_model, expected_provider, expected_provider)
        );
    }

    #[rstest]
    #[case::openai(
        "gpt-5",
        None,
        "unable to resolve custom_llm_provider for messages request"
    )]
    #[case::vertex_gemini("vertex_ai/gemini-2.5-pro", None, "vertex_ai")]
    fn resolver_rejects_providers_without_a_messages_configuration(
        #[case] model: &str,
        #[case] custom_llm_provider: Option<&str>,
        #[case] rejected_provider: &str,
    ) {
        let Err(error) = resolve_provider_config(model, custom_llm_provider) else {
            panic!("provider without a Messages config must be rejected");
        };
        assert_eq!(
            error,
            crate::Error::InvalidProvider(rejected_provider.to_string())
        );
    }
}
