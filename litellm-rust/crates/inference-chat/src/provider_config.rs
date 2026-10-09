use litellm_llms::base_llm::chat::transformation::BaseConfig;

use crate::{Error, common_utils::chat_completions_provider};

pub(crate) fn resolve_provider_config<'a>(
    model: &'a str,
    custom_llm_provider: Option<&'a str>,
) -> Result<
    (
        litellm_inference::provider::ResolvedProvider<'a>,
        &'static dyn BaseConfig,
    ),
    Error,
> {
    let provider = litellm_inference::provider::resolve_llm_provider(
        model,
        custom_llm_provider,
        "chat completions",
    )?;
    let config = chat_completions_provider(provider.provider)
        .ok_or_else(|| Error::InvalidProvider(<&str>::from(provider.provider).to_string()))?
        .config();
    Ok((provider, config))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::resolve_provider_config;
    use crate::Error;

    #[rstest]
    #[case::anthropic(
        "anthropic/claude-sonnet-4-5",
        None,
        "claude-sonnet-4-5",
        "anthropic",
        ("anthropic-version", "2023-06-01")
    )]
    #[case::bedrock(
        "bedrock/us-east-1/anthropic.claude-v2",
        None,
        "us-east-1/anthropic.claude-v2",
        "bedrock",
        ("Content-Type", "application/json")
    )]
    fn resolves_the_provider_configuration(
        #[case] model: &str,
        #[case] custom_llm_provider: Option<&str>,
        #[case] expected_model: &str,
        #[case] expected_provider: &str,
        #[case] expected_default_header: (&str, &str),
    ) {
        let (provider, config) = resolve_provider_config(model, custom_llm_provider).unwrap();
        assert_eq!(provider.model, expected_model);
        assert_eq!(<&'static str>::from(provider.provider), expected_provider);
        assert_eq!(config.default_headers()[0], expected_default_header);
    }

    #[rstest]
    #[case::openai("openai/gpt-4o", None, "openai")]
    fn rejects_providers_without_a_chat_configuration(
        #[case] model: &str,
        #[case] custom_llm_provider: Option<&str>,
        #[case] expected: &str,
    ) {
        let Err(error) = resolve_provider_config(model, custom_llm_provider) else {
            panic!("provider should be rejected");
        };
        assert_eq!(error, Error::InvalidProvider(expected.to_string()));
    }
}
