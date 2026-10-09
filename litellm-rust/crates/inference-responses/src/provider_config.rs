use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use litellm_inference::provider::ResolvedProvider;
use litellm_llms::{
    base_llm::responses::transformation::BaseResponsesApiConfig,
    openai::responses::transformation::OpenAiResponsesApiConfig,
};

use crate::Error;

pub(crate) fn resolve_provider_config<'a>(
    model: &'a str,
    custom_llm_provider: Option<&'a str>,
) -> Result<(ResolvedProvider<'a>, &'static dyn BaseResponsesApiConfig), Error> {
    if custom_llm_provider.is_some_and(|provider| provider != "openai") {
        return Err(Error::Unsupported("native HTTP responses provider"));
    }
    let model_name = model.strip_prefix("openai/").unwrap_or(model);
    if model_name.is_empty() || model_name.contains('/') {
        return Err(Error::InvalidProvider(model.into()));
    }
    Ok((
        ResolvedProvider {
            model: model_name,
            provider: LlmProviders::Openai,
        },
        &OpenAiResponsesApiConfig,
    ))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[derive(Debug)]
    enum ExpectedResolution {
        OpenAi(&'static str),
        InvalidProvider(&'static str),
        Unsupported,
    }

    #[rstest]
    #[case::unprefixed("test", None, ExpectedResolution::OpenAi("test"))]
    #[case::openai_prefix("openai/gpt-4o", None, ExpectedResolution::OpenAi("gpt-4o"))]
    #[case::explicit_openai("gpt-4o", Some("openai"), ExpectedResolution::OpenAi("gpt-4o"))]
    #[case::other_prefix(
        "anthropic/x",
        None,
        ExpectedResolution::InvalidProvider("anthropic/x")
    )]
    #[case::other_explicit_provider("x", Some("anthropic"), ExpectedResolution::Unsupported)]
    #[case::nested_openai_model(
        "openai/a/b",
        None,
        ExpectedResolution::InvalidProvider("openai/a/b")
    )]
    #[case::empty_openai_model("openai/", None, ExpectedResolution::InvalidProvider("openai/"))]
    fn provider_resolution_preserves_the_previous_responses_contract(
        #[case] model: &str,
        #[case] custom_llm_provider: Option<&str>,
        #[case] expected: ExpectedResolution,
    ) {
        match (
            resolve_provider_config(model, custom_llm_provider),
            expected,
        ) {
            (Ok((provider, _)), ExpectedResolution::OpenAi(expected_model)) => {
                assert_eq!(provider.provider, LlmProviders::Openai);
                assert_eq!(provider.model, expected_model);
            }
            (
                Err(Error::InvalidProvider(actual)),
                ExpectedResolution::InvalidProvider(expected),
            ) => {
                assert_eq!(actual, expected);
            }
            (Err(Error::Unsupported(actual)), ExpectedResolution::Unsupported) => {
                assert_eq!(actual, "native HTTP responses provider");
            }
            (_, expected) => panic!("unexpected resolution, expected {expected:?}"),
        }
    }
}
