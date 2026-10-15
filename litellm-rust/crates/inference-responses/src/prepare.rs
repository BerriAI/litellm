use litellm_core_utils::get_llm_provider_logic::{
    CustomLlmProvider, LlmProviders, get_custom_llm_provider,
};
use litellm_host::interceptors::{ProviderIdentity, RequestContext};
use litellm_llms::{
    azure::responses::transformation::AZURE_RESPONSES_CONFIG,
    base_llm::responses::transformation::BaseResponsesApiConfig,
    fireworks_ai::responses::transformation::FIREWORKS_AI_RESPONSES_CONFIG,
    hosted_vllm::responses::transformation::HOSTED_VLLM_RESPONSES_CONFIG,
    openai::responses::transformation::OpenAiResponsesApiConfig,
    openrouter::responses::transformation::OPENROUTER_RESPONSES_CONFIG,
    xai::responses::transformation::XAI_RESPONSES_CONFIG,
};
use litellm_secrets::source::SecretSource;
use serde_json::Value;

use super::{
    Error,
    types::{ProviderResponsesRequest, ResponsesCall},
};

#[tracing::instrument(name = "litellm.prepare", level = "debug", skip_all)]
pub(super) async fn prepare(
    call: ResponsesCall,
    secrets: &dyn SecretSource,
) -> Result<ProviderResponsesRequest, Error> {
    let ResponsesProvider { identity, config } =
        resolve_provider(&call.model, call.custom_llm_provider.as_deref())?;
    let provider = identity.provider.as_str();
    let model = identity.model.as_str();
    let snapshot = secrets
        .resolve(config.secret_names(call.api_key.as_deref(), call.api_base.as_deref()))
        .await?;
    let lookup = |name: &str| snapshot.get(name);
    let environment = config.validate_environment(
        litellm_http::request::string_headers("responses", call.extra_headers)?,
        call.api_key.as_deref(),
        &lookup,
    )?;
    let context = RequestContext {
        model: model.into(),
        custom_llm_provider: provider.into(),
        optional_params: Value::Object(call.optional_params.clone()),
        secret_fields: Vec::new(),
        api_key: match &environment.auth {
            litellm_llms::base_llm::auth::AuthScheme::Credential { secret, .. } => {
                Some(secret.clone())
            }
            _ => None,
        },
    };
    let body = config.transform_responses_api_request(model, call.input, call.optional_params)?;
    Ok(ProviderResponsesRequest {
        url: config.get_complete_url(
            call.api_base.as_deref(),
            call.api_version.as_deref(),
            &lookup,
        )?,
        config,
        environment,
        body,
        context,
        timeout: call.timeout,
    })
}

pub(super) struct ResponsesProvider {
    pub identity: ProviderIdentity,
    pub config: &'static dyn BaseResponsesApiConfig,
}

/// Python's `get_llm_provider`: a bare model is OpenAI's, and with an explicit OpenRouter
/// provider a model without a vendor segment keeps the `openrouter/` prefix as its wire id.
pub(super) fn resolve_provider(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> Result<ResponsesProvider, Error> {
    let explicit = custom_llm_provider.filter(|provider| !provider.is_empty());
    let (provider, resolved) = match get_custom_llm_provider(model, explicit) {
        Some(CustomLlmProvider {
            model,
            custom_llm_provider,
        }) => (custom_llm_provider, model),
        None => ("openai", model),
    };
    if resolved.is_empty() {
        return Err(Error::InvalidProvider(model.into()));
    }
    let (config, wire_model): (&'static dyn BaseResponsesApiConfig, String) =
        match provider.parse::<LlmProviders>() {
            Ok(LlmProviders::Openai) if resolved.contains('/') => {
                return Err(Error::InvalidProvider(model.into()));
            }
            Ok(LlmProviders::Openai) => (&OpenAiResponsesApiConfig, resolved.into()),
            Ok(LlmProviders::Openrouter) if explicit.is_some() && !resolved.contains('/') => (
                &OPENROUTER_RESPONSES_CONFIG,
                format!("openrouter/{resolved}"),
            ),
            Ok(LlmProviders::Openrouter) => (&OPENROUTER_RESPONSES_CONFIG, resolved.into()),
            Ok(LlmProviders::Azure) => (&AZURE_RESPONSES_CONFIG, resolved.into()),
            Ok(LlmProviders::Xai) => (&XAI_RESPONSES_CONFIG, resolved.into()),
            Ok(LlmProviders::FireworksAi) => (&FIREWORKS_AI_RESPONSES_CONFIG, resolved.into()),
            Ok(LlmProviders::HostedVllm) => (&HOSTED_VLLM_RESPONSES_CONFIG, resolved.into()),
            _ => return Err(Error::Unsupported("native HTTP responses provider")),
        };
    Ok(ResponsesProvider {
        identity: ProviderIdentity {
            model: wire_model,
            provider: provider.into(),
        },
        config,
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// Pinned against Python's `get_llm_provider` on 2026-10-11.
    #[rstest]
    #[case::bare("gpt-5", None, "openai", "gpt-5")]
    #[case::openai_prefix("openai/gpt-5", None, "openai", "gpt-5")]
    #[case::openai_explicit("gpt-5", Some("openai"), "openai", "gpt-5")]
    #[case::openrouter_prefix(
        "openrouter/anthropic/claude",
        None,
        "openrouter",
        "anthropic/claude"
    )]
    #[case::openrouter_explicit(
        "openrouter/anthropic/claude",
        Some("openrouter"),
        "openrouter",
        "anthropic/claude"
    )]
    #[case::openrouter_explicit_unprefixed(
        "anthropic/claude",
        Some("openrouter"),
        "openrouter",
        "anthropic/claude"
    )]
    #[case::openrouter_native_id(
        "openrouter/auto",
        Some("openrouter"),
        "openrouter",
        "openrouter/auto"
    )]
    #[case::openrouter_native_id_implicit("openrouter/auto", None, "openrouter", "auto")]
    #[case::azure_prefix("azure/my-deployment", None, "azure", "my-deployment")]
    #[case::azure_explicit("my-deployment", Some("azure"), "azure", "my-deployment")]
    #[case::fireworks_prefix(
        "fireworks_ai/accounts/fireworks/models/qwen3",
        None,
        "fireworks_ai",
        "accounts/fireworks/models/qwen3"
    )]
    #[case::xai_prefix("xai/grok-4", None, "xai", "grok-4")]
    #[case::hosted_vllm_keeps_the_vendor(
        "hosted_vllm/meta-llama/Llama-3.1-8B",
        None,
        "hosted_vllm",
        "meta-llama/Llama-3.1-8B"
    )]
    fn providers_and_wire_models_match_python(
        #[case] model: &str,
        #[case] custom_llm_provider: Option<&str>,
        #[case] provider: &str,
        #[case] wire_model: &str,
    ) {
        let resolved = resolve_provider(model, custom_llm_provider).unwrap();
        assert_eq!(resolved.identity.provider, provider);
        assert_eq!(resolved.identity.model, wire_model);
    }

    #[rstest]
    #[case::other_prefix("other/model", None)]
    #[case::other_explicit("model", Some("other"))]
    #[case::openai_with_vendor_segment("openai/other/model", None)]
    #[case::empty_model("openai/", None)]
    #[case::conflicting_prefix("openrouter/anthropic/claude", Some("openai"))]
    fn everything_else_is_rejected(#[case] model: &str, #[case] custom_llm_provider: Option<&str>) {
        assert!(matches!(
            resolve_provider(model, custom_llm_provider),
            Err(Error::InvalidProvider(_) | Error::Unsupported(_))
        ));
    }
}
