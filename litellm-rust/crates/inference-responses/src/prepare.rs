use litellm_core_utils::get_llm_provider_logic::{
    CustomLlmProvider, LlmProviders, get_custom_llm_provider,
};
use litellm_host::interceptors::{ProviderIdentity, RequestContext};
use litellm_llms::{
    azure::responses::transformation::AZURE_RESPONSES_CONFIG,
    base_llm::responses::transformation::BaseResponsesApiConfig,
    bedrock_mantle::responses::transformation::{
        BedrockMantleResponsesApiConfig, split_region_prefix,
    },
    databricks::responses::transformation::DATABRICKS_RESPONSES_CONFIG,
    fireworks_ai::responses::transformation::FIREWORKS_AI_RESPONSES_CONFIG,
    hosted_vllm::responses::transformation::HOSTED_VLLM_RESPONSES_CONFIG,
    openai::responses::transformation::OpenAiResponsesApiConfig,
    openrouter::responses::transformation::OPENROUTER_RESPONSES_CONFIG,
    xai::responses::transformation::XAI_RESPONSES_CONFIG,
};
use litellm_router_types::LitellmParams;
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
    let ResponsesProvider { identity, config } = resolve_provider(
        &call.model,
        call.custom_llm_provider.as_deref(),
        call.api_base.as_deref(),
        &call.litellm_params,
    )?;
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
    pub config: Box<dyn BaseResponsesApiConfig + Send>,
}

/// Python's `get_llm_provider`: a bare model is OpenAI's, and with an explicit OpenRouter
/// provider a model without a vendor segment keeps the `openrouter/` prefix as its wire id.
pub(super) fn resolve_provider(
    model: &str,
    custom_llm_provider: Option<&str>,
    api_base: Option<&str>,
    params: &LitellmParams,
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
    let (config, wire_model): (Box<dyn BaseResponsesApiConfig + Send>, String) = match provider
        .parse::<LlmProviders>(
    ) {
        Ok(LlmProviders::Openai) if resolved.contains('/') => {
            return Err(Error::InvalidProvider(model.into()));
        }
        Ok(LlmProviders::Openai) => (Box::new(OpenAiResponsesApiConfig), resolved.into()),
        Ok(LlmProviders::Openrouter) if explicit.is_some() && !resolved.contains('/') => (
            Box::new(OPENROUTER_RESPONSES_CONFIG),
            format!("openrouter/{resolved}"),
        ),
        Ok(LlmProviders::Openrouter) => (Box::new(OPENROUTER_RESPONSES_CONFIG), resolved.into()),
        Ok(LlmProviders::Azure) => (Box::new(AZURE_RESPONSES_CONFIG), resolved.into()),
        Ok(LlmProviders::Xai) => (Box::new(XAI_RESPONSES_CONFIG), resolved.into()),
        Ok(LlmProviders::Databricks) => (Box::new(DATABRICKS_RESPONSES_CONFIG), resolved.into()),
        Ok(LlmProviders::FireworksAi) => (Box::new(FIREWORKS_AI_RESPONSES_CONFIG), resolved.into()),
        Ok(LlmProviders::HostedVllm) => (Box::new(HOSTED_VLLM_RESPONSES_CONFIG), resolved.into()),
        Ok(LlmProviders::BedrockMantle) => {
            let (region, wire) = split_region_prefix(resolved);
            (
                Box::new(BedrockMantleResponsesApiConfig::new(
                    api_base, region, params,
                )),
                wire.into(),
            )
        }
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
    #[case::databricks_prefix(
        "databricks/databricks-gpt-5-nano",
        None,
        "databricks",
        "databricks-gpt-5-nano"
    )]
    #[case::fireworks_prefix(
        "fireworks_ai/accounts/fireworks/models/qwen3",
        None,
        "fireworks_ai",
        "accounts/fireworks/models/qwen3"
    )]
    #[case::xai_prefix("xai/grok-4", None, "xai", "grok-4")]
    #[case::bedrock_mantle_region_prefix(
        "bedrock_mantle/us-west-2/openai.gpt-5.5",
        None,
        "bedrock_mantle",
        "openai.gpt-5.5"
    )]
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
        let resolved =
            resolve_provider(model, custom_llm_provider, None, &LitellmParams::default()).unwrap();
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
            resolve_provider(model, custom_llm_provider, None, &LitellmParams::default()),
            Err(Error::InvalidProvider(_) | Error::Unsupported(_))
        ));
    }
}
