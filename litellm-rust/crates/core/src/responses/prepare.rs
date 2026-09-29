use litellm_host::interceptors::RequestContext;
use litellm_llms::{
    base_llm::responses::transformation::BaseResponsesApiConfig,
    openai::responses::transformation::OpenAiResponsesApiConfig,
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
    let identity = resolve_provider(&call.model, call.custom_llm_provider.as_deref())?;
    let provider = identity.provider.as_str();
    let model = identity.model.as_str();
    let config: &'static dyn BaseResponsesApiConfig = &OpenAiResponsesApiConfig;
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
        endpoint: config.get_complete_url(call.api_base.as_deref(), &lookup)?,
        config,
        environment,
        body,
        context,
        timeout: call.timeout,
    })
}

pub(super) fn resolve_provider(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> Result<litellm_host::interceptors::ProviderIdentity, Error> {
    let provider = custom_llm_provider.unwrap_or("openai");
    if provider != "openai" {
        return Err(Error::Unsupported("native HTTP responses provider"));
    }
    let resolved = model.strip_prefix("openai/").unwrap_or(model);
    if resolved.is_empty() || resolved.contains('/') {
        return Err(Error::InvalidProvider(model.into()));
    }
    Ok(litellm_host::interceptors::ProviderIdentity {
        model: resolved.into(),
        provider: provider.into(),
    })
}
