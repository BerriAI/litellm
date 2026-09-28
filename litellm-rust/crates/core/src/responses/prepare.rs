use litellm_host::hooks::RequestContext;
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
    let provider = call.custom_llm_provider.as_deref().unwrap_or("openai");
    if provider != "openai" {
        return Err(Error::Unsupported("native HTTP responses provider"));
    }
    let model = call.model.strip_prefix("openai/").unwrap_or(&call.model);
    if model.is_empty() || model.contains('/') {
        return Err(Error::InvalidProvider(call.model));
    }
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
        url: config.get_complete_url(call.api_base.as_deref(), &lookup),
        config,
        environment,
        body,
        context,
        timeout: call.timeout,
    })
}
