use std::sync::Arc;

use litellm_core::router::Router;
use litellm_core::{CoreError, CoreResult};
use serde_json::{Map, Value};

use crate::constants::ANTHROPIC_MESSAGES_PROVIDER;
use crate::messages::{execute_messages, MessagesRequest};

pub(crate) enum MessagesResponse {
    Json(Value),
    Stream(reqwest::Response),
}

pub async fn run(
    router: &Arc<Router>,
    body: Value,
    extra_headers: Option<Map<String, Value>>,
) -> CoreResult<MessagesResponse> {
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .ok_or_else(|| CoreError::InvalidRequest("messages body requires a model".to_string()))?;
    let deployment = router.get_available_deployment(model).ok_or_else(|| {
        CoreError::Routing(format!("no deployment available for model '{model}'"))
    })?;
    let provider_model = deployment.litellm_params.model.as_str();
    let upstream_model = provider_model
        .split_once('/')
        .map_or(provider_model, |(_, model)| model);
    let custom_llm_provider = if provider_model.contains('/') {
        None
    } else {
        Some(ANTHROPIC_MESSAGES_PROVIDER)
    };
    let mut body = body;
    body.as_object_mut()
        .ok_or_else(|| CoreError::InvalidRequest("messages body must be an object".to_string()))?
        .insert(
            "model".to_string(),
            Value::String(upstream_model.to_string()),
        );

    let request = MessagesRequest {
        model: provider_model,
        body,
        api_key: deployment.litellm_params.api_key.as_deref(),
        api_base: deployment.litellm_params.api_base.as_deref(),
        custom_llm_provider,
        extra_headers,
        timeout: None,
    };
    let stream = request.body.get("stream").and_then(Value::as_bool) == Some(true);
    execute_messages(request, stream)
        .await
        .map(|response| match response {
            crate::messages::MessagesResponse::Json(body) => MessagesResponse::Json(body),
            crate::messages::MessagesResponse::Stream(upstream) => {
                MessagesResponse::Stream(upstream)
            }
        })
}
