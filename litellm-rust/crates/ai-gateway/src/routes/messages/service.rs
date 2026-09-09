use std::sync::Arc;

use litellm_core::Error;
use litellm_core::constants::ANTHROPIC_MESSAGES_PROVIDER;
use litellm_core::integrations::custom_logger::CustomLogger;
use litellm_core::lifecycle::{CallLifecycleContext, StreamingCall};
use litellm_core::messages::lifecycle::Options;
use litellm_core::messages::types::{AnthropicMessagesRequest, MessagesRequest};
use litellm_core::router::Router;
use serde_json::{Map, Value};

use crate::state::GatewayMessagesServices;

pub(crate) enum MessagesResponse {
    Json(Value),
    Stream(Box<StreamingCall>),
}

#[tracing::instrument(
    name = "messages_gateway_service",
    target = "litellm::function_trace",
    level = "trace",
    skip_all
)]
pub async fn run(
    router: &Arc<Router>,
    client: &litellm_core::runtime::LiteLlm<GatewayMessagesServices>,
    loggers: Arc<Vec<Arc<dyn CustomLogger>>>,
    body: AnthropicMessagesRequest,
    extra_headers: Option<Map<String, Value>>,
) -> Result<MessagesResponse, Error> {
    let model = body.model.trim();
    if model.is_empty() {
        return Err(Error::InvalidRequest(
            "messages body requires a model".to_string(),
        ));
    }
    let deployment = router
        .get_available_deployment(model)
        .ok_or_else(|| Error::Routing(format!("no deployment available for model '{model}'")))?;
    let provider_model = deployment.litellm_params.model.as_str();
    let upstream_model = provider_model
        .split_once('/')
        .map_or(provider_model, |(_, model)| model);
    let custom_llm_provider = if provider_model.contains('/') {
        None
    } else {
        Some(ANTHROPIC_MESSAGES_PROVIDER)
    };
    let stream = body.stream == Some(true);
    let body = serde_json::to_value(AnthropicMessagesRequest {
        model: upstream_model.to_string(),
        ..body
    })
    .map_err(|error| {
        Error::InvalidRequest(format!(
            "failed to serialize Anthropic messages request: {error}"
        ))
    })?;

    let request = MessagesRequest {
        model: provider_model.to_string(),
        body,
        api_key: deployment.litellm_params.api_key.clone(),
        api_base: deployment.litellm_params.api_base.clone(),
        custom_llm_provider: custom_llm_provider.map(str::to_string),
        extra_headers,
        timeout: None,
    };
    let context = CallLifecycleContext::new(
        "messages",
        provider_model,
        custom_llm_provider.unwrap_or(ANTHROPIC_MESSAGES_PROVIDER),
        format!(
            "messages-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ),
    );
    if stream {
        return client
            .messages_stream_with(request, Options::default(), context, loggers)
            .await
            .map(Box::new)
            .map(MessagesResponse::Stream);
    }

    let response = client.messages_with(request, context, loggers).await?;
    serde_json::to_value(response)
        .map(MessagesResponse::Json)
        .map_err(|err| {
            Error::InvalidResponse(format!("failed to serialize messages response: {err}"))
        })
}
