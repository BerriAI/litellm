//! Harness-only adapters. Never mounted as production routes.

use std::sync::Arc;

use litellm_core::Error;
use litellm_core::router::{Deployment, LiteLLMParams, Router};
use serde::Serialize;
use serde_json::Value;

use crate::runtime::messages::{MessagesResponse, run};

#[derive(Debug, Serialize)]
pub struct GatewayResponse {
    pub status: u16,
    pub body: Value,
}

#[tracing::instrument(
    name = "messages_gateway_route",
    target = "litellm::function_trace",
    level = "trace",
    skip_all
)]
pub async fn messages_request(
    model_alias: String,
    provider_model: String,
    api_base: String,
    body: Value,
) -> Result<GatewayResponse, Error> {
    let router = Arc::new(Router::new(vec![Deployment {
        model_name: model_alias,
        litellm_params: LiteLLMParams {
            model: provider_model,
            api_key: Some("trace-provider-key".to_string()),
            api_base: Some(api_base),
        },
    }]));
    match run(&router, body, None).await {
        Ok(MessagesResponse::Json(body)) => Ok(GatewayResponse { status: 200, body }),
        Ok(MessagesResponse::Stream(_)) => Err(Error::InvalidResponse(
            "gateway returned a streaming trace response".to_string(),
        )),
        Err(error) => Ok(error_response(error)),
    }
}

fn error_response(error: Error) -> GatewayResponse {
    let (status, message) = match error {
        Error::InvalidRequest(message) => (400, message),
        Error::InvalidProvider(_) | Error::Routing(_) => (
            404,
            "no messages deployment is configured for this model".to_string(),
        ),
        Error::Auth(_) => (502, "messages provider authentication failed".to_string()),
        Error::Http { .. }
        | Error::Network(_)
        | Error::Connect(_)
        | Error::InvalidResponse(_)
        | Error::InvalidType { .. }
        | Error::MissingField(_) => (502, "messages provider request failed".to_string()),
        Error::Unsupported(reason) => (400, format!("messages request is not supported: {reason}")),
    };
    GatewayResponse {
        status,
        body: serde_json::json!({"error": {"message": message}}),
    }
}
