use std::sync::Arc;

use axum::{Json, extract::State};
use litellm_auth::SecretValue;
use litellm_gateway_auth::AuthenticatedRequest;
use litellm_inference_tokenize::TokenizeRequest;
use litellm_llms_types::formats::tokenize::{TokenizeRequest as TokenizePayload, TokenizeResponse};
use serde_json::{Map, Value};

use crate::{Error, Gateway, JsonObject, request};

pub(crate) async fn create(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    JsonObject(body): JsonObject,
) -> Result<Json<TokenizeResponse>, Error> {
    let deployment = request::resolve_deployment(&gateway, &body)?;
    request::authorize_model(&identity, deployment, &body).await?;
    let payload: TokenizePayload = serde_json::from_value(Value::Object(body.clone()))
        .map_err(|error| Error::InvalidBody(error.to_string()))?;
    let extra_headers: Option<Map<String, Value>> = serde_json::from_value(
        deployment
            .litellm_params
            .extra
            .get("extra_headers")
            .cloned()
            .unwrap_or(Value::Null),
    )
    .map_err(|_| Error::InvalidBody("deployment extra_headers must be an object or null".into()))?;
    let deployment_params = deployment
        .litellm_params
        .extra
        .iter()
        .filter(|(key, _)| key.as_str() != "extra_headers")
        .map(|(key, value)| (key.clone(), value.clone()))
        .chain(
            deployment
                .litellm_params
                .api_version
                .as_ref()
                .map(|version| ("api_version".into(), Value::String(version.clone()))),
        )
        .collect();
    let optional_params = body
        .into_iter()
        .filter(|(key, _)| !matches!(key.as_str(), "model" | "prompt"))
        .collect();
    gateway
        .tokenize
        .execute(
            TokenizeRequest {
                model: deployment.model.clone(),
                prompt: payload.prompt,
                custom_llm_provider: deployment.custom_llm_provider.clone(),
                api_key: deployment.api_key.clone().map(SecretValue::new),
                api_base: deployment.api_base.clone(),
                extra_headers: extra_headers.unwrap_or_default(),
                deployment_params,
                optional_params,
                timeout: deployment.timeout,
            },
            &(),
            None,
        )
        .await
        .map(Json)
        .map_err(Error::from)
}
