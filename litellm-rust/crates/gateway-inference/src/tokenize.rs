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
    let optional_params: Map<String, Value> = deployment
        .litellm_params
        .extra
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .chain(
            deployment
                .litellm_params
                .api_version
                .as_ref()
                .map(|version| ("api_version".into(), Value::String(version.clone()))),
        )
        .chain(
            body.into_iter()
                .filter(|(key, _)| !matches!(key.as_str(), "model" | "prompt")),
        )
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
                optional_params,
                timeout: deployment.timeout,
                ..Default::default()
            },
            &(),
            None,
        )
        .await
        .map(Json)
        .map_err(Error::from)
}
