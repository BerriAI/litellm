use std::sync::Arc;

use axum::{Json, body::Bytes, extract::State, response::Response};
use litellm_core::responses::{route::Responses, types::ResponsesCall};
use litellm_gateway_auth::AuthenticatedRequest;
use litellm_host_http::Sse;
use serde_json::json;

use crate::{Error, Gateway, JsonObject, request};

pub(crate) async fn create(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    JsonObject(body): JsonObject,
) -> Result<Response, Error> {
    let deployment = request::resolve_deployment(&gateway, &body)?;
    request::authorize_model(&identity, deployment, &body).await?;
    let accounting = match &gateway.responses_accounting {
        Some(service) => Some(
            service
                .begin(crate::accounting::AdmissionRequest {
                    caller: identity.caller().clone(),
                    public_model: body
                        .get("model")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    deployment_model: deployment.model.clone(),
                    request: body.clone(),
                })
                .await?,
        ),
        None => None,
    };
    let call = ResponsesCall {
        model: deployment.model.clone(),
        input: body.get("input").cloned().unwrap_or_default(),
        optional_params: body
            .into_iter()
            .filter(|(name, _)| !matches!(name.as_str(), "model" | "input"))
            .collect(),
        api_key: deployment.api_key.clone(),
        api_base: deployment.api_base.clone(),
        custom_llm_provider: deployment.custom_llm_provider.clone(),
        extra_headers: None,
        timeout: deployment.timeout,
    };
    let machine = gateway.responses.clone().machine(call, None);
    let stream = Sse::<Responses, _, _>::new(Json, |error| {
        let error = Error::from(error);
        Bytes::from(format!(
            "event: error\ndata: {}\n\n",
            json!({"type": "error", "code": error.status().as_u16().to_string(), "message": error.to_string(), "param": null})
        ))
    });
    match accounting {
        Some(hooks) => {
            Ok(litellm_host_http::serve_with_hooks(machine, (), hooks, stream, None).await?)
        }
        None => Ok(litellm_host_http::serve(machine, (), (), stream, None).await?),
    }
}
