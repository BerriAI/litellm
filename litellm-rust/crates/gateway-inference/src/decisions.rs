use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use litellm_gateway_auth::AuthenticatedRequest;
use litellm_gateway_hooks::Route;
use litellm_inference_decisions::{DecisionsCall, parse_request};
use serde_json::{Map, Value};

use crate::{Deployment, Error, Gateway, JsonBody, hooks::Tracking, request};

pub(crate) async fn create(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    headers: HeaderMap,
    body: JsonBody,
) -> Result<Response, Error> {
    handle(&gateway, &identity, &headers, body).await
}

async fn handle(
    gateway: &Gateway,
    identity: &AuthenticatedRequest,
    headers: &HeaderMap,
    JsonBody { object, raw }: JsonBody,
) -> Result<Response, Error> {
    gateway.check_body(&object)?;
    let decisions = Arc::new(
        parse_request(&mut serde_json::Deserializer::from_slice(&raw))
            .map_err(litellm_inference_decisions::Error::from)?,
    );
    let group = request::resolve_group(gateway, &object)?;
    let name = request::model_name(&object)?.to_owned();
    let tracking = crate::hooks::admit(
        gateway,
        identity,
        Route::Decisions,
        headers,
        &raw,
        &object,
        &name,
    )
    .await?;
    let extra_headers = hook_headers(tracking.as_ref());
    let requested_drop = object
        .get("drop_params")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let routed = gateway
        .models
        .route_where(
            group,
            request::usable(identity),
            |attempt| {
                let name = name.as_str();
                let decisions = decisions.clone();
                let extra_headers = extra_headers.clone();
                if let Some(tracking) = &tracking {
                    tracking.attempt(name, &attempt);
                }
                async move {
                    let deployment = attempt.deployment;
                    crate::observe::record(name, deployment);
                    request::authorize(identity, name, deployment).await?;
                    let call = DecisionsCall {
                        request: decisions,
                        model: deployment.model.clone(),
                        api_key: deployment.api_key.clone(),
                        api_base: deployment.api_base.clone(),
                        custom_llm_provider: deployment.custom_llm_provider.clone(),
                        litellm_params: provider_params(deployment),
                        extra_headers,
                        timeout: deployment.timeout,
                        drop_params: requested_drop || drop_params(gateway, deployment),
                        supports_box_questions: deployment.supports_box_questions,
                        model_info: gateway.model_info(),
                    };
                    let response = gateway.decisions.execute(call, &(), None).await?;
                    Ok(Json(response).into_response())
                }
            },
            Error::failure,
        )
        .await
        .ok_or_else(|| Error::UnknownModel(name.clone()))
        .and_then(|routed| routed)
        .map(request::hold);
    match tracking {
        Some(tracking) => tracking.finish(routed).await,
        None => routed,
    }
}

fn hook_headers(tracking: Option<&Tracking>) -> Option<Map<String, Value>> {
    let headers = tracking?.edit_headers(Vec::new())?;
    let headers: Map<String, Value> = headers
        .into_iter()
        .map(|(name, value)| (name, Value::from(value)))
        .collect();
    (!headers.is_empty()).then_some(headers)
}

fn provider_params(deployment: &Deployment) -> Map<String, Value> {
    deployment
        .litellm_params
        .extra
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

fn drop_params(gateway: &Gateway, deployment: &Deployment) -> bool {
    gateway.drop_params()
        || deployment
            .litellm_params
            .drop_params
            .as_ref()
            .is_some_and(|flag| flag.is_set())
}
