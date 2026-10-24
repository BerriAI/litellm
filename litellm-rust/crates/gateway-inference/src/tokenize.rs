use std::{collections::BTreeMap, sync::Arc};

use axum::{Json, Router, extract::State, response::Response, routing::post};
use litellm_auth::{InputSource, SecretValue, Sourced};
use litellm_gateway_auth::AuthenticatedRequest;
use litellm_host::{interceptors::Interceptors, observation::ObservationSender};
use litellm_inference_tokenize::{TokenizeBackend, TokenizeRequest};
use litellm_llms_types::formats::tokenize::TokenizeRequest as TokenizePayload;
use litellm_token_counter::raw_text::LocalTokenizerConfig;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::{Deployment, Error, Gateway, JsonObject, request};

#[derive(Clone)]
struct TokenizeState<H> {
    gateway: Arc<Gateway>,
    interceptors: H,
    observers: Option<ObservationSender>,
}

#[derive(Default, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum ConfiguredBackend {
    #[default]
    Provider,
    Local {
        config: LocalTokenizerConfig,
    },
}

#[derive(Deserialize)]
struct DeploymentOptions {
    #[serde(default)]
    backend: ConfiguredBackend,
    extra_headers: Option<Map<String, Value>>,
    #[serde(flatten)]
    params: Map<String, Value>,
}

pub(crate) fn router<H>(
    gateway: Arc<Gateway>,
    interceptors: H,
    observers: Option<ObservationSender>,
) -> Router
where
    H: Interceptors<litellm_inference::Error> + Clone + 'static,
{
    Router::new()
        .route("/tokenize", post(create::<H>))
        .route("/v1/tokenize", post(create::<H>))
        .with_state(TokenizeState {
            gateway,
            interceptors,
            observers,
        })
}

async fn create<H>(
    State(state): State<TokenizeState<H>>,
    identity: AuthenticatedRequest,
    JsonObject(body): JsonObject,
) -> Response
where
    H: Interceptors<litellm_inference::Error> + Clone + 'static,
{
    match handle(state, identity, body).await {
        Ok(response) => response,
        Err(error) => error_response(error),
    }
}

async fn handle<H>(
    state: TokenizeState<H>,
    identity: AuthenticatedRequest,
    body: Map<String, Value>,
) -> Result<Response, Error>
where
    H: Interceptors<litellm_inference::Error> + Clone + 'static,
{
    let deployment = request::resolve_deployment(&state.gateway, &body)?;
    request::authorize_model(&identity, deployment, &body).await?;
    let payload: TokenizePayload = serde_json::from_value(Value::Object(body))
        .map_err(|error| Error::InvalidBody(error.to_string()))?;
    let call = project(payload, deployment)?;
    litellm_host_http::serve_unary(
        state
            .gateway
            .tokenize
            .clone()
            .machine(call, state.observers.clone()),
        (),
        state.interceptors,
        litellm_host_http::Unary::new(Json),
        state.observers,
    )
    .await
    .map_err(Error::from)
}

fn project(payload: TokenizePayload, deployment: &Deployment) -> Result<TokenizeRequest, Error> {
    let options: DeploymentOptions = serde_json::from_value(Value::Object(
        deployment
            .litellm_params
            .extra
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect(),
    ))
    .map_err(|_| Error::InvalidBody("invalid tokenization deployment configuration".into()))?;
    let backend = match options.backend {
        ConfiguredBackend::Provider => TokenizeBackend::Provider,
        ConfiguredBackend::Local { config } => TokenizeBackend::Local {
            config: Sourced::new(Arc::new(config), InputSource::Deployment),
        },
    };
    let params = options
        .params
        .into_iter()
        .chain(
            deployment
                .litellm_params
                .api_version
                .as_ref()
                .map(|version| ("api_version".into(), Value::String(version.clone()))),
        )
        .collect();
    Ok(TokenizeRequest {
        model: deployment.model.clone(),
        backend,
        custom_llm_provider: deployment.custom_llm_provider.clone(),
        api_key: deployment
            .api_key
            .clone()
            .map(|key| Sourced::new(SecretValue::new(key), InputSource::Deployment)),
        api_base: deployment
            .api_base
            .clone()
            .map(|base| Sourced::new(base, InputSource::Deployment)),
        extra_headers: deployment_values(options.extra_headers.unwrap_or_default()),
        deployment_params: deployment_values(params),
        timeout: deployment.timeout,
        ..payload.into()
    })
}

fn deployment_values(values: Map<String, Value>) -> BTreeMap<String, Sourced<Value>> {
    values
        .into_iter()
        .map(|(name, value)| (name, Sourced::new(value, InputSource::Deployment)))
        .collect()
}

fn error_response(error: Error) -> Response {
    let headers = match &error {
        Error::Route(litellm_inference::Error::Transport(
            litellm_http::transport::Error::Http { headers, .. },
        )) => Some(litellm_http::transport::safe_headers(headers)),
        _ => None,
    };
    let mut response = error.openai_response();
    if let Some(headers) = headers {
        response.headers_mut().extend(*headers);
    }
    response
}
