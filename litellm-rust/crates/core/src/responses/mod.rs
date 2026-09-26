pub use crate::error::RouteError as Error;
pub mod types;
pub mod websocket;

use std::time::Duration;

use futures_util::{StreamExt, TryStreamExt};
use litellm_core_utils::get_llm_provider_logic::get_custom_llm_provider;
use litellm_http::{
    ClientVariant, HttpClientConfig,
    outbound::OutboundRequest,
    request::{has_bearer_auth, string_headers, truncate_error_body},
    transport::Error as TransportError,
};
use litellm_llms::{
    base_llm::auth::resolve_auth, openai::responses::transformation::OpenAiResponsesApiConfig,
};
use litellm_secrets::source::SecretSource;
use serde_json::Value;

use crate::resources::CoreResources;
use types::{ResponsesBody, ResponsesOutput, ResponsesRequest};

pub async fn responses(
    resources: &CoreResources,
    http: &HttpClientConfig,
    secrets: &dyn SecretSource,
    request: ResponsesRequest<'_>,
) -> Result<ResponsesOutput, Error> {
    let model = resolve_model(request.model, request.custom_llm_provider)?;
    let config = OpenAiResponsesApiConfig;
    let body = config.transform_request(model, request.body)?;
    let streaming = body.get("stream").and_then(Value::as_bool) == Some(true);
    let headers = string_headers("responses", request.extra_headers)?;
    let api_key = match request.api_key.filter(|key| !key.is_empty()) {
        Some(key) => Some(litellm_secrets::SecretValue::new(key)),
        None if has_bearer_auth(&headers) => None,
        None => secrets.get_secret_str("OPENAI_API_KEY").await?,
    };
    let api_base = match request.api_base {
        Some(base) => Some(litellm_secrets::SecretValue::new(base)),
        None => match secrets.get_secret_str("OPENAI_BASE_URL").await? {
            Some(base) => Some(base),
            None => secrets.get_secret_str("OPENAI_API_BASE").await?,
        },
    };
    let environment =
        config.validate_environment(headers, api_key.as_ref().map(|key| key.expose()))?;
    let authenticated = resolve_auth(&resources.auth, environment, &|_| None).await?;
    let outbound = OutboundRequest::json(
        config.get_complete_url(api_base.as_ref().map(|base| base.expose()))?,
        authenticated.headers,
        &body,
        Some(request.timeout.unwrap_or(Duration::from_secs(600))),
    )?;
    let client = resources.pool.client(http, ClientVariant::Provider)?;
    let response = outbound.send(&client).await.map_err(|error| {
        if error.is_connect() || error.is_builder() {
            Error::Transport(TransportError::Connect(error.to_string()))
        } else {
            network_error(error)
        }
    })?;
    if !response.status().is_success() {
        let status = response.status().as_u16();
        let text = response.text().await.map_err(network_error)?;
        return Err(TransportError::Http {
            status,
            body: truncate_error_body(&text),
        }
        .into());
    }
    let headers = response
        .headers()
        .iter()
        .filter(|(name, _)| {
            name.as_str() == "x-request-id" || name.as_str().starts_with("x-ratelimit-")
        })
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    if streaming {
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split(';').next());
        if !content_type.is_some_and(|value| value.trim().eq_ignore_ascii_case("text/event-stream"))
        {
            return Err(Error::InvalidResponse("expected text/event-stream".into()));
        }
        return Ok(ResponsesOutput {
            headers,
            body: ResponsesBody::Stream(response.bytes_stream().map_err(network_error).boxed()),
        });
    }
    let bytes = response.bytes().await.map_err(network_error)?;
    Ok(ResponsesOutput {
        headers,
        body: ResponsesBody::Response(config.transform_response(&bytes)?),
    })
}

fn resolve_model<'a>(model: &'a str, provider: Option<&'a str>) -> Result<&'a str, Error> {
    match get_custom_llm_provider(model, provider) {
        Some(resolved) if resolved.custom_llm_provider == "openai" => Ok(resolved.model),
        Some(_) => Err(Error::Unsupported("Responses provider other than OpenAI")),
        None if !model.contains('/') => Ok(model),
        None => Err(Error::InvalidProvider(model.into())),
    }
}

fn network_error(error: reqwest::Error) -> Error {
    TransportError::Network(error.to_string()).into()
}
