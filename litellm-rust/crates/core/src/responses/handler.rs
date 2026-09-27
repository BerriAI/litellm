use std::time::Duration;

use futures_util::StreamExt;
use litellm_host::{
    event::{MachineEvent, RawResponse, WireRequest},
    hooks::RouteHooks,
};
use litellm_llms::base_llm::auth::{Authenticated, resolve_auth};

use super::{
    Error,
    types::{ProviderResponsesRequest, ResponsesOutput, ResponsesStreamHead},
};

pub(super) async fn execute(
    http: &litellm_http::Client,
    auth: &litellm_auth::AuthServices,
    request: ProviderResponsesRequest,
    hooks: &impl RouteHooks<Error>,
) -> Result<ResponsesOutput, Error> {
    let authenticated = resolve_auth(auth, request.environment, &|_| None).await?;
    let wire = hooks
        .before_provider_request(
            WireRequest {
                url: request.url,
                headers: authenticated.headers,
                body: request.body,
            },
            request.context,
        )
        .await?;
    let stream = match wire.body.get("stream") {
        None => false,
        Some(serde_json::Value::Bool(value)) => *value,
        Some(_) => return Err(Error::InvalidRequest("stream must be a boolean".into())),
    };
    let outbound = crate::outbound::outbound_request(
        Authenticated {
            headers: wire.headers,
            signer: authenticated.signer,
        },
        wire.url,
        &wire.body,
        Some(request.timeout.unwrap_or(Duration::from_secs(600))),
    )?;
    let response = crate::outbound::send(outbound, http)
        .await
        .map_err(network)?;
    let status = response.status().as_u16();
    if !response.status().is_success() {
        let body = response.text().await.map_err(network)?;
        return Err(litellm_http::transport::Error::Http {
            status,
            body: litellm_http::request::truncate_error_body(&body),
        }
        .into());
    }
    if stream {
        let headers = response
            .headers()
            .iter()
            .filter_map(|(name, value)| Some((name.to_string(), value.to_str().ok()?.to_owned())))
            .collect();
        let chunks = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(network))
            .boxed();
        return Ok(ResponsesOutput::Stream {
            head: ResponsesStreamHead { headers },
            chunks,
        });
    }
    let body = response.text().await.map_err(network)?;
    hooks
        .on_event(MachineEvent::ResponseReceived {
            raw: RawResponse { body: body.clone() },
        })
        .await
        .map_err(Error::post_call)?;
    let value =
        serde_json::from_str(&body).map_err(|error| Error::InvalidResponse(error.to_string()))?;
    request
        .config
        .transform_response_api_response(value)
        .map(ResponsesOutput::Complete)
        .map_err(Error::from)
}

fn network(error: reqwest::Error) -> Error {
    litellm_http::transport::Error::Network(error.to_string()).into()
}
