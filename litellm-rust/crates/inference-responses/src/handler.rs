use std::time::Duration;

use futures_util::StreamExt;
use litellm_host::{
    interceptors::{Interceptors, RawResponse, WireRequest},
    lifecycle::ExecutionEvent,
    observation::ObservationSender,
};
use litellm_inference::call::{self, Failure};
use litellm_llms::base_llm::auth::{Authenticated, resolve_auth};

use super::{
    Error,
    types::{ProviderResponsesRequest, ResponsesOutput, ResponsesStreamHead},
};

pub(super) async fn execute(
    http: &litellm_http::Client,
    auth: &litellm_auth::AuthServices,
    request: ProviderResponsesRequest,
    cache: Option<litellm_cache_response::ScopedCache>,
    cache_options: Option<litellm_cache_response::CachePolicy>,
    interceptors: &impl Interceptors<Error>,
    observers: Option<&ObservationSender>,
) -> Result<ResponsesOutput, Failure<Error>> {
    let identity = litellm_host::interceptors::ProviderIdentity {
        model: request.context.model.clone(),
        provider: request.context.custom_llm_provider.clone(),
    };
    let (authenticated, wire, stream) = call::prepare(async {
        let authenticated = resolve_auth(auth, request.environment, &|_| None).await?;
        let wire = interceptors
            .before_provider_request(
                WireRequest {
                    url: request.url,
                    headers: authenticated.headers.clone(),
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
        Ok((authenticated, wire, stream))
    })
    .await?;
    let cache = cache.filter(|_| authenticated.signer.is_none());
    let cache_request = litellm_inference::caching::CacheRequest::from_wire(
        identity,
        cache.as_ref().map(|_| &wire),
    );
    litellm_inference::caching::execute_streaming::<super::route::Responses, _, _>(
        cache_request,
        cache.as_ref().map(|cache| cache.service.clone()),
        cache.as_ref().map(|cache| cache.options(cache_options)),
        interceptors,
        observers,
        || async move {
            let outbound = call::prepare(async {
                litellm_inference::outbound::outbound_request(
                    Authenticated {
                        headers: wire.headers,
                        signer: authenticated.signer,
                    },
                    wire.url,
                    &wire.body,
                    Some(request.timeout.unwrap_or(Duration::from_secs(600))),
                )
                .map_err(Error::from)
            })
            .await?;
            let response = call::send(http, outbound).await?;
            if stream {
                let headers = response
                    .headers()
                    .iter()
                    .filter_map(|(name, value)| {
                        Some((name.to_string(), value.to_str().ok()?.to_owned()))
                    })
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
            let body = call::receive(async { response.text().await.map_err(network) }).await?;
            let raw = RawResponse { body: body.clone() };
            if let Some(observers) = observers {
                observers.emit(litellm_host::lifecycle::CallEvent::Execution(
                    ExecutionEvent::ProviderResponseReceived { raw: raw.clone() },
                ));
            }
            call::post_call(interceptors.after_provider_response(raw)).await?;
            call::receive(async {
                let value = serde_json::from_str(&body)
                    .map_err(|error| Error::InvalidResponse(error.to_string().into()))?;
                request
                    .config
                    .transform_response_api_response(value)
                    .map(ResponsesOutput::Complete)
                    .map_err(Error::from)
            })
            .await
        },
    )
    .await
}

fn network(error: reqwest::Error) -> Error {
    litellm_http::transport::Error::from(error).into()
}
