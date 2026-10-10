use litellm_host::lifecycle::ExecutionEvent;
use litellm_host::observation::ObservationSender;
use std::time::Duration;

use futures_util::StreamExt;
use litellm_host::interceptors::{Interceptors, RawResponse, WireRequest};
use litellm_llms::base_llm::auth::{Authenticated, resolve_auth};

use super::{
    Error,
    types::{ProviderResponsesRequest, ResponsesOutput},
};

pub(super) async fn execute(
    http: &litellm_http::Client,
    auth: &litellm_auth::AuthServices,
    request: ProviderResponsesRequest,
    cache: Option<litellm_cache_response::ScopedCache>,
    cache_options: Option<litellm_cache_response::CachePolicy>,
    interceptors: &impl Interceptors<Error>,
    observers: Option<&ObservationSender>,
) -> Result<ResponsesOutput, Error> {
    let authenticated = resolve_auth(auth, request.environment, &|_| None).await?;
    let identity = litellm_host::interceptors::ProviderIdentity {
        model: request.context.model.clone(),
        provider: request.context.custom_llm_provider.clone(),
    };
    let wire = interceptors
        .before_provider_request(
            WireRequest {
                url: request.url,
                headers: authenticated.headers,
                body: request.body,
            },
            request.context,
        )
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
            let stream = match wire.body.get("stream") {
                None => false,
                Some(serde_json::Value::Bool(value)) => *value,
                Some(_) => return Err(Error::InvalidRequest("stream must be a boolean".into())),
            };
            let outbound = litellm_inference::outbound::outbound_request(
                Authenticated {
                    headers: wire.headers,
                    signer: authenticated.signer,
                },
                wire.url,
                &wire.body,
                Some(request.timeout.unwrap_or(Duration::from_secs(600))),
            )?;
            let response = litellm_inference::outbound::send(outbound, http).await?;
            if stream {
                let head = litellm_http::transport::parts(&response);
                let chunks = response
                    .bytes_stream()
                    .map(|chunk| chunk.map_err(network))
                    .boxed();
                return Ok(ResponsesOutput::Stream { head, chunks });
            }
            let (parts, body) = litellm_inference::outbound::read(response)
                .await?
                .into_parts();
            let body = litellm_inference::outbound::text(&body);
            let raw = RawResponse { body: body.clone() };
            if let Some(observers) = observers {
                observers.emit(litellm_host::lifecycle::CallEvent::Execution(
                    ExecutionEvent::ProviderResponseReceived { raw: raw.clone() },
                ));
            }
            interceptors
                .after_provider_response(raw)
                .await
                .map_err(Error::post_call)?;
            let value = serde_json::from_str(&body)
                .map_err(|error| Error::InvalidResponse(error.to_string().into()))?;
            request
                .config
                .transform_response_api_response(value)
                .map(|body| ResponsesOutput::Complete(http::Response::from_parts(parts, body)))
                .map_err(Error::from)
        },
    )
    .await
}

fn network(error: reqwest::Error) -> Error {
    litellm_http::transport::Error::from(error).into()
}
