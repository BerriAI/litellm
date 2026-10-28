use litellm_host::lifecycle::ExecutionEvent;
use litellm_host::observation::ObservationSender;
use std::time::Duration;

use futures_util::StreamExt;
use litellm_host::interceptors::{Interceptors, RawResponse, RequestContext, WireRequest};
use litellm_llms::base_llm::auth::{AuthScheme, Authenticated, resolve_auth};
use litellm_secrets::source::Secrets;

use super::{
    Error, ResponsesRoute,
    types::{ResponsesCall, ResponsesOutput, ResponsesProvider, ResponsesStreamHead},
};

pub(super) async fn execute(
    route: &ResponsesRoute,
    ResponsesProvider { identity, config }: ResponsesProvider,
    call: ResponsesCall,
    secrets: Secrets,
    cache_options: Option<litellm_cache_response::CachePolicy>,
    interceptors: &impl Interceptors<Error>,
    observers: Option<&ObservationSender>,
) -> Result<ResponsesOutput, Error> {
    let lookup = |name: &str| secrets.get(name);
    let environment = config.validate_environment(
        litellm_http::request::string_headers("responses", call.extra_headers)?,
        call.api_key.as_deref(),
        &lookup,
    )?;
    let url = config.get_complete_url(call.api_base.as_deref(), &lookup);
    let context = RequestContext {
        model: identity.model.clone(),
        custom_llm_provider: identity.provider.clone(),
        optional_params: serde_json::Value::Object(call.optional_params.clone()),
        secret_fields: Vec::new(),
        api_key: match &environment.auth {
            AuthScheme::Credential { secret, .. } => Some(secret.clone()),
            _ => None,
        },
    };
    let body = config.transform_responses_api_request(
        &identity.model,
        call.input,
        call.optional_params,
    )?;
    let authenticated = resolve_auth(&route.auth, environment, &|_| None).await?;
    let wire = interceptors
        .before_provider_request(
            WireRequest {
                url,
                headers: authenticated.headers,
                body,
            },
            context,
        )
        .await?;
    let cache = route
        .cache
        .clone()
        .filter(|_| authenticated.signer.is_none());
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
                &wire.url,
                &wire.body,
                Some(call.timeout.unwrap_or(Duration::from_secs(600))),
            )?;
            let response = litellm_inference::outbound::send(outbound, &route.http)
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
            let body = response.text().await.map_err(network)?;
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
            config
                .transform_response_api_response(value)
                .map(ResponsesOutput::Complete)
                .map_err(Error::from)
        },
    )
    .await
}

fn network(error: reqwest::Error) -> Error {
    litellm_http::transport::Error::Network(error.to_string()).into()
}
