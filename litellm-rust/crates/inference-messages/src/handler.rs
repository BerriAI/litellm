use std::time::Duration;

use bytes::Bytes;
use futures_util::{StreamExt, TryStreamExt, stream::BoxStream};
use litellm_host::interceptors::{Interceptors, ProviderIdentity, RequestContext, WireRequest};
use litellm_http::transport::Error as TransportError;
use litellm_llms::base_llm::{
    auth::{Authenticated, resolve_auth},
    messages::{
        streaming::{ByteStream, StreamDecoder, encode_anthropic_sse},
        transformation::BaseMessagesConfig,
    },
};
use litellm_llms_types::formats::messages::MessagesResponse;
use litellm_tracing::ByteChunk;
use serde_json::Value;

use super::{
    Error, MessagesCallResponse, MessagesRoute, common_utils::truncate_error_body,
    prepare::ProviderMessagesRequest,
};
use crate::constants::MESSAGES_TIMEOUT_SECS;
use litellm_inference::{context::CallContext, outbound::outbound_request};

pub(super) struct ProviderCall {
    pub identity: ProviderIdentity,
    pub wire: WireRequest,
    provider: super::common_utils::MessagesProvider,
    signer: Option<litellm_auth_aws::SigV4Signer>,
    timeout: Option<Duration>,
    stream: bool,
}

impl ProviderCall {
    pub fn cacheable(&self) -> bool {
        self.signer.is_none()
    }
}

impl MessagesRoute {
    pub(super) async fn prepare_outbound(
        &self,
        request: ProviderMessagesRequest,
        context: &CallContext<'_, impl Interceptors<Error>>,
    ) -> Result<ProviderCall, Error> {
        let ProviderMessagesRequest {
            provider,
            url,
            body,
            environment,
            timeout,
            api_key,
        } = request;
        let request_context = RequestContext {
            model: body.model.clone(),
            custom_llm_provider: provider.as_str().to_string(),
            optional_params: serde_json::to_value(&body.params).map_err(serialize_failure)?,
            secret_fields: Vec::new(),
            api_key,
        };
        let authenticated =
            resolve_auth(&self.auth, environment, &|key| std::env::var(key).ok()).await?;
        let identity = ProviderIdentity {
            model: request_context.model.clone(),
            provider: request_context.custom_llm_provider.clone(),
        };
        let wire = context
            .interceptors
            .before_provider_request(
                WireRequest {
                    url,
                    headers: authenticated.headers,
                    body: provider
                        .config()
                        .wire_body(serde_json::to_value(&body).map_err(serialize_failure)?),
                },
                request_context,
            )
            .await?;
        let stream = match wire.body.get("stream") {
            None | Some(Value::Null) => false,
            Some(Value::Bool(stream)) => *stream,
            Some(value) => {
                return Err(Error::InvalidRequest(
                    litellm_llms::ErrorDetail::InvalidValue {
                        field: "stream",
                        expected: "a boolean",
                        actual: value.clone(),
                    },
                ));
            }
        };
        Ok(ProviderCall {
            identity,
            wire,
            provider,
            signer: authenticated.signer,
            timeout,
            stream,
        })
    }

    pub(super) async fn call_provider(
        &self,
        request: ProviderCall,
        context: &CallContext<'_, impl Interceptors<Error>>,
    ) -> Result<MessagesCallResponse, Error> {
        let ProviderCall {
            identity,
            wire,
            provider,
            signer,
            timeout,
            stream,
        } = request;
        let provider_name = provider.as_str();
        log_request_body(provider_name, stream, &wire.body);
        let response = send(
            &self.http,
            Authenticated {
                headers: wire.headers,
                signer,
            },
            &wire.url,
            &wire.body,
            timeout,
        )
        .await?;
        if !response.status().is_success() {
            return Err(provider_error(response).await);
        }
        let config = provider.config();
        if stream {
            return Ok(streaming_response(
                response,
                config.stream_decoder(),
                provider_name,
            ));
        }
        let headers = litellm_http::request::response_headers(response.headers());
        let text = response.text().await.map_err(network)?;
        log_response_body(&text);
        context.response_received(&text).await?;
        decode_response(config, &identity.model, &text).map(|message| {
            MessagesCallResponse::Complete(litellm_http::response::ProviderResponse {
                body: Box::new(message),
                headers,
            })
        })
    }
}

fn serialize_failure(err: serde_json::Error) -> Error {
    Error::InvalidRequest(litellm_llms::ErrorDetail::failed(
        "Anthropic messages request serialization",
        err,
    ))
}

fn network(error: reqwest::Error) -> Error {
    Error::Transport(TransportError::Network(error.to_string()))
}

async fn send(
    http: &litellm_http::Client,
    authenticated: Authenticated,
    url: &str,
    body: &Value,
    timeout: Option<Duration>,
) -> Result<reqwest::Response, Error> {
    let request = outbound_request(
        authenticated,
        url.to_string(),
        body,
        Some(timeout.unwrap_or(Duration::from_secs(MESSAGES_TIMEOUT_SECS))),
    )?;
    litellm_inference::outbound::send(request, http)
        .await
        .map_err(network)
}

async fn provider_error(response: reqwest::Response) -> Error {
    let status = response.status().as_u16();
    let headers = litellm_http::request::response_headers(response.headers());
    match response.text().await {
        Ok(text) => {
            log_error_body(status, &text);
            Error::Transport(TransportError::Http {
                status,
                body: truncate_error_body(&text),
                headers,
            })
        }
        Err(error) => network(error),
    }
}

fn decode_response(
    config: &dyn BaseMessagesConfig,
    model: &str,
    text: &str,
) -> Result<MessagesResponse, Error> {
    let response = serde_json::from_str(text).map_err(|err| {
        Error::InvalidResponse(litellm_llms::ErrorDetail::invalid(
            "messages response JSON",
            err,
        ))
    })?;
    config
        .transform_anthropic_messages_response(model, response)
        .map_err(Error::from)
}

fn streaming_response(
    response: reqwest::Response,
    decoder: Option<StreamDecoder>,
    provider: &'static str,
) -> MessagesCallResponse {
    let headers = response
        .headers()
        .iter()
        .filter_map(|(name, value)| Some((name.to_string(), value.to_str().ok()?.to_string())))
        .collect();
    let chunks = match decoder {
        None => futures_util::stream::try_unfold(response, move |mut response| async move {
            let chunk = response.chunk().await.map_err(network)?;
            Ok(chunk.map(|chunk| {
                log_chunk(provider, "provider_response", &chunk);
                (chunk, response)
            }))
        })
        .boxed(),
        Some(decode) => decoded_chunks(response, decode, provider),
    };
    MessagesCallResponse::Stream {
        head: super::route::MessagesStreamHead { headers },
        chunks,
    }
}

fn decoded_chunks(
    response: reqwest::Response,
    decode: StreamDecoder,
    provider: &'static str,
) -> BoxStream<'static, Result<Bytes, Error>> {
    let bytes: ByteStream = response
        .bytes_stream()
        .inspect_ok(move |chunk| log_chunk(provider, "provider_response", chunk))
        .map_err(std::io::Error::other)
        .boxed();
    futures_util::stream::try_unfold(decode(bytes), move |mut events| async move {
        let Some(event) = events.try_next().await? else {
            return Ok(None);
        };
        let chunk = encode_anthropic_sse(&event)?;
        log_chunk(provider, "client_response", &chunk);
        Ok(Some((chunk, events)))
    })
    .boxed()
}

fn log_request_body(provider: &str, stream: bool, body: &serde_json::Value) {
    tracing::debug!(provider, stream, body = %body, "provider request");
}

fn log_response_body(body: &str) {
    tracing::debug!(body, "provider response body");
}

fn log_error_body(status: u16, body: &str) {
    tracing::debug!(status, body, "provider error body");
}

fn log_chunk(provider: &str, stage: &str, data: &bytes::Bytes) {
    let chunk = ByteChunk::new(data);
    tracing::debug!(provider, stage, encoding = chunk.encoding(), chunk = %chunk, "stream chunk");
}

#[cfg(test)]
mod tests {
    use litellm_llms::base_llm::messages::streaming::anthropic_sse_event_stream;
    use rstest::rstest;
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::any};

    use super::*;

    #[rstest]
    #[case::event(
        "data: {\"type\":\"ping\"}\n\n",
        Some("event: ping\ndata: {\"type\":\"ping\"}\n\n")
    )]
    #[rstest::rstest]
    #[case::invalid_event("data: invalid\n\ndata: {\"type\":\"ping\"}\n\n", None)]
    #[tokio::test]
    async fn decoded_streams_encode_events_and_stop_at_the_first_error(
        #[case] body: &'static str,
        #[case] expected: Option<&str>,
    ) {
        let upstream = MockServer::start().await;
        Mock::given(any())
            .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
            .mount(&upstream)
            .await;
        let response = litellm_http::Client::plain_for_test()
            .get(upstream.uri())
            .send()
            .await
            .unwrap();
        let MessagesCallResponse::Stream { mut chunks, .. } =
            streaming_response(response, Some(anthropic_sse_event_stream), "test")
        else {
            panic!("a streaming response returns chunks");
        };

        let chunk = chunks.next().await.unwrap();
        match expected {
            Some(expected) => assert_eq!(chunk.unwrap().as_ref(), expected.as_bytes()),
            None => assert!(matches!(chunk, Err(Error::InvalidResponse(_))), "{chunk:?}"),
        }
        assert!(chunks.next().await.is_none());
    }
}
