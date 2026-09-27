use std::time::Duration;

use bytes::Bytes;
use futures_util::{StreamExt, TryStreamExt, stream::BoxStream};
use litellm_auth::AuthServices;
use litellm_host::{
    event::{MachineEvent, RawResponse, RequestContext, WireRequest},
    hooks::RouteHooks,
};
use litellm_http::transport::Error as TransportError;
use litellm_llms::base_llm::{
    anthropic_messages::{
        streaming::{ByteStream, StreamDecoder, encode_anthropic_sse},
        transformation::BaseAnthropicMessagesConfig,
    },
    auth::{Authenticated, resolve_auth},
};
use litellm_tracing::ByteChunk;
use litellm_types::llms::anthropic_messages::anthropic_response::AnthropicMessagesResponse;
use serde_json::Value;

use super::{
    Error, MessagesResponse, common_utils::truncate_error_body, prepare::ProviderMessagesRequest,
};
use crate::{constants::MESSAGES_TIMEOUT_SECS, outbound::outbound_request};

pub(super) async fn execute(
    http: &litellm_http::Client,
    auth: &AuthServices,
    request: ProviderMessagesRequest,
    hooks: &impl RouteHooks<Error>,
) -> Result<MessagesResponse, Error> {
    let ProviderMessagesRequest {
        provider,
        url,
        body,
        environment,
        timeout,
        api_key,
    } = request;
    let stream = body.params.stream == Some(true);
    let context = RequestContext {
        model: body.model.clone(),
        custom_llm_provider: provider.as_str().to_string(),
        optional_params: serde_json::to_value(&body.params).map_err(serialize_failure)?,
        secret_fields: Vec::new(),
        api_key,
    };
    let authenticated = resolve_auth(auth, environment, &|key| std::env::var(key).ok()).await?;
    let wire = hooks
        .before_provider_request(
            WireRequest {
                url,
                headers: authenticated.headers,
                body: serde_json::to_value(&body).map_err(serialize_failure)?,
            },
            context,
        )
        .await?;
    let provider_name = provider.as_str();
    log_request_body(provider_name, stream, &wire.body);
    let response = send(
        http,
        Authenticated {
            headers: wire.headers,
            signer: authenticated.signer,
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
    let text = response.text().await.map_err(network)?;
    log_response_body(&text);
    hooks
        .on_event(MachineEvent::ResponseReceived {
            raw: RawResponse { body: text.clone() },
        })
        .await
        .map_err(Error::post_call)?;
    decode_response(config, &body.model, &text)
        .map(|message| MessagesResponse::Complete(Box::new(message)))
}

fn serialize_failure(err: serde_json::Error) -> Error {
    Error::InvalidRequest(format!(
        "failed to serialize Anthropic messages request: {err}"
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
    crate::outbound::send(request, http).await.map_err(network)
}

async fn provider_error(response: reqwest::Response) -> Error {
    let status = response.status().as_u16();
    match response.text().await {
        Ok(text) => {
            log_error_body(status, &text);
            Error::Transport(TransportError::Http {
                status,
                body: truncate_error_body(&text),
            })
        }
        Err(error) => network(error),
    }
}

fn decode_response(
    config: &dyn BaseAnthropicMessagesConfig,
    model: &str,
    text: &str,
) -> Result<AnthropicMessagesResponse, Error> {
    let response = serde_json::from_str(text)
        .map_err(|err| Error::InvalidResponse(format!("invalid messages response JSON: {err}")))?;
    config
        .transform_anthropic_messages_response(model, response)
        .map_err(Error::from)
}

fn streaming_response(
    response: reqwest::Response,
    decoder: Option<StreamDecoder>,
    provider: &'static str,
) -> MessagesResponse {
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
    MessagesResponse::Stream {
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
    litellm_tracing::debug!(provider, stream, body = %body, "provider request");
}

fn log_response_body(body: &str) {
    litellm_tracing::debug!(body, "provider response body");
}

fn log_error_body(status: u16, body: &str) {
    litellm_tracing::debug!(status, body, "provider error body");
}

fn log_chunk(provider: &str, stage: &str, data: &bytes::Bytes) {
    let chunk = ByteChunk::new(data);
    litellm_tracing::debug!(provider, stage, encoding = chunk.encoding(), chunk = %chunk, "stream chunk");
}

#[cfg(test)]
mod tests {
    use litellm_llms::base_llm::anthropic_messages::streaming::anthropic_sse_event_stream;
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
        let MessagesResponse::Stream { mut chunks, .. } =
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
