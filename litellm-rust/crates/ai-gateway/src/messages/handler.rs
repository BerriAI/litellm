use bytes::Bytes;
use futures_util::StreamExt;
use futures_util::stream::{self, BoxStream};
use litellm_core::CoreResult;
use litellm_core::error::CoreError;
use litellm_core::messages::transformation::{MessagesAuthKind, MessagesStreaming};
use litellm_core::providers::bedrock::aws_base::{resolve_credentials, sign_bedrock_post};
use litellm_core::providers::bedrock::common_utils::aws_auth_config;
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::SystemTime;

use super::client::http_client;
use super::common_utils::truncate_error_body;
use super::types::ProviderMessagesRequest;
pub(super) async fn execute_messages_provider_call(
    request: ProviderMessagesRequest,
) -> CoreResult<Value> {
    let body = serde_json::to_vec(&request.body).map_err(|error| {
        CoreError::InvalidRequest(format!("invalid messages request body: {error}"))
    })?;
    let headers = signed_headers(&request, &body).await?;
    let mut request_builder = http_client().post(&request.url).body(body);
    for (key, value) in &headers {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = request_builder
        .send()
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?;

    let status = response.status();
    let text = response
        .text()
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?;

    if !status.is_success() {
        return Err(CoreError::Http {
            status: status.as_u16(),
            body: truncate_error_body(&text),
        });
    }

    let response = serde_json::from_str(&text).map_err(|err| {
        CoreError::InvalidResponse(format!("invalid messages response JSON: {err}"))
    })?;
    let transformed = request
        .config
        .transform_response(&request.model, response)?;
    serde_json::to_value(transformed).map_err(|err| {
        CoreError::InvalidResponse(format!("failed to serialize messages response: {err}"))
    })
}

pub(super) async fn execute_messages_provider_stream(
    request: ProviderMessagesRequest,
) -> CoreResult<MessagesStream> {
    if matches!(request.streaming, MessagesStreaming::Unsupported) {
        return Err(CoreError::InvalidRequest(
            "streaming messages is not supported for this provider".to_string(),
        ));
    }

    let body = serde_json::to_vec(&request.body).map_err(|error| {
        CoreError::InvalidRequest(format!("invalid messages request body: {error}"))
    })?;
    let headers = signed_headers(&request, &body).await?;
    let mut request_builder = http_client().post(&request.url).body(body);
    for (key, value) in &headers {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = request_builder
        .send()
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?;
    let status = response.status();
    if !status.is_success() {
        let text = response
            .text()
            .await
            .map_err(|err| CoreError::Network(err.to_string()))?;
        return Err(CoreError::Http {
            status: status.as_u16(),
            body: truncate_error_body(&text),
        });
    }
    let content_type = response_content_type(request.streaming, response.headers());
    let cache_control = response
        .headers()
        .get(reqwest::header::CACHE_CONTROL)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let body = match request.streaming {
        MessagesStreaming::SsePassthrough | MessagesStreaming::Unsupported => response
            .bytes_stream()
            .map(|result| result.map_err(|error| CoreError::Network(error.to_string())))
            .boxed(),
        MessagesStreaming::BedrockEventStream => bedrock_stream(response),
    };
    Ok(MessagesStream {
        content_type,
        cache_control,
        body,
    })
}

fn response_content_type(
    streaming: MessagesStreaming,
    headers: &reqwest::header::HeaderMap,
) -> String {
    if matches!(streaming, MessagesStreaming::BedrockEventStream) {
        return "text/event-stream".to_string();
    }
    headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("text/event-stream")
        .to_string()
}

#[cfg_attr(not(feature = "server"), allow(dead_code))]
pub(crate) struct MessagesStream {
    pub(crate) content_type: String,
    pub(crate) cache_control: Option<String>,
    pub(crate) body: BoxStream<'static, CoreResult<Bytes>>,
}

fn bedrock_stream(response: reqwest::Response) -> BoxStream<'static, CoreResult<Bytes>> {
    use std::collections::VecDeque;

    use litellm_core::providers::bedrock::messages::streaming::{
        BedrockEventStreamDecoder, serialize_sse,
    };

    let upstream = response.bytes_stream().boxed();
    stream::unfold(
        (
            upstream,
            BedrockEventStreamDecoder::new(),
            VecDeque::<Bytes>::new(),
        ),
        |(mut upstream, mut decoder, mut pending)| async move {
            loop {
                if let Some(bytes) = pending.pop_front() {
                    return Some((Ok(bytes), (upstream, decoder, pending)));
                }
                let chunk = upstream.next().await?;
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        return Some((
                            Err(CoreError::Network(error.to_string())),
                            (upstream, decoder, pending),
                        ));
                    }
                };
                match decoder.push(&chunk) {
                    Ok(events) => {
                        for event in events {
                            match serialize_sse(&event) {
                                Ok(bytes) => pending.push_back(Bytes::from(bytes)),
                                Err(error) => {
                                    return Some((Err(error), (upstream, decoder, pending)));
                                }
                            }
                        }
                    }
                    Err(error) => {
                        return Some((Err(error), (upstream, decoder, pending)));
                    }
                }
            }
        },
    )
    .boxed()
}

async fn signed_headers(
    request: &ProviderMessagesRequest,
    body: &[u8],
) -> CoreResult<Vec<(String, String)>> {
    if let MessagesAuthKind::AwsSigV4 { region } = &request.auth_kind {
        let env_lookup = environment_lookup;
        let credentials = resolve_credentials(
            aws_auth_config(&serde_json::Map::new(), &env_lookup, Some(region)),
            &env_lookup,
        )
        .await?;
        let host = reqwest::Url::parse(&request.url)
            .map_err(|error| CoreError::InvalidRequest(format!("invalid Bedrock URL: {error}")))?
            .host_str()
            .ok_or_else(|| CoreError::InvalidRequest("Bedrock URL has no host".to_string()))?
            .to_string();
        let mut headers = BTreeMap::from([
            ("content-type".to_string(), "application/json".to_string()),
            ("host".to_string(), host),
        ]);
        let signed = sign_bedrock_post(
            &request.url,
            body,
            &headers,
            region,
            &credentials,
            SystemTime::now(),
        )?;
        headers.extend(signed);
        return Ok(headers.into_iter().collect());
    }
    Ok(request.upstream_headers.clone())
}

fn environment_lookup(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

#[cfg(test)]
mod tests {
    use super::response_content_type;
    use litellm_core::messages::transformation::MessagesStreaming;
    use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};

    #[test]
    fn bedrock_transcoding_uses_anthropic_sse_content_type() {
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/vnd.amazon.eventstream"),
        );
        assert_eq!(
            response_content_type(MessagesStreaming::BedrockEventStream, &headers),
            "text/event-stream"
        );
    }

    #[test]
    fn passthrough_stream_keeps_upstream_content_type() {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        assert_eq!(
            response_content_type(MessagesStreaming::SsePassthrough, &headers),
            "text/event-stream"
        );
    }
}
