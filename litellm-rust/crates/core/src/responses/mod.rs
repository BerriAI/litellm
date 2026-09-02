pub mod instrumentation;
pub mod types;
pub mod websocket;

use crate::error::Error;
use crate::streaming::OpenedStream;
use types::{ResponsesStreamEvent, ResponsesStreamRequest, ResponsesWebSocketRequest};
use websocket::TypedResponsesWebSocketSession;

pub async fn responses_stream(
    _request: ResponsesStreamRequest,
) -> Result<OpenedStream<ResponsesStreamEvent>, Error> {
    Err(Error::Unsupported(
        "responses HTTP streaming provider registration",
    ))
}

pub async fn responses_websocket(
    _request: ResponsesWebSocketRequest,
) -> Result<Box<dyn TypedResponsesWebSocketSession>, Error> {
    Err(Error::Unsupported(
        "responses WebSocket streaming provider registration",
    ))
}

#[cfg(test)]
mod stream_entrypoint_tests {
    use serde_json::json;

    use super::*;
    use crate::streaming::{
        ProviderCredentials, StreamProviderId, StreamTarget, StreamTransportOptions,
    };

    fn target() -> StreamTarget {
        StreamTarget::new(
            StreamProviderId::OpenAi,
            ProviderCredentials::default(),
            None,
        )
    }

    #[tokio::test]
    async fn typed_http_stream_declines_until_a_provider_is_registered() {
        let body = serde_json::from_value(json!({
            "model": "gpt-5",
            "input": "hello",
            "stream": true
        }))
        .expect("valid Responses stream request");
        let result = responses_stream(ResponsesStreamRequest {
            body,
            target: target(),
            transport: StreamTransportOptions::default(),
        })
        .await;

        assert!(matches!(
            result,
            Err(Error::Unsupported(
                "responses HTTP streaming provider registration"
            ))
        ));
    }

    #[tokio::test]
    async fn typed_websocket_declines_until_a_provider_is_registered() {
        let result = responses_websocket(ResponsesWebSocketRequest {
            target: target(),
            transport: StreamTransportOptions::default(),
        })
        .await;

        assert!(matches!(
            result,
            Err(Error::Unsupported(
                "responses WebSocket streaming provider registration"
            ))
        ));
    }
}
