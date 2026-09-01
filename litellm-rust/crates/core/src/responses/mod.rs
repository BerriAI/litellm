pub mod instrumentation;
pub mod types;
pub mod websocket;

use crate::error::Error;
use crate::streaming::{
    OpenedStream, StreamApi, StreamCapability, StreamTransport, supports_streaming,
};
use types::{ResponsesStreamEvent, ResponsesStreamRequest, ResponsesWebSocketRequest};
use websocket::TypedResponsesWebSocketSession;

pub async fn responses_stream(
    request: ResponsesStreamRequest,
) -> Result<OpenedStream<ResponsesStreamEvent>, Error> {
    let capability = StreamCapability {
        api: StreamApi::Responses,
        provider: request.context.provider,
        transport: StreamTransport::Http,
    };
    if !supports_streaming(capability) {
        return Err(Error::Unsupported("responses HTTP streaming"));
    }
    Err(Error::Unsupported(
        "responses HTTP streaming provider registration",
    ))
}

pub async fn responses_websocket(
    request: ResponsesWebSocketRequest,
) -> Result<Box<dyn TypedResponsesWebSocketSession>, Error> {
    let capability = StreamCapability {
        api: StreamApi::Responses,
        provider: request.context.provider,
        transport: StreamTransport::WebSocket,
    };
    if !supports_streaming(capability) {
        return Err(Error::Unsupported("responses WebSocket streaming"));
    }
    Err(Error::Unsupported(
        "responses WebSocket streaming provider registration",
    ))
}
