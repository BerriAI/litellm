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
