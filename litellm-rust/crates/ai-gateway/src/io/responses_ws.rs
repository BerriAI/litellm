//! Compatibility re-exports: the Responses WebSocket transport moved to
//! `litellm-core` (`litellm_core::responses::connection`) so the python bridge
//! can use it without depending on this crate. Re-exported here so existing
//! gateway imports keep working.

pub use litellm_core::responses::connection::{
    ResponsesUpstreamWs, ResponsesWebSocketConnection, ResponsesWebSocketStreaming,
    async_responses_websocket, responses_ws,
};
