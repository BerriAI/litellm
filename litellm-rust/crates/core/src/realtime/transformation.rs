use crate::Error;
use crate::realtime::types::{RealtimeEvent, RealtimeTransformResult};

pub trait RealtimeProviderConfig: Sync {
    /// Build the upstream WebSocket URL (e.g. `wss://api.openai.com/v1/realtime?model=…`).
    /// Pure string construction only — no network, no env.
    fn complete_url(&self, api_base: Option<&str>, model: &str) -> Result<String, Error>;

    /// Transform a client → backend event before it is forwarded upstream.
    fn transform_request(
        &self,
        event: &RealtimeEvent,
        model: &str,
    ) -> Result<RealtimeTransformResult, Error>;

    /// Transform a backend → client event before it is forwarded downstream.
    fn transform_response(
        &self,
        event: &RealtimeEvent,
        model: &str,
    ) -> Result<RealtimeTransformResult, Error>;
}
