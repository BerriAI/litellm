use crate::realtime::types::RealtimeTransformResult;
use crate::CoreResult;

pub trait RealtimeProviderConfig {
    /// Build the upstream WebSocket URL (e.g. `wss://api.openai.com/v1/realtime?model=…`).
    /// Pure string construction only — no network, no env.
    fn complete_url(&self, api_base: Option<&str>, model: &str) -> String;

    /// Transform a client → backend message before it is forwarded upstream.
    fn transform_realtime_request(
        &self,
        message: &str,
        model: &str,
    ) -> CoreResult<RealtimeTransformResult>;

    /// Transform a backend → client message before it is forwarded downstream.
    fn transform_realtime_response(
        &self,
        message: &str,
        model: &str,
    ) -> CoreResult<RealtimeTransformResult>;
}
