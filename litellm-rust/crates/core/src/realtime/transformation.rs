use crate::Error;
use crate::realtime::types::{RealtimeEvent, RealtimeTransformResult};

pub trait RealtimeProviderConfig {
    fn resolve_api_key(
        &self,
        _api_key: Option<&str>,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Err(Error::Unsupported("provider credential resolution"))
    }

    /// Build the upstream WebSocket URL (e.g. `wss://api.openai.com/v1/realtime?model=…`).
    /// Pure string construction only — no network, no env.
    fn complete_url(&self, api_base: Option<&str>, model: &str) -> String;

    /// Transform a client → backend event before it is forwarded upstream.
    fn transform_realtime_request(
        &self,
        event: &RealtimeEvent,
        model: &str,
    ) -> Result<RealtimeTransformResult, Error>;

    /// Transform a backend → client event before it is forwarded downstream.
    fn transform_realtime_response(
        &self,
        event: &RealtimeEvent,
        model: &str,
    ) -> Result<RealtimeTransformResult, Error>;
}
