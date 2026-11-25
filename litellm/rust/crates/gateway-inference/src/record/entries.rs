//! What the gateway knows of one call so far, as typed entries. How a callback reads and writes
//! them by contract field is the callbacks' own view (`gateway/src/callbacks/fields.rs`).

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value};

use super::{AttemptRecord, ServedDeployment};
use crate::{CallType, ServedCall, headers::Routing};

const MODEL: &str = "model";
const STREAM: &str = "stream";

/// The request body of one call. The route owns it, so it is only held while the pre-call hook
/// runs, and what it named of the call is all that stays once it is handed back. The model and
/// the stream flag are read from whichever is there, so a change to a held body is what every
/// later reader sees.
enum Request {
    Held(Map<String, Value>),
    Named { model: Option<String>, stream: bool },
}

impl Request {
    fn model(&self) -> Option<&str> {
        match self {
            Self::Held(body) => body.get(MODEL).and_then(Value::as_str),
            Self::Named { model, .. } => model.as_deref(),
        }
    }

    fn stream(&self) -> bool {
        match self {
            Self::Held(body) => body.get(STREAM).and_then(Value::as_bool) == Some(true),
            Self::Named { stream, .. } => *stream,
        }
    }

    fn named(&self) -> Self {
        Self::Named {
            model: self.model().map(str::to_owned),
            stream: self.stream(),
        }
    }
}

/// A header added to a response whose head was already written.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("the response's head was already written")]
pub struct HeadSent;

/// The headers callbacks and Rust code add to the response, in the order they were added, until
/// its head is written. After that the bag is closed and an add is refused.
#[derive(Default)]
struct ResponseHeaders {
    added: Map<String, Value>,
    /// Headers the response carries only when nothing else set one of the same name, such as the
    /// model group's remaining usage.
    defaults: Map<String, Value>,
    sent: bool,
}

/// What the response to a call that ended on an error tells of how it ended.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ErrorResponse {
    /// The error was raised inside the route's handler, past the checks the proxy makes before
    /// it, so its response carries the proxy's error headers.
    pub in_handler: bool,
    /// The timeout of the provider attempt the call ended on, `None` when no provider attempt
    /// raised the error.
    pub timeout: Option<Duration>,
    /// The provider's response headers on the error the call ended on.
    pub provider_headers: Vec<(String, String)>,
}

/// How a call ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ended {
    Served,
    Failed,
}

/// Who a call is charged to, as Python's `UserAPIKeyAuth` names its caller, and what the proxy's
/// `get_sanitized_user_information_from_key` logs of the key as `user_api_key_*` metadata.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Charged {
    /// The hash the key's row is stored under, or Python's alias for the master key.
    pub key_hash: Option<String>,
    pub key_alias: Option<String>,
    pub user_id: Option<String>,
    /// The key's `spend`, `user_api_key_spend`.
    pub spend: Option<f64>,
    /// The key's `max_budget`, `user_api_key_max_budget`.
    pub max_budget: Option<f64>,
    /// `budget_reset_at.isoformat()`, `user_api_key_budget_reset_at`.
    pub budget_reset_at: Option<String>,
    /// The key's `metadata` without its callback settings, `user_api_key_auth_metadata`.
    pub metadata: Option<Map<String, Value>>,
}

/// One call as it stands.
pub struct CallState {
    /// The route the call came in on, which the route names once it runs.
    pub call_type: Option<CallType>,
    /// The request's `x-litellm-call-id`, which the route names once it runs.
    pub call_id: Option<String>,
    /// Python's `litellm_trace_id`, which groups the call's spend log row as its session.
    pub trace_id: String,
    /// Who the call is charged to, which the callbacks name once they begin.
    pub charged: Charged,
    start: SystemTime,
    /// When the pre-call hook handed the body back for routing, Python's `api_call_start_time`.
    pub api_call_start: Option<SystemTime>,
    end: Option<SystemTime>,
    /// When the served stream's first event arrived, when the stream was held before it was
    /// responded to.
    pub first_event: Option<SystemTime>,
    started: Instant,
    request: Request,
    pub rate_limit_statuses: Option<Value>,
    response_headers: ResponseHeaders,
    served: Option<ServedCall>,
    ended: Option<Ended>,
    routing: Option<Routing>,
    /// The deployment that served the call, which `CallState::routed` writes.
    pub served_deployment: Option<ServedDeployment>,
    /// What the attempt that served the call recorded, which `CallState::routed` writes.
    pub attempt: Option<AttemptRecord>,
    /// What each attempt that failed recorded, in the order they failed. A call that ended on an
    /// error has every attempt here.
    pub failed_attempts: Vec<AttemptRecord>,
    /// How the call ended, for a call that ended on an error, whose response names only what
    /// [`CallState::respond_with_error`] was told.
    error_response: Option<ErrorResponse>,
    /// Whether the call's attempts run under a deployment's stream timeout, by the route's own
    /// rule.
    pub stream_timeout: bool,
    /// The served response is a stream, whose head goes out before its body.
    pub stream_response: bool,
    /// The time the callbacks spent on the call's hooks so far.
    pub callback_time: Duration,
}

/// `at` in seconds since the Unix epoch.
pub fn unix(at: SystemTime) -> f64 {
    at.duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
}

impl CallState {
    pub(crate) fn new(trace_id: String) -> Self {
        Self {
            call_type: None,
            call_id: None,
            trace_id,
            charged: Charged::default(),
            start: SystemTime::now(),
            api_call_start: None,
            end: None,
            first_event: None,
            started: Instant::now(),
            request: Request::Named {
                model: None,
                stream: false,
            },
            rate_limit_statuses: None,
            response_headers: ResponseHeaders::default(),
            served: None,
            ended: None,
            routing: None,
            served_deployment: None,
            attempt: None,
            failed_attempts: Vec::new(),
            error_response: None,
            stream_timeout: false,
            stream_response: false,
            callback_time: Duration::ZERO,
        }
    }

    /// What the router did for the call once it served it, the deployment that served it and
    /// what the served attempt recorded. An attempt that failed is in [`Self::failed_attempts`]
    /// instead, a held stream that failed before it sent content among them.
    pub fn routed(
        &mut self,
        routing: Routing,
        deployment: Option<ServedDeployment>,
        attempt: AttemptRecord,
    ) {
        self.routing = Some(routing);
        self.served_deployment = deployment;
        self.attempt = Some(attempt);
    }

    /// The call ended on an error, whose response carries `routing`, what the router did for the
    /// attempt it ended on, in place of what it did for the call before.
    pub(crate) fn respond_with_error(&mut self, routing: Option<Routing>, error: ErrorResponse) {
        self.routing = routing;
        self.error_response = Some(error);
    }

    /// How the call ended, for a call whose response is an error.
    pub fn error_response(&self) -> Option<&ErrorResponse> {
        self.error_response.as_ref()
    }

    /// When the record was made, as the request came in on its route.
    pub fn started(&self) -> Instant {
        self.started
    }

    pub fn routing(&self) -> Option<&Routing> {
        self.routing.as_ref()
    }

    /// When the call began.
    pub fn start(&self) -> SystemTime {
        self.start
    }

    /// When the call was served or failed.
    pub fn end(&self) -> Option<SystemTime> {
        self.end
    }

    /// Takes the route's `body` for the pre-call hook.
    pub fn hold(&mut self, body: Map<String, Value>) {
        self.request = Request::Held(body);
    }

    /// The body as the pre-call hook left it, back to the route, which calls the provider with it
    /// next.
    pub fn release(&mut self) -> Map<String, Value> {
        let named = self.request.named();
        match std::mem::replace(&mut self.request, named) {
            Request::Held(body) => body,
            Request::Named { .. } => Map::new(),
        }
    }

    /// The request body while it is held, `None` once the route has it back.
    pub fn body(&self) -> Option<&Map<String, Value>> {
        match &self.request {
            Request::Held(body) => Some(body),
            Request::Named { .. } => None,
        }
    }

    pub fn body_mut(&mut self) -> Option<&mut Map<String, Value>> {
        match &mut self.request {
            Request::Held(body) => Some(body),
            Request::Named { .. } => None,
        }
    }

    pub fn model(&self) -> Option<&str> {
        self.request.model()
    }

    pub fn stream(&self) -> bool {
        self.request.stream()
    }

    pub fn response_headers(&self) -> &Map<String, Value> {
        &self.response_headers.added
    }

    /// Adds a header to the response. Refused once the response's head was written.
    pub fn add_header(&mut self, name: String, value: Value) -> Result<(), HeadSent> {
        if self.response_headers.sent {
            return Err(HeadSent);
        }
        self.response_headers.added.insert(name, value);
        Ok(())
    }

    /// Adds a header the response carries only when nothing else set one of the same name.
    /// Refused once the response's head was written.
    pub(crate) fn add_default_header(
        &mut self,
        name: String,
        value: Value,
    ) -> Result<(), HeadSent> {
        if self.response_headers.sent {
            return Err(HeadSent);
        }
        self.response_headers.defaults.insert(name, value);
        Ok(())
    }

    /// Closes the header bag as the response's head is written, and hands back what was added
    /// and the defaults. `headers::render_response_headers` is its one caller.
    pub(crate) fn send_head(&mut self) -> (&Map<String, Value>, &Map<String, Value>) {
        self.response_headers.sent = true;
        (
            &self.response_headers.added,
            &self.response_headers.defaults,
        )
    }

    /// Closes the header bag as [`Self::send_head`] does, for another crate's tests.
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn close_head(&mut self) {
        self.send_head();
    }

    pub fn serve(&mut self, served: ServedCall) {
        self.end = Some(SystemTime::now());
        self.served = Some(served);
        self.ended = Some(Ended::Served);
    }

    pub fn fail(&mut self) {
        self.end = Some(SystemTime::now());
        self.ended = Some(Ended::Failed);
    }

    pub fn served(&self) -> Option<&ServedCall> {
        self.served.as_ref()
    }

    pub fn ended(&self) -> Option<Ended> {
        self.ended
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn body(stream: bool) -> Map<String, Value> {
        Map::from_iter([
            ("model".to_owned(), json!("claude")),
            ("stream".to_owned(), json!(stream)),
        ])
    }

    #[rstest]
    #[case::held(false)]
    #[case::released(true)]
    fn the_model_and_the_stream_flag_outlast_the_body(#[case] released: bool) {
        let mut entries = CallState::new("trace-1".to_owned());
        entries.hold(body(true));
        entries
            .body_mut()
            .unwrap()
            .insert(MODEL.to_owned(), json!("claude-fallback"));
        if released {
            entries.release();
        }
        assert_eq!(
            (entries.model(), entries.stream(), entries.body().is_some()),
            (Some("claude-fallback"), true, !released)
        );
    }

    #[rstest]
    fn a_header_added_after_the_head_was_written_is_refused() {
        let mut entries = CallState::new("trace-1".to_owned());
        entries
            .add_header("x-early".to_owned(), json!("yes"))
            .unwrap();
        entries.send_head();
        assert_eq!(
            (
                entries.add_header("x-late".to_owned(), json!("yes")),
                json!(entries.response_headers())
            ),
            (Err(HeadSent), json!({"x-early": "yes"}))
        );
    }
}
