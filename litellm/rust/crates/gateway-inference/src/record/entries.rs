//! What the gateway knows of one call so far, as typed entries. How a callback reads and writes
//! them by contract field is the callbacks' own view (`gateway/src/callbacks/fields.rs`).

use std::time::{SystemTime, UNIX_EPOCH};

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

/// How a call ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ended {
    Served,
    Failed,
}

/// Who a call is charged to, as Python's `UserAPIKeyAuth` names its caller.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Charged {
    /// The hash the key's row is stored under, or Python's alias for the master key.
    pub key_hash: Option<String>,
    pub key_alias: Option<String>,
    pub user_id: Option<String>,
}

/// One call as it stands.
pub struct CallState {
    call_type: Option<CallType>,
    call_id: Option<String>,
    /// Python's `litellm_trace_id`, which groups the call's spend log row as its session.
    trace_id: String,
    charged: Charged,
    start: SystemTime,
    /// When the pre-call hook handed the body back for routing, Python's `api_call_start_time`.
    api_call_start: Option<SystemTime>,
    end: Option<SystemTime>,
    /// When the served stream's first event arrived, as a stream held before it was answered
    /// read it.
    first_event: Option<SystemTime>,
    request: Request,
    rate_limit_statuses: Option<Value>,
    response_headers: ResponseHeaders,
    served: Option<ServedCall>,
    ended: Option<Ended>,
    routing: Option<Routing>,
    served_deployment: Option<ServedDeployment>,
    attempt: Option<AttemptRecord>,
    failed_attempts: Vec<AttemptRecord>,
    /// The call ended on an error, whose response names only what
    /// [`CallState::respond_with_error`] was told.
    error_response: bool,
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
            error_response: false,
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
    pub(crate) fn respond_with_error(&mut self, routing: Option<Routing>) {
        self.routing = routing;
        self.error_response = true;
    }

    pub fn is_error_response(&self) -> bool {
        self.error_response
    }

    pub fn routing(&self) -> Option<&Routing> {
        self.routing.as_ref()
    }

    pub fn served_deployment(&self) -> Option<&ServedDeployment> {
        self.served_deployment.as_ref()
    }

    /// What the attempt that served the call recorded.
    pub fn attempt(&self) -> Option<&AttemptRecord> {
        self.attempt.as_ref()
    }

    /// What each attempt that failed recorded, in the order they failed. A call that ended on an
    /// error has every attempt here.
    pub fn failed_attempts(&self) -> &[AttemptRecord] {
        &self.failed_attempts
    }

    /// Keeps the record of an attempt that failed.
    pub(crate) fn failed(&mut self, attempt: AttemptRecord) {
        self.failed_attempts.push(attempt);
    }

    /// The route the call came in on, which the route names once it runs.
    pub fn serve_as(&mut self, call_type: CallType) {
        self.call_type = Some(call_type);
    }

    pub fn call_type(&self) -> Option<CallType> {
        self.call_type
    }

    /// The request's `x-litellm-call-id`, which the route names once it runs.
    pub fn name_call(&mut self, call_id: String) {
        self.call_id = Some(call_id);
    }

    pub fn call_id(&self) -> Option<&str> {
        self.call_id.as_deref()
    }

    pub fn trace_id(&self) -> &str {
        &self.trace_id
    }

    /// Who the call is charged to, which the callbacks name once they begin.
    pub fn charge_to(&mut self, charged: Charged) {
        self.charged = charged;
    }

    pub fn charged(&self) -> &Charged {
        &self.charged
    }

    /// When the call began.
    pub fn start(&self) -> SystemTime {
        self.start
    }

    /// When the route took the body back to call the provider with it.
    pub fn api_call_start(&self) -> Option<SystemTime> {
        self.api_call_start
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

    /// The pre-call hook let the call on to the provider.
    pub fn call_provider(&mut self) {
        self.api_call_start = Some(SystemTime::now());
    }

    /// A held stream's first event arrived `at`.
    pub fn see_first_event(&mut self, at: SystemTime) {
        self.first_event = Some(at);
    }

    /// When the served stream's first event arrived, when the stream was held before it was
    /// answered.
    pub fn first_event(&self) -> Option<SystemTime> {
        self.first_event
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

    pub fn rate_limit_statuses(&self) -> Option<&Value> {
        self.rate_limit_statuses.as_ref()
    }

    pub fn set_rate_limit_statuses(&mut self, statuses: Value) {
        self.rate_limit_statuses = Some(statuses);
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
