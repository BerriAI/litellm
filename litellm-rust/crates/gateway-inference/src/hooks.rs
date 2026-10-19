//! Runs the configured hooks around an inference request and hands its record to the observers
//! once the response body ends. Without hooks or observers none of this runs.

use std::{
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
    time::{Duration, Instant, SystemTime},
};

use axum::{
    body::{Body, BodyDataStream, Bytes, HttpBody},
    http::{HeaderMap, HeaderName, HeaderValue, header::CONTENT_TYPE},
    response::Response,
};
use futures_util::Stream;
use litellm_gateway_auth::{AuthenticatedRequest, AuthenticationMethod, Permissions};
use litellm_gateway_hooks::{
    Admission, DeploymentFacts, Facts, KeyFacts, Outcome, Pipeline, RequestFacts, Route,
    StreamFormat, StreamUsage, Usage,
};
use litellm_inference_chat::stream::ReportedUsage;
use litellm_router::Attempt;
use serde_json::{Map, Value};

use crate::{Deployment, Error, Gateway};

/// The user the proxy attributes master key requests to, `litellm_proxy_admin_name`.
const PROXY_ADMIN: &str = "default_user_id";

/// The hooks and observers every inference request runs through.
pub struct Hooks {
    pipeline: Arc<Pipeline>,
    /// The hash of the master key, which the proxy records as the key of its requests.
    master_key_hash: Option<Arc<str>>,
}

impl Hooks {
    pub fn new(pipeline: Arc<Pipeline>, master_key_hash: Option<String>) -> Self {
        Self {
            pipeline,
            master_key_hash: master_key_hash.map(Into::into),
        }
    }

    pub fn pipeline(&self) -> &Arc<Pipeline> {
        &self.pipeline
    }
}

/// One request's admission, and the deployment its last attempt went to. Dropped before it
/// finishes, as when the client goes away mid-route, it records the request as interrupted so
/// the hooks still return what they reserved.
pub(crate) struct Tracking {
    pipeline: Arc<Pipeline>,
    route: Route,
    admission: Option<Admission>,
    /// The deployment the latest attempt went to, and when it was sent.
    deployment: Mutex<Option<(DeploymentFacts, SystemTime)>>,
    started: Instant,
}

/// Builds the request's facts and runs the hooks. `None` when nothing is configured.
pub(crate) async fn admit(
    gateway: &Gateway,
    identity: &AuthenticatedRequest,
    route: Route,
    headers: &HeaderMap,
    raw: &Bytes,
    body: &Map<String, Value>,
    name: &str,
) -> Result<Option<Tracking>, Error> {
    let Some(hooks) = gateway
        .hooks
        .get()
        .filter(|hooks| !hooks.pipeline.is_empty())
    else {
        return Ok(None);
    };
    let pipeline = &hooks.pipeline;
    let key = pipeline
        .reads(Facts::KEY)
        .then(|| key_facts(hooks, identity))
        .flatten();
    let client_headers = match pipeline.reads(Facts::CLIENT_HEADERS) {
        true => Arc::new(owned_headers(headers)),
        false => Arc::default(),
    };
    let request = RequestFacts::new(
        uuid::Uuid::new_v4().to_string().into(),
        route,
        key,
        end_user(body).map(Into::into),
        name.into(),
        client_headers,
        body.get("stream").and_then(Value::as_bool).unwrap_or(false),
        body.get("max_tokens")
            .or_else(|| body.get("max_completion_tokens"))
            .and_then(Value::as_u64),
        raw.clone(),
    );
    let admission = pipeline.admit(Arc::new(request)).await?;
    Ok(Some(Tracking {
        pipeline: pipeline.clone(),
        route,
        admission: Some(admission),
        deployment: Mutex::new(None),
        started: Instant::now(),
    }))
}

/// A copy of the client's headers that owns its bytes. Hyper parses header values, and names it
/// does not know, as slices of the connection's read buffer, so a plain clone kept in the facts
/// would keep that whole buffer for as long as the response streams.
fn owned_headers(headers: &HeaderMap) -> HeaderMap {
    let mut owned = HeaderMap::with_capacity(headers.len());
    for (name, value) in headers {
        let name =
            HeaderName::from_bytes(name.as_str().as_bytes()).unwrap_or_else(|_| name.clone());
        let mut copy = HeaderValue::from_bytes(value.as_bytes()).unwrap_or_else(|_| value.clone());
        copy.set_sensitive(value.is_sensitive());
        owned.append(name, copy);
    }
    owned
}

fn key_facts(hooks: &Hooks, identity: &AuthenticatedRequest) -> Option<Arc<KeyFacts>> {
    let caller = identity.caller();
    let authentication = caller.authentication();
    match (&authentication.method, caller.restrictions()) {
        (AuthenticationMethod::VirtualKey, Permissions::VirtualKey(grant)) => {
            let profile = &grant.profile;
            Some(Arc::new(KeyFacts {
                hash: authentication.credential_id.clone(),
                alias: profile.alias.clone(),
                user_id: profile.user_id.clone(),
                team_id: grant.team_id.clone(),
                team_alias: profile.team_alias.clone(),
                org_id: None,
                metadata: profile.metadata.clone(),
                team_metadata: profile.team_metadata.clone(),
                budgets: profile.budgets.clone(),
                limits: profile.limits,
            }))
        }
        (AuthenticationMethod::MasterKey, _) => Some(Arc::new(KeyFacts {
            hash: hooks
                .master_key_hash
                .as_deref()
                .unwrap_or_default()
                .to_owned(),
            user_id: Some(PROXY_ADMIN.to_owned()),
            ..KeyFacts::default()
        })),
        _ => None,
    }
}

/// The proxy's end user: the body's `user`, else `litellm_metadata.user`, else
/// `metadata.user_id`.
fn end_user(body: &Map<String, Value>) -> Option<&str> {
    let nested = |object: &str, field: &str| body.get(object)?.get(field)?.as_str();
    body.get("user")
        .and_then(Value::as_str)
        .or_else(|| nested("litellm_metadata", "user"))
        .or_else(|| nested("metadata", "user_id"))
        .filter(|user| !user.is_empty())
}

impl Tracking {
    /// Applies the hooks' body changes to an attempt's body.
    pub(crate) fn edit_body(&self, body: &mut Map<String, Value>) {
        if let Some(admission) = &self.admission {
            admission.edit().apply_upstream_body(body);
        }
    }

    /// The headers the gateway would forward, after the hooks' changes; `None` when no hook
    /// changes upstream headers.
    pub(crate) fn edit_headers(
        &self,
        forwarded: Vec<(String, String)>,
    ) -> Option<Vec<(String, String)>> {
        let edit = self.admission.as_ref()?.edit();
        edit.changes_upstream_headers().then(|| {
            let mut forwarded = forwarded;
            edit.apply_upstream_headers(&mut forwarded);
            forwarded
        })
    }

    /// Notes the deployment an attempt went to; the last one is the one that served.
    pub(crate) fn attempt(&self, name: &str, attempt: &Attempt<'_>) {
        if !self.pipeline.reads(Facts::DEPLOYMENT) {
            return;
        }
        let deployment: &Deployment = attempt.deployment;
        let facts = DeploymentFacts {
            id: attempt.id.into(),
            model_group: name.into(),
            model: deployment.model.clone(),
            provider: deployment.custom_llm_provider.clone(),
            api_base: deployment.api_base.clone(),
        };
        *self.deployment.lock().unwrap_or_else(|e| e.into_inner()) =
            Some((facts, SystemTime::now()));
    }

    /// Records a request that ended with an error before a response.
    pub(crate) fn fail(mut self, error: &Error) {
        self.record_failure(error.status().as_u16(), false);
    }

    fn record_failure(&mut self, status: u16, interrupted: bool) {
        let Some(admission) = self.admission.take() else {
            return;
        };
        let (deployment, upstream_started_at) = self.take_deployment();
        let outcome = Outcome {
            deployment,
            status,
            usage: None,
            cost: None,
            time_to_first_byte: None,
            upstream_started_at,
            ended_at: SystemTime::now(),
            interrupted,
        };
        self.pipeline.finish(admission, outcome, None);
    }

    fn take_deployment(&self) -> (Option<DeploymentFacts>, Option<SystemTime>) {
        self.deployment
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .unzip()
    }

    /// Adds the hooks' response headers and follows the body to its end; an error is recorded
    /// at once.
    pub(crate) async fn finish(
        mut self,
        result: Result<Response, Error>,
    ) -> Result<Response, Error> {
        let mut response = match result {
            Ok(response) => response,
            Err(error) => {
                self.fail(&error);
                return Err(error);
            }
        };
        let Some(admission) = self.admission.take() else {
            return Ok(response);
        };
        let (deployment, upstream_started_at) = self.take_deployment();
        let (pipeline, route, started) = (self.pipeline.clone(), self.route, self.started);
        if response.status().is_success() {
            let headers = response.headers_mut();
            for (name, value) in admission.edit().response_headers() {
                headers.insert(name.clone(), value.clone());
            }
        }
        let reported = response.extensions().get::<ReportedUsage>().cloned();
        let streaming = response
            .headers()
            .get(CONTENT_TYPE)
            .is_some_and(|value| value.as_bytes().starts_with(b"text/event-stream"));
        let meter = match (pipeline.reads(Facts::USAGE), streaming, route) {
            (false, _, _) => Meter::Off,
            (true, true, Route::Messages) => {
                Meter::Stream(StreamUsage::new(StreamFormat::Anthropic))
            }
            (true, true, Route::ChatCompletions) => {
                Meter::Stream(StreamUsage::new(StreamFormat::OpenAi))
            }
            (true, false, _) => Meter::Body(Vec::new()),
        };
        let mut finish = Finish {
            body: pipeline.reads(Facts::RESPONSE_BODY).then(Vec::new),
            pipeline,
            route,
            admission: Some(admission),
            deployment,
            upstream_started_at,
            status: response.status().as_u16(),
            meter,
            reported,
            started,
            first_byte: None,
            ended: false,
        };
        if response.body().size_hint().exact().is_some() {
            // A complete body is already in memory: read it now and keep its length header.
            let (parts, body) = response.into_parts();
            let bytes = axum::body::to_bytes(body, usize::MAX)
                .await
                .map_err(|error| Error::Internal(error.to_string()))?;
            finish.feed(&bytes);
            finish.ended = true;
            drop(finish);
            return Ok(Response::from_parts(parts, Body::from(bytes)));
        }
        Ok(response.map(|body| {
            Body::from_stream(Tapped {
                inner: body.into_data_stream(),
                finish,
            })
        }))
    }
}

enum Meter {
    Off,
    Stream(StreamUsage),
    /// A JSON body, read for its `usage` once complete.
    Body(Vec<u8>),
}

impl Drop for Tracking {
    fn drop(&mut self) {
        // 499, the status the proxy's logs give a request the client closed.
        self.record_failure(499, true);
    }
}

/// Records the request when the response body ends or is dropped.
struct Finish {
    pipeline: Arc<Pipeline>,
    route: Route,
    admission: Option<Admission>,
    deployment: Option<DeploymentFacts>,
    upstream_started_at: Option<SystemTime>,
    status: u16,
    meter: Meter,
    /// The usage a translated chat stream ended with, sent to the client or not.
    reported: Option<ReportedUsage>,
    /// The response text for observers that read it. Copied, not kept as chunks: each chunk is a
    /// slice of the upstream connection's read buffer, and holding it would hold that buffer for
    /// the rest of the stream.
    body: Option<Vec<u8>>,
    started: Instant,
    first_byte: Option<Duration>,
    ended: bool,
}

impl Finish {
    fn feed(&mut self, chunk: &Bytes) {
        self.first_byte
            .get_or_insert_with(|| self.started.elapsed());
        match &mut self.meter {
            Meter::Off => {}
            Meter::Stream(meter) => meter.feed(chunk),
            Meter::Body(body) => body.extend_from_slice(chunk),
        }
        if let Some(body) = &mut self.body {
            body.extend_from_slice(chunk);
        }
    }

    /// The usage the response reported, with its cost when the deployment is priced by the
    /// model its response names, which only a chat completion carries.
    fn usage_and_cost(&self) -> (Option<Usage>, Option<f64>) {
        let by_model = self.pipeline.prices().prices_by_response_model()
            && self.route == Route::ChatCompletions;
        let (usage, model) = match &self.meter {
            Meter::Off => (None, None),
            Meter::Stream(meter) => match self.reported.as_ref().and_then(reported_usage) {
                Some(usage) => (Some(usage), None),
                None => (meter.usage(), meter.model().map(str::to_owned)),
            },
            Meter::Body(body) => {
                let model = by_model.then(|| body_model(body)).flatten();
                (body_usage(self.route, body), model)
            }
        };
        let cost = by_model
            .then_some(())
            .zip(self.deployment.as_ref().zip(usage.as_ref()))
            .and_then(|((), (deployment, usage))| {
                self.pipeline
                    .prices()
                    .response_cost(&deployment.id, model.as_deref(), usage)
            });
        (usage, cost)
    }
}

impl Drop for Finish {
    fn drop(&mut self) {
        let Some(admission) = self.admission.take() else {
            return;
        };
        let (usage, cost) = self.usage_and_cost();
        let outcome = Outcome {
            deployment: self.deployment.take(),
            status: self.status,
            usage,
            cost,
            time_to_first_byte: self.first_byte,
            upstream_started_at: self.upstream_started_at,
            ended_at: SystemTime::now(),
            interrupted: !self.ended,
        };
        let body = self.body.take().map(Bytes::from);
        self.pipeline.finish(admission, outcome, body);
    }
}

fn reported_usage(reported: &ReportedUsage) -> Option<Usage> {
    Usage::from_openai(&serde_json::to_value(reported.get()?).ok()?)
}

/// The `usage` object of a complete JSON response. Only that object is parsed: it is found by
/// its key, which inside a string would be escaped.
fn body_usage(route: Route, body: &[u8]) -> Option<Usage> {
    const KEY: &[u8] = b"\"usage\":";
    let start = memchr::memmem::rfind(body, KEY)? + KEY.len();
    let value: Value = serde_json::Deserializer::from_slice(&body[start..])
        .into_iter()
        .next()?
        .ok()?;
    match route {
        Route::Messages => Usage::from_anthropic(&value),
        Route::ChatCompletions => Usage::from_openai(&value),
    }
}

/// The `model` a complete chat completion names, found by its first key like [`body_usage`].
fn body_model(body: &[u8]) -> Option<String> {
    const KEY: &[u8] = b"\"model\":";
    let start = memchr::memmem::find(body, KEY)? + KEY.len();
    serde_json::Deserializer::from_slice(&body[start..])
        .into_iter::<String>()
        .next()?
        .ok()
        .filter(|model| !model.is_empty())
}

struct Tapped {
    inner: BodyDataStream,
    finish: Finish,
}

impl Stream for Tapped {
    type Item = Result<Bytes, axum::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let polled = Pin::new(&mut this.inner).poll_next(cx);
        match &polled {
            Poll::Ready(Some(Ok(chunk))) => this.finish.feed(chunk),
            Poll::Ready(None) => this.finish.ended = true,
            Poll::Ready(Some(Err(_))) | Poll::Pending => {}
        }
        polled
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::user(json!({"user": "u1", "metadata": {"user_id": "u2"}}), Some("u1"))]
    #[case::metadata(json!({"metadata": {"user_id": "u2"}}), Some("u2"))]
    #[case::litellm_metadata(json!({"litellm_metadata": {"user": "u3"}}), Some("u3"))]
    #[case::none(json!({"user": ""}), None)]
    fn end_user_follows_the_proxy(#[case] body: Value, #[case] expected: Option<&str>) {
        assert_eq!(end_user(body.as_object().unwrap()), expected);
    }

    #[rstest]
    fn body_usage_reads_the_top_level_object() {
        let body = json!({
            "id": "msg",
            "content": [{"type": "text", "text": "\"usage\": {\"input_tokens\": 99}"}],
            "usage": {"input_tokens": 3, "output_tokens": 4},
        });
        let usage = body_usage(Route::Messages, &serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!((usage.input_tokens, usage.output_tokens), (3, 4));

        let chat = json!({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 6}});
        let usage =
            body_usage(Route::ChatCompletions, &serde_json::to_vec(&chat).unwrap()).unwrap();
        assert_eq!((usage.prompt_tokens(), usage.output_tokens), (5, 6));
    }

    #[rstest]
    #[case::top_level(
        br#"{"id":"c","model":"gpt-4o-2024-11-20","choices":[]}"#,
        Some("gpt-4o-2024-11-20")
    )]
    #[case::spaced(br#"{"model": "gpt-4o", "choices":[]}"#, Some("gpt-4o"))]
    #[case::inside_content(br#"{"choices":[{"message":{"content":"\"model\":\"x\""}}]}"#, None)]
    #[case::empty(br#"{"model":"","choices":[]}"#, None)]
    fn body_model_reads_the_response_model(#[case] body: &[u8], #[case] expected: Option<&str>) {
        assert_eq!(body_model(body).as_deref(), expected);
    }
}
