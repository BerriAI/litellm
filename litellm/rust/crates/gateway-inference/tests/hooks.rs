//! When a route tells the host's callback hooks about a call, and what they change of it.

mod support;

use std::{sync::Arc, time::Duration};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::Request,
    response::Response,
};
use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Abort, Callback, CallbackError, ErasedCallback, Field, FieldSet, Hook, HookInput, HookOutput,
    Manifest, Patch, Selection, Subscription, Verdict,
};
use parking_lot::Mutex;
use rstest::rstest;
use serde_json::{Map, Value, json};
use tower::ServiceExt;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_partial_json, method},
};

const STREAM: &str = concat!(
    "event: message_start\n",
    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"test-model\",\"content\":[],\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n",
    "event: message_delta\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n",
    "event: message_stop\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

type Answer = fn(Hook) -> HookOutput;

/// A callback on every hook the route tells, which records each one with what it read and
/// answers as `answer` says.
struct Recording {
    manifest: Manifest,
    seen: Mutex<Vec<(Hook, Value)>>,
    answer: Answer,
}

const ENDED: [Field; 3] = [Field::Outcome, Field::Usage, Field::LoggedModelGroup];

impl Recording {
    fn answering(answer: Answer) -> Arc<Self> {
        Arc::new(Self {
            manifest: Manifest {
                selection: Selection::named("recording"),
                subscriptions: vec![
                    Subscription {
                        writes: FieldSet::of(&[Field::OptionalParams]),
                        may_abort: true,
                        ..Subscription::on(Hook::PreCall)
                    },
                    Subscription::on(Hook::PostCallSuccess),
                    Subscription {
                        writes: FieldSet::of(&[Field::ResponseHeaders]),
                        ..Subscription::on(Hook::ResponseHeaders)
                    },
                    Subscription {
                        reads: FieldSet::of(&[Field::Outcome]),
                        ..Subscription::on(Hook::PostCallFailure)
                    },
                    Subscription {
                        reads: FieldSet::of(&ENDED),
                        ..Subscription::on(Hook::AttemptEnded)
                    },
                ],
                has_request_state: false,
            },
            seen: Mutex::default(),
            answer,
        })
    }

    fn silent() -> Arc<Self> {
        Self::answering(|_| Ok(None))
    }

    /// The hooks seen once `count` of them were, since the success log runs after the answer.
    async fn seen(&self, count: usize) -> Vec<(Hook, Value)> {
        for _ in 0..200 {
            if self.seen.lock().len() >= count {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        self.seen.lock().clone()
    }
}

impl Callback for Recording {
    type State = ();

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn handle<'a>(
        &'a self,
        input: &'a HookInput,
        _state: Option<&'a ()>,
    ) -> BoxFuture<'a, HookOutput> {
        let read: Map<String, Value> = ENDED
            .into_iter()
            .filter_map(|field| {
                let value = input.fields.raw(field).ok()??.into_owned();
                Some((format!("{field:?}"), value))
            })
            .collect();
        self.seen.lock().push((input.hook, Value::Object(read)));
        Box::pin(async move { (self.answer)(input.hook) })
    }
}

fn app(upstream: &MockServer, recording: &Arc<Recording>) -> Router {
    app_of("anthropic/test-model", upstream, recording)
}

/// The app whose one deployment, `public/model`, is `model` at `upstream`.
fn app_of(model: &str, upstream: &MockServer, recording: &Arc<Recording>) -> Router {
    let callback: Arc<dyn ErasedCallback> = recording.clone();
    support::app_with_callbacks(model, &upstream.uri(), vec![callback])
}

async fn ask(app: Router, stream: bool) -> Response {
    let body = json!({"model": "public/model", "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16, "stream": stream});
    app.oneshot(
        Request::post("/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap(),
    )
    .await
    .unwrap()
}

async fn answering(template: ResponseTemplate) -> MockServer {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(template)
        .mount(&upstream)
        .await;
    upstream
}

fn message() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({"id": "msg_1", "type": "message",
        "role": "assistant", "model": "test-model", "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 8, "output_tokens": 14, "cache_read_input_tokens": 100}}))
}

fn hooks(seen: &[(Hook, Value)]) -> Vec<Hook> {
    seen.iter().map(|(hook, _)| *hook).collect()
}

fn ended(outcome: &str, prompt: u64, completion: u64, cached: u64) -> Value {
    json!({
        "Outcome": outcome,
        "Usage": {"rejected": null, "streamed": null, "combined": null, "upstream": null,
            "socket_session": null, "logged": null,
            "response": {"rejected": null, "prompt_tokens": prompt,
                "completion_tokens": completion, "total_tokens": prompt + completion,
                "cached_tokens": cached, "prompt_audio_tokens": null, "cache_write_tokens": null,
                "cache_creation_tokens": null, "reasoning_tokens": null,
                "completion_audio_tokens": null, "cache_read_input_tokens": null,
                "cache_creation_input_tokens": null, "duration_seconds": null,
                "output_image_count": null}},
        "LoggedModelGroup": {"rejected": null, "value": "public/model", "error_type": null,
            "error_text": null},
    })
}

#[rstest]
#[tokio::test]
async fn a_complete_answer_runs_the_hooks_in_the_proxys_order_and_logs_what_it_used() {
    let upstream = answering(message()).await;
    let recording = Recording::silent();

    let response = ask(app(&upstream, &recording), false).await;

    assert_eq!(response.status(), 200);
    let seen = recording.seen(4).await;
    assert_eq!(
        (hooks(&seen), seen.last().map(|(_, read)| read.clone())),
        (
            vec![
                Hook::PreCall,
                Hook::PostCallSuccess,
                Hook::ResponseHeaders,
                Hook::AttemptEnded
            ],
            Some(ended("success", 108, 14, 100))
        )
    );
}

#[rstest]
#[tokio::test]
async fn a_stream_is_logged_when_its_body_ends_and_never_reaches_post_call_success() {
    let upstream =
        answering(ResponseTemplate::new(200).set_body_raw(STREAM, "text/event-stream")).await;
    let recording = Recording::silent();

    let response = ask(app(&upstream, &recording), true).await;
    let before_the_body = hooks(&recording.seen.lock());
    to_bytes(response.into_body(), 4096).await.unwrap();

    let seen = recording.seen(3).await;
    assert_eq!(
        (
            before_the_body,
            hooks(&seen),
            seen.last().map(|(_, read)| read.clone())
        ),
        (
            vec![Hook::PreCall, Hook::ResponseHeaders],
            vec![Hook::PreCall, Hook::ResponseHeaders, Hook::AttemptEnded],
            Some(ended("success", 3, 5, 0))
        )
    );
}

#[rstest]
#[tokio::test]
async fn a_stream_the_client_drops_before_its_end_tells_neither_log() {
    let upstream =
        answering(ResponseTemplate::new(200).set_body_raw(STREAM, "text/event-stream")).await;
    let recording = Recording::silent();

    drop(ask(app(&upstream, &recording), true).await);

    assert_eq!(
        hooks(&recording.seen(3).await),
        vec![Hook::PreCall, Hook::ResponseHeaders]
    );
}

#[tokio::test]
async fn what_the_hooks_change_reaches_the_provider_and_the_client() {
    let upstream = MockServer::start().await;
    Mock::given(body_partial_json(json!({"max_tokens": 5})))
        .respond_with(message())
        .expect(1)
        .mount(&upstream)
        .await;
    let recording = Recording::answering(|hook| {
        Ok(match hook {
            Hook::PreCall => Some(Verdict::Patch(Patch::default().merge(
                Field::OptionalParams,
                Map::from_iter([("max_tokens".to_owned(), json!(5))]),
            ))),
            Hook::ResponseHeaders => Some(Verdict::Patch(Patch::default().merge(
                Field::ResponseHeaders,
                Map::from_iter([("x-litellm-test-hook".to_owned(), json!("seen"))]),
            ))),
            _ => None,
        })
    });

    let response = ask(app(&upstream, &recording), false).await;

    assert_eq!(
        (
            response.status().as_u16(),
            response.headers().get("x-litellm-test-hook")
        ),
        (200, Some(&"seen".parse().unwrap()))
    );
}

fn refuses(hook: Hook) -> HookOutput {
    Ok((hook == Hook::PreCall).then(|| Verdict::Abort(Abort::new(429, "slow down"))))
}

fn fails(hook: Hook) -> HookOutput {
    match hook {
        Hook::PreCall => Err(CallbackError::Failed(
            "the counter store is down".to_owned(),
        )),
        _ => Ok(None),
    }
}

#[rstest]
#[case::refuses_in_its_status_and_words(refuses, 429, "rate_limit_error", "slow down")]
#[case::fails_with_what_failed_and_why(
    fails,
    500,
    "api_error",
    "the recording callback failed on PreCall: the counter store is down"
)]
#[tokio::test]
async fn a_call_a_pre_call_hook_ends_is_answered_for_it_and_never_reaches_the_provider(
    #[case] answer: Answer,
    #[case] answered: u16,
    #[case] kind: &str,
    #[case] message: &str,
) {
    let upstream = answering(self::message()).await;
    let recording = Recording::answering(answer);

    let response = ask(app(&upstream, &recording), false).await;

    let status = response.status().as_u16();
    let body = support::json(response).await;
    let seen = recording.seen(3).await;
    assert_eq!(
        (
            status,
            body,
            upstream.received_requests().await.unwrap().len(),
            hooks(&seen),
            seen.last().map(|(_, read)| read["Outcome"].clone())
        ),
        (
            answered,
            json!({"type": "error", "error": {"type": kind, "message": message}}),
            0,
            vec![Hook::PreCall, Hook::PostCallFailure, Hook::AttemptEnded],
            Some(json!("failure"))
        )
    );
}

#[rstest]
#[case::the_provider_refuses_it(false, 3)]
#[case::its_stream_breaks(true, 4)]
#[tokio::test]
async fn a_call_that_fails_tells_the_failure_hooks(#[case] stream: bool, #[case] told: usize) {
    let recording = Recording::silent();
    let response = match stream {
        false => {
            let upstream = answering(ResponseTemplate::new(400).set_body_json(
                json!({"type": "error", "error": {"type": "invalid_request_error", "message": "no"}}),
            ))
            .await;
            ask(app(&upstream, &recording), false).await
        }
        true => {
            let base = support::truncating_upstream("event: message_start\ndata: {}\n\n").await;
            let callback: Arc<dyn ErasedCallback> = recording.clone();
            let app = support::app_with_callbacks("anthropic/test-model", &base, vec![callback]);
            ask(app, true).await
        }
    };
    let _ = to_bytes(response.into_body(), 4096).await;

    let seen = recording.seen(told).await;
    let after_the_call: Vec<(Hook, Value)> = seen
        .into_iter()
        .filter(|(hook, _)| !matches!(hook, Hook::PreCall | Hook::ResponseHeaders))
        .map(|(hook, read)| (hook, read["Outcome"].clone()))
        .collect();
    assert_eq!(
        after_the_call,
        vec![
            (Hook::PostCallFailure, json!("failure")),
            (Hook::AttemptEnded, json!("failure"))
        ]
    );
}

#[rstest]
#[case::messages("/v1/messages")]
#[case::ocr("/v1/ocr")]
#[tokio::test]
async fn every_route_asks_the_pre_call_hook_before_it_reaches_a_provider(#[case] path: &str) {
    let upstream = answering(message()).await;
    let recording = Recording::answering(refuses);

    let response = support::post(
        app(&upstream, &recording),
        path,
        json!({"model": "public/model"}),
    )
    .await;

    assert_eq!(
        (
            response.status().as_u16(),
            upstream.received_requests().await.unwrap().len(),
            hooks(&recording.seen(3).await)
        ),
        (
            429,
            0,
            vec![Hook::PreCall, Hook::PostCallFailure, Hook::AttemptEnded]
        )
    );
}
