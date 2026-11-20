//! The callbacks the gateway runs itself around an inference call, the native counterpart of
//! the proxy's `custom_logger_v2` dispatcher. [`Callbacks`] holds them in the order Python runs
//! them, and the [`Call`] of one request runs each hook the route tells it over them in that
//! order.

mod fields;

use std::sync::Arc;

use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use litellm_callbacks::{Abort, ErasedCallback, Hook, ManifestError, RequestState, Verdict};
use litellm_gateway_inference::{CallHooks, CallType, Refusal, ServedCall};
use parking_lot::Mutex;
use serde_json::{Map, Value};

use self::fields::Known;

/// The callbacks every call runs, in the order each hook reaches them: the config's `callbacks`
/// as written, then the proxy's own hooks, as `litellm.callbacks` holds them. None is built from
/// a config yet, so each one a config names still sends its requests to Python.
#[derive(Default)]
pub struct Callbacks {
    callbacks: Vec<Arc<dyn ErasedCallback>>,
}

impl Callbacks {
    /// `callbacks` in the order their hooks run. A manifest that subscribes to what its hook
    /// point cannot do is refused here, so no call runs it.
    pub fn new(callbacks: Vec<Arc<dyn ErasedCallback>>) -> Result<Self, ManifestError> {
        for callback in &callbacks {
            callback.manifest().validate()?;
        }
        Ok(Self { callbacks })
    }

    pub(crate) fn begin(self: &Arc<Self>, call_type: CallType) -> Call {
        Call {
            states: self
                .callbacks
                .iter()
                .map(|callback| callback.request_state())
                .collect(),
            callbacks: self.clone(),
            known: Mutex::new(Known::of(call_type)),
        }
    }
}

/// The hooks of one request: each callback's own state for it, and what is known of the call so
/// far, which one callback's patch changes for the callbacks after it.
pub struct Call {
    callbacks: Arc<Callbacks>,
    /// By each callback's place in `callbacks`.
    states: Vec<Option<RequestState>>,
    known: Mutex<Known>,
}

/// Whether a callback that fails on `hook` fails the call, as the proxy raises what
/// `pre_call_hook` and `post_call_success_hook` raise and only logs the rest.
const fn decides(hook: Hook) -> bool {
    matches!(hook, Hook::PreCall | Hook::PostCallSuccess)
}

fn refusal(abort: Abort) -> Refusal {
    Refusal {
        status: StatusCode::from_u16(abort.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        message: abort.message,
    }
}

fn header(name: &str, value: &Value) -> Option<(HeaderName, HeaderValue)> {
    let value = match value {
        Value::String(text) => HeaderValue::try_from(text.as_str()),
        other => HeaderValue::try_from(other.to_string()),
    };
    Some((HeaderName::try_from(name).ok()?, value.ok()?))
}

impl Call {
    /// Runs `hook` over the callbacks subscribed to it, in order. Each is handed the fields its
    /// manifest reads as the callbacks before it left them, and the first to abort ends the hook
    /// with its refusal.
    async fn run(&self, hook: Hook) -> Result<(), Refusal> {
        for (callback, state) in self.callbacks.callbacks.iter().zip(&self.states) {
            let manifest = callback.manifest();
            let Some(subscription) = manifest.subscription(hook) else {
                continue;
            };
            let name = &manifest.selection.name;
            let input = self.known.lock().input(hook, subscription.reads);
            let verdict = match callback.handle(&input, state.as_ref()).await {
                Ok(verdict) => verdict,
                Err(error) if decides(hook) => {
                    return Err(Refusal {
                        status: StatusCode::INTERNAL_SERVER_ERROR,
                        message: error.to_string(),
                    });
                }
                Err(error) => {
                    tracing::error!(callback = %name, ?hook, %error, "the callback failed");
                    continue;
                }
            };
            match verdict {
                None => {}
                Some(Verdict::Patch(patch)) => {
                    let mut known = self.known.lock();
                    for op in patch.ops {
                        if let Err(error) = known.write(op) {
                            tracing::error!(callback = %name, ?hook, %error, "the patch was not applied");
                        }
                    }
                }
                Some(Verdict::Abort(abort)) if subscription.may_abort => {
                    return Err(refusal(abort));
                }
                Some(Verdict::Abort(_)) => {
                    tracing::error!(callback = %name, ?hook, "the callback aborted where its manifest says it never does");
                }
            }
        }
        Ok(())
    }
}

impl CallHooks for Call {
    async fn pre_call(&self, body: Map<String, Value>) -> Result<Map<String, Value>, Refusal> {
        self.known.lock().hold(body);
        let ran = self.run(Hook::PreCall).await;
        let body = self.known.lock().release();
        ran.map(|()| body)
    }

    async fn post_call_success(&self) -> Result<(), Refusal> {
        self.run(Hook::PostCallSuccess).await
    }

    async fn response_headers(&self) -> HeaderMap {
        let _ = self.run(Hook::ResponseHeaders).await;
        self.known
            .lock()
            .response_headers()
            .iter()
            .filter_map(|(name, value)| header(name, value))
            .collect()
    }

    async fn succeeded(&self, served: ServedCall) {
        self.known.lock().serve(served);
        let _ = self.run(Hook::AttemptEnded).await;
    }

    async fn failed(&self) {
        self.known.lock().fail();
        let _ = self.run(Hook::PostCallFailure).await;
        let _ = self.run(Hook::AttemptEnded).await;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use futures_util::future::BoxFuture;
    use litellm_callbacks::{
        Callback, CallbackError, Field, FieldSet, HookInput, HookOutput, Manifest, Patch,
        Selection, Subscription,
    };
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    type Log = Arc<Mutex<Vec<String>>>;
    type Answer = fn(&HookInput) -> HookOutput;

    /// A callback that writes `<name>:<what it read of the limit statuses>` to the log all the
    /// callbacks of a test share, counts the hooks it saw of one request in its own state, and
    /// answers as `answer` says.
    struct Scripted {
        manifest: Manifest,
        log: Log,
        answer: Answer,
    }

    fn scripted(name: &str, log: &Log, hooks: &[Subscription], answer: Answer) -> Arc<Scripted> {
        Arc::new(Scripted {
            manifest: Manifest {
                selection: Selection::named(name),
                subscriptions: hooks.to_vec(),
                has_request_state: true,
            },
            log: log.clone(),
            answer,
        })
    }

    impl Callback for Scripted {
        type State = AtomicU32;

        fn manifest(&self) -> &Manifest {
            &self.manifest
        }

        fn request_state(&self) -> Option<AtomicU32> {
            Some(AtomicU32::new(0))
        }

        fn handle<'a>(
            &'a self,
            input: &'a HookInput,
            state: Option<&'a AtomicU32>,
        ) -> BoxFuture<'a, HookOutput> {
            let nth = state.map(|seen| seen.fetch_add(1, Ordering::SeqCst) + 1);
            let statuses = input.fields.raw(Field::RateLimitStatuses).ok().flatten();
            self.log.lock().push(format!(
                "{}:{:?}:{}:{}",
                self.manifest.selection.name,
                input.hook,
                nth.unwrap_or(0),
                statuses.map_or_else(|| "-".to_owned(), ToString::to_string),
            ));
            Box::pin(async move { (self.answer)(input) })
        }
    }

    fn pre_call() -> Subscription {
        Subscription {
            reads: FieldSet::of(&[Field::RateLimitStatuses]),
            writes: FieldSet::of(&[Field::RateLimitStatuses, Field::OptionalParams]),
            may_abort: true,
            ..Subscription::on(Hook::PreCall)
        }
    }

    fn ended() -> Subscription {
        Subscription {
            reads: FieldSet::of(&[Field::RateLimitStatuses]),
            ..Subscription::on(Hook::AttemptEnded)
        }
    }

    fn silent(_: &HookInput) -> HookOutput {
        Ok(None)
    }

    fn publishes(input: &HookInput) -> HookOutput {
        Ok((input.hook == Hook::PreCall).then(|| {
            Verdict::Patch(
                Patch::default()
                    .set(Field::RateLimitStatuses, json!({"overall_code": "OK"}))
                    .merge(
                        Field::OptionalParams,
                        Map::from_iter([("max_tokens".to_owned(), json!(5))]),
                    ),
            )
        }))
    }

    fn refuses(_: &HookInput) -> HookOutput {
        Ok(Some(Verdict::Abort(Abort::new(429, "slow down"))))
    }

    fn fails(_: &HookInput) -> HookOutput {
        Err(CallbackError::Failed(
            "the counter store is down".to_owned(),
        ))
    }

    fn call(callbacks: Vec<Arc<Scripted>>) -> Call {
        let callbacks = callbacks
            .into_iter()
            .map(|callback| callback as Arc<dyn ErasedCallback>)
            .collect();
        Arc::new(Callbacks::new(callbacks).unwrap()).begin(CallType::AnthropicMessages)
    }

    fn body() -> Map<String, Value> {
        Map::from_iter([("model".to_owned(), json!("claude"))])
    }

    fn served() -> ServedCall {
        ServedCall {
            model_group: "claude".to_owned(),
            usage: None,
        }
    }

    #[tokio::test]
    async fn a_patch_is_read_by_the_callbacks_after_it_on_every_later_hook_and_changes_the_body() {
        let log = Log::default();
        let call = call(vec![
            scripted("before", &log, &[pre_call(), ended()], silent),
            scripted("limiter", &log, &[pre_call()], publishes),
            scripted("after", &log, &[pre_call(), ended()], silent),
        ]);
        let admitted = call.pre_call(body()).await;
        call.succeeded(served()).await;

        assert_eq!(
            (admitted, log.lock().clone()),
            (
                Ok(Map::from_iter([
                    ("model".to_owned(), json!("claude")),
                    ("max_tokens".to_owned(), json!(5))
                ])),
                vec![
                    "before:PreCall:1:-".to_owned(),
                    "limiter:PreCall:1:-".to_owned(),
                    r#"after:PreCall:1:{"overall_code":"OK"}"#.to_owned(),
                    r#"before:AttemptEnded:2:{"overall_code":"OK"}"#.to_owned(),
                    r#"after:AttemptEnded:2:{"overall_code":"OK"}"#.to_owned(),
                ]
            )
        );
    }

    #[rstest]
    #[case::an_abort(refuses, StatusCode::TOO_MANY_REQUESTS, "slow down")]
    #[case::a_failure(fails, StatusCode::INTERNAL_SERVER_ERROR, "the counter store is down")]
    #[tokio::test]
    async fn what_ends_the_pre_call_hook_keeps_later_callbacks_from_the_call(
        #[case] answer: Answer,
        #[case] status: StatusCode,
        #[case] message: &str,
    ) {
        let log = Log::default();
        let call = call(vec![
            scripted("first", &log, &[pre_call()], answer),
            scripted("second", &log, &[pre_call()], silent),
        ]);
        let refused = call.pre_call(body()).await;

        assert_eq!(
            (refused, log.lock().clone()),
            (
                Err(Refusal {
                    status,
                    message: message.to_owned()
                }),
                vec!["first:PreCall:1:-".to_owned()]
            )
        );
    }

    #[rstest]
    #[case::a_failure(fails)]
    #[case::an_abort_its_manifest_does_not_declare(refuses)]
    #[tokio::test]
    async fn what_a_callback_does_wrong_on_the_log_never_keeps_the_log_from_the_next(
        #[case] answer: Answer,
    ) {
        let log = Log::default();
        let call = call(vec![
            scripted("first", &log, &[ended()], answer),
            scripted("second", &log, &[ended()], silent),
        ]);

        call.failed().await;

        assert_eq!(
            log.lock().clone(),
            vec![
                "first:AttemptEnded:1:-".to_owned(),
                "second:AttemptEnded:1:-".to_owned()
            ]
        );
    }

    #[tokio::test]
    async fn each_request_has_its_own_state_for_a_callback() {
        let log = Log::default();
        let callbacks = Arc::new(
            Callbacks::new(vec![scripted("counting", &log, &[ended()], silent) as _]).unwrap(),
        );
        callbacks.begin(CallType::AnthropicMessages).failed().await;
        callbacks.begin(CallType::AnthropicMessages).failed().await;

        assert_eq!(
            log.lock().clone(),
            vec!["counting:AttemptEnded:1:-", "counting:AttemptEnded:1:-"]
        );
    }

    #[rstest]
    fn a_callback_whose_manifest_its_hook_point_cannot_honor_is_never_registered() {
        let log = Log::default();
        let awaited_and_aborting = Subscription {
            awaited: true,
            ..pre_call()
        };
        let registered =
            Callbacks::new(vec![
                scripted("wrong", &log, &[awaited_and_aborting], silent) as _,
            ]);
        assert!(matches!(
            registered.err(),
            Some(ManifestError::Subscription {
                hook: Hook::PreCall,
                ..
            })
        ));
    }

    #[rstest]
    #[case::text("x-a", json!("one"), Some("one"))]
    #[case::a_number_as_python_formats_it("x-a", json!(7), Some("7"))]
    #[case::a_name_no_header_can_have("x a", json!("one"), None)]
    fn a_header_a_callback_adds_goes_out_as_text(
        #[case] name: &str,
        #[case] value: Value,
        #[case] sent: Option<&str>,
    ) {
        let header = header(name, &value);
        assert_eq!(
            header.as_ref().and_then(|(_, value)| value.to_str().ok()),
            sent
        );
    }
}
