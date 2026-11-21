//! The callbacks the gateway runs itself around an inference call, the native counterpart of
//! the proxy's `custom_logger_v2` dispatcher. [`Callbacks`] holds them in the order Python runs
//! them, and the [`Call`] of one request runs each hook the route tells it over them in that
//! order. [`Point`] names each hook the gateway runs and how it ends, and a hook it does not run
//! yet is never told.

mod fields;

use std::sync::Arc;

use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use litellm_callbacks::{
    Abort, CallbackError, ErasedCallback, Hook, ManifestError, RequestState, Subscription, Verdict,
    Violation,
};
use litellm_error::{Report, Source};
use litellm_gateway_inference::{CallHooks, CallType, HookError, Refusal, ServedCall};
use parking_lot::Mutex;
use serde_json::{Map, Value};
use strum::VariantArray;

use self::fields::Known;

/// A hook the gateway runs for a call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, VariantArray)]
enum Point {
    PreCall,
    PostCallSuccess,
    ResponseHeaders,
    PostCallFailure,
    AttemptEnded,
}

/// What a callback that aborts or fails on a hook does to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ending {
    /// It ends the hook and the call, as the proxy raises what `pre_call_hook` and
    /// `post_call_success_hook` raise.
    Decides,
    /// It is logged and the next callback still hears the hook, as the proxy only logs the rest.
    Told,
}

impl Point {
    const fn hook(self) -> Hook {
        match self {
            Self::PreCall => Hook::PreCall,
            Self::PostCallSuccess => Hook::PostCallSuccess,
            Self::ResponseHeaders => Hook::ResponseHeaders,
            Self::PostCallFailure => Hook::PostCallFailure,
            Self::AttemptEnded => Hook::AttemptEnded,
        }
    }

    const fn ending(self) -> Ending {
        match self {
            Self::PreCall | Self::PostCallSuccess => Ending::Decides,
            Self::ResponseHeaders | Self::PostCallFailure | Self::AttemptEnded => Ending::Told,
        }
    }
}

/// A callback the gateway cannot run as its manifest declares.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum Unregistrable {
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    #[error("{name} may abort on {hook:?}, where the gateway ends no call")]
    Abort { name: String, hook: Hook },
}

/// A callback that failed on a hook.
#[derive(Debug, thiserror::Error)]
#[error("the {callback} callback failed on {hook:?}")]
struct Failed {
    callback: String,
    hook: Hook,
    #[source]
    cause: Cause,
}

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Callback(#[from] CallbackError),
    #[error(transparent)]
    Violation(#[from] Violation),
}

/// The callbacks every call runs, in the order each hook reaches them: the config's `callbacks`
/// as written, then the proxy's own hooks, as `litellm.callbacks` holds them. None is built from
/// a config yet, so each one a config names still sends its requests to Python.
#[derive(Default)]
pub struct Callbacks {
    callbacks: Vec<Arc<dyn ErasedCallback>>,
}

impl Callbacks {
    /// `callbacks` in the order their hooks run. A manifest that subscribes to what its hook
    /// point cannot do, or that may abort on a hook the gateway ends no call from, is refused
    /// here, so no call runs it.
    pub fn new(callbacks: Vec<Arc<dyn ErasedCallback>>) -> Result<Self, Unregistrable> {
        for callback in &callbacks {
            let manifest = callback.manifest();
            manifest.validate()?;
            let aborting = Point::VARIANTS
                .iter()
                .filter(|point| point.ending() == Ending::Told)
                .map(|point| point.hook())
                .find(|hook| {
                    manifest
                        .subscription(*hook)
                        .is_some_and(|subscription| subscription.may_abort)
                });
            if let Some(hook) = aborting {
                return Err(Unregistrable::Abort {
                    name: manifest.selection.name.clone(),
                    hook,
                });
            }
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
    /// Runs the hook of `point` over the callbacks subscribed to it, in order. On a hook that
    /// decides, the first to abort or fail ends it and the call.
    async fn run(&self, point: Point) -> Result<(), HookError> {
        let hook = point.hook();
        for (callback, state) in self.callbacks.callbacks.iter().zip(&self.states) {
            let manifest = callback.manifest();
            let Some(subscription) = manifest.subscription(hook) else {
                continue;
            };
            let name = &manifest.selection.name;
            let turn = self.turn(callback.as_ref(), name, subscription, state.as_ref());
            let ended = match turn.await {
                Ok(None) => continue,
                Ok(Some(abort)) => HookError::Refused(refusal(abort)),
                Err(cause) => HookError::Failed(Source::new(Failed {
                    callback: name.clone(),
                    hook,
                    cause,
                })),
            };
            match point.ending() {
                Ending::Decides => return Err(ended),
                Ending::Told => {
                    tracing::error!(callback = %name, ?hook, error = %Report::new(&ended), "the callback failed");
                }
            }
        }
        Ok(())
    }

    /// One callback's turn on a hook. It is handed the fields its manifest reads as the
    /// callbacks before it left them, what it returns is held to what its subscription declared,
    /// its patch is written for the callbacks after it, and its abort is handed back.
    async fn turn(
        &self,
        callback: &dyn ErasedCallback,
        name: &str,
        subscription: &Subscription,
        state: Option<&RequestState>,
    ) -> Result<Option<Abort>, Cause> {
        let hook = subscription.hook;
        let input = self.known.lock().input(hook, subscription.reads);
        let Some(verdict) = callback.handle(&input, state).await? else {
            return Ok(None);
        };
        subscription.check(&verdict)?;
        match verdict {
            Verdict::Abort(abort) => Ok(Some(abort)),
            Verdict::Patch(patch) => {
                let mut known = self.known.lock();
                for op in patch.ops {
                    if let Err(error) = known.write(op) {
                        tracing::error!(callback = %name, ?hook, %error, "the patch was not applied");
                    }
                }
                Ok(None)
            }
        }
    }
}

impl CallHooks for Call {
    async fn pre_call(&self, body: Map<String, Value>) -> Result<Map<String, Value>, HookError> {
        self.known.lock().hold(body);
        let ran = self.run(Point::PreCall).await;
        let body = self.known.lock().release();
        ran.map(|()| body)
    }

    async fn post_call_success(&self) -> Result<(), HookError> {
        self.run(Point::PostCallSuccess).await
    }

    async fn response_headers(&self) -> HeaderMap {
        let _ = self.run(Point::ResponseHeaders).await;
        self.known
            .lock()
            .response_headers()
            .iter()
            .filter_map(|(name, value)| header(name, value))
            .collect()
    }

    async fn succeeded(&self, served: ServedCall) {
        self.known.lock().serve(served);
        let _ = self.run(Point::AttemptEnded).await;
    }

    async fn failed(&self) {
        self.known.lock().fail();
        let _ = self.run(Point::PostCallFailure).await;
        let _ = self.run(Point::AttemptEnded).await;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use futures_util::future::BoxFuture;
    use litellm_callbacks::{
        Callback, Field, FieldSet, HookInput, HookOutput, Manifest, Patch, Selection,
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

    fn overwrites(_: &HookInput) -> HookOutput {
        Ok(Some(Verdict::Patch(Patch::default().set(
            Field::RateLimitStatuses,
            json!({"overall_code": "OVER"}),
        ))))
    }

    fn renames(_: &HookInput) -> HookOutput {
        Ok(Some(Verdict::Patch(
            Patch::default().set(Field::Model, json!("other")),
        )))
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

    /// How a hook ended, as the route is told it.
    fn told(ended: HookError) -> String {
        match ended {
            HookError::Refused(refusal) => {
                format!("{} {}", refusal.status.as_u16(), refusal.message)
            }
            HookError::Failed(failure) => Report::new(&failure).to_string(),
        }
    }

    #[rstest]
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
            (admitted.map_err(told), log.lock().clone()),
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
    #[case::an_abort(pre_call(), refuses, "429 slow down")]
    #[case::a_failure(
        pre_call(),
        fails,
        "the first callback failed on PreCall: the counter store is down"
    )]
    #[case::an_abort_its_manifest_does_not_declare(
        Subscription::on(Hook::PreCall),
        refuses,
        "the first callback failed on PreCall: aborted without may_abort"
    )]
    #[case::a_write_its_manifest_does_not_declare(
        pre_call(),
        renames,
        "the first callback failed on PreCall: patched [Model] outside its declared writes"
    )]
    #[tokio::test]
    async fn what_ends_the_pre_call_hook_keeps_later_callbacks_from_the_call(
        #[case] subscription: Subscription,
        #[case] answer: Answer,
        #[case] ended: &str,
    ) {
        let log = Log::default();
        let call = call(vec![
            scripted("first", &log, &[subscription], answer),
            scripted("second", &log, &[pre_call()], silent),
        ]);
        let refused = call.pre_call(body()).await;

        assert_eq!(
            (refused.map_err(told), log.lock().clone()),
            (Err(ended.to_owned()), vec!["first:PreCall:1:-".to_owned()])
        );
    }

    #[rstest]
    #[case::a_failure(fails)]
    #[case::an_abort_its_manifest_does_not_declare(refuses)]
    #[case::a_write_its_manifest_does_not_declare(overwrites)]
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

    #[rstest]
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
            Some(Unregistrable::Manifest(ManifestError::Subscription {
                hook: Hook::PreCall,
                ..
            }))
        ));
    }

    #[rstest]
    fn a_callback_that_may_abort_where_the_gateway_ends_no_call_is_never_registered() {
        let log = Log::default();
        let aborting = Subscription {
            may_abort: true,
            ..Subscription::on(Hook::PostCallFailure)
        };
        let registered = Callbacks::new(vec![scripted("wrong", &log, &[aborting], silent) as _]);
        assert_eq!(
            registered.err(),
            Some(Unregistrable::Abort {
                name: "wrong".to_owned(),
                hook: Hook::PostCallFailure
            })
        );
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
