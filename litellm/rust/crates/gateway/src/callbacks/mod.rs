//! The callbacks the gateway runs itself around an inference call, the native counterpart of
//! the proxy's `custom_logger_v2` dispatcher. [`Callbacks`] holds them in the order Python runs
//! them, and the [`Call`] of one request runs each hook the route tells it over them in that
//! order, reading and patching the call's record (`litellm_gateway_inference::record`) by contract field
//! through [`fields`]. [`Point`] names each hook the gateway runs and how it ends, and a hook it does not run
//! yet is never told. A hook a callback reading the call's `StandardLoggingPayload` subscribes
//! to builds the payload once from the record, for every such callback on it ([`payloads`]).

mod fields;
pub mod hosted;
pub mod model_capacity;
pub mod parallel_request_limiter;
pub mod payloads;
pub mod s3_log;
pub mod spend_log;
mod spend_log_input;
pub mod standard_logging;

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use axum::http::StatusCode;
use litellm_callbacks::{
    Abort, Callback, CallbackError, ErasedCallback, Hook, ManifestError, RequestState,
    Subscription, Verdict, Violation,
};
use litellm_error::{Report, Source};
use litellm_gateway_auth::Authenticated;
use litellm_gateway_inference::{
    CallHooks, HookError, Refusal, ServedCall,
    record::{self, CallState, Charged},
};
use litellm_router_core::Router;
use litellm_standard_logging::StandardLoggingPayload;
use serde_json::{Map, Value};
use strum::VariantArray;
use tokio::sync::oneshot;

use self::{
    fields::FROM_THE_PAYLOAD,
    hosted::Answers,
    payloads::{Logged, LoggingPayloads},
};
use crate::host::NativeRows;

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
    #[error("no call record is in scope")]
    NoRecord,
}

type FromCaller = Box<dyn Fn(&Authenticated<NativeRows>) -> RequestState + Send + Sync>;

/// One callback of the stage and how its state for a request is made.
pub struct Registered {
    callback: Arc<dyn ErasedCallback>,
    /// For a callback that keeps something of each request's caller, as Python's hooks are
    /// handed the caller's `UserAPIKeyAuth`. `None` takes the callback's own `request_state`.
    from_caller: Option<FromCaller>,
    /// What answers the callback's host calls, for a callback that makes them.
    host: Option<Arc<dyn Answers>>,
    /// The schedule a callback that batches what it logs is flushed on, for as long as it is
    /// registered.
    _flushing: Option<FlushLoop>,
}

/// Flushes a callback every interval from when it is made, and once more when it is dropped, so
/// what the callback still holds of the calls of a config the gateway no longer serves is sent.
struct FlushLoop {
    _stop: oneshot::Sender<()>,
}

impl FlushLoop {
    fn start(callback: Arc<dyn ErasedCallback>, every: Duration) -> Self {
        let (stop, mut stopped) = oneshot::channel::<()>();
        litellm_call_scope::spawn_detached(async move {
            loop {
                tokio::select! {
                    () = tokio::time::sleep(every) => callback.flush().await,
                    _ = &mut stopped => {
                        callback.flush().await;
                        return;
                    }
                }
            }
        });
        Self { _stop: stop }
    }
}

impl Registered {
    pub fn hosted(callback: Arc<dyn ErasedCallback>, host: Arc<dyn Answers>) -> Self {
        Self {
            callback,
            from_caller: None,
            host: Some(host),
            _flushing: None,
        }
    }

    /// `callback`, flushed every `every` while it is registered and once more after, as the
    /// proxy's `CustomBatchLogger` runs `periodic_flush`.
    pub fn flushed_every(callback: Arc<dyn ErasedCallback>, every: Duration) -> Self {
        Self {
            _flushing: Some(FlushLoop::start(callback.clone(), every)),
            ..Self::from(callback)
        }
    }

    /// `callback`, with its state for each request made from that request's caller.
    pub fn for_caller<C: Callback>(
        callback: Arc<C>,
        state: impl Fn(&Authenticated<NativeRows>) -> C::State + Send + Sync + 'static,
    ) -> Self {
        Self {
            callback,
            from_caller: Some(Box::new(move |identity| Arc::new(state(identity)))),
            host: None,
            _flushing: None,
        }
    }
}

impl From<Arc<dyn ErasedCallback>> for Registered {
    fn from(callback: Arc<dyn ErasedCallback>) -> Self {
        Self {
            callback,
            from_caller: None,
            host: None,
            _flushing: None,
        }
    }
}

/// The callbacks every call runs, in the order each hook reaches them: the config's `callbacks`
/// as written, then the proxy's own hooks, as `litellm.callbacks` holds them.
#[derive(Default)]
pub struct Callbacks {
    callbacks: Vec<Registered>,
    /// What builds a call's `StandardLoggingPayload` for the callbacks that read it.
    payloads: LoggingPayloads,
}

impl Callbacks {
    /// `callbacks` in the order their hooks run. A manifest that subscribes to what its hook
    /// point cannot do, or that may abort on a hook the gateway ends no call from, is refused
    /// here, so no call runs it.
    pub fn new(callbacks: Vec<Registered>) -> Result<Self, Unregistrable> {
        for registered in &callbacks {
            let manifest = registered.callback.manifest();
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
        Ok(Self {
            callbacks,
            payloads: LoggingPayloads::default(),
        })
    }

    /// These callbacks, with the payload of a call that failed naming the deployment `router`
    /// registers under the id it last ran on.
    #[must_use]
    pub fn logging_over(self, router: Router) -> Self {
        Self {
            payloads: LoggingPayloads::over(router),
            ..self
        }
    }

    /// Flushes what each callback still batches, as the proxy's shutdown runs
    /// `flush_rust_loggers`.
    pub async fn flush(&self) {
        futures_util::future::join_all(
            self.callbacks
                .iter()
                .map(|registered| registered.callback.flush()),
        )
        .await;
    }

    pub(crate) fn begin(self: &Arc<Self>, identity: &Authenticated<NativeRows>) -> Call {
        Call {
            charged: fields::charged(identity),
            states: self
                .callbacks
                .iter()
                .map(|registered| match &registered.from_caller {
                    Some(state) => Some(state(identity)),
                    None => registered.callback.request_state(),
                })
                .collect(),
            callbacks: self.clone(),
        }
    }
}

/// The hooks of one request and each callback's own state for it. What is known of the call is
/// its record, which one callback's patch changes for the callbacks after it.
pub struct Call {
    callbacks: Arc<Callbacks>,
    /// Who the call is charged to, written to its record with the body the pre-call hook holds.
    charged: Charged,
    /// By each callback's place in `callbacks`.
    states: Vec<Option<RequestState>>,
}

/// The exception Python raises for `abort`: its `ProxyRateLimitError` for a limiter's refusal,
/// with the headers in the order the limiter lists them, and the abort's own status and words
/// for any other.
fn refusal(abort: Abort) -> Refusal {
    if let Some(limited) = parallel_request_limiter::rate_limited(&abort) {
        return limited.into();
    }
    Refusal {
        headers: abort.headers.into_iter().collect(),
        ..Refusal::new(
            StatusCode::from_u16(abort.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            abort.message,
        )
    }
}

impl Call {
    /// Runs the hook of `point` over the callbacks subscribed to it, in order. On a hook that
    /// decides, the first to abort or fail ends it and the call. The time the hook took is added
    /// to the call's callback time.
    async fn run(&self, point: Point) -> Result<(), HookError> {
        let began = Instant::now();
        let ran = self.run_turns(point).await;
        record::write(|entries| entries.callback_time += began.elapsed());
        ran
    }

    async fn run_turns(&self, point: Point) -> Result<(), HookError> {
        let hook = point.hook();
        let mut logged: Option<Option<StandardLoggingPayload>> = None;
        for (registered, state) in self.callbacks.callbacks.iter().zip(&self.states) {
            let callback = &registered.callback;
            let manifest = callback.manifest();
            let Some(subscription) = manifest.subscription(hook) else {
                continue;
            };
            let name = &manifest.selection.name;
            let payload = match subscription.reads.intersection(FROM_THE_PAYLOAD).is_empty() {
                true => None,
                false => logged
                    .get_or_insert_with(|| {
                        record::read(|entries| self.callbacks.payloads.build(entries)).flatten()
                    })
                    .as_ref(),
            };
            let turn = self.turn(registered, name, subscription, state.as_ref(), payload);
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
    /// callbacks before it left them, and the hook's `payload` of the call, what it returns is
    /// held to what its subscription declared, its patch is written for the callbacks after it,
    /// and its abort is handed back.
    async fn turn(
        &self,
        registered: &Registered,
        name: &str,
        subscription: &Subscription,
        state: Option<&RequestState>,
        payload: Option<&StandardLoggingPayload>,
    ) -> Result<Option<Abort>, Cause> {
        let hook = subscription.hook;
        let logged = payload.map(|payload| Logged {
            payload,
            host: self.callbacks.payloads.host(),
        });
        let input =
            record::read(|entries| fields::input(entries, hook, subscription.reads, logged))
                .ok_or(Cause::NoRecord)?;
        let output = match &registered.host {
            Some(host) => {
                let callback = registered.callback.clone();
                hosted::turn(callback, input, state.cloned(), host.as_ref()).await
            }
            None => registered.callback.handle(&input, state).await,
        };
        let Some(verdict) = output? else {
            return Ok(None);
        };
        subscription.check(&verdict)?;
        match verdict {
            Verdict::Abort(abort) => Ok(Some(abort)),
            Verdict::Patch(patch) => {
                record::write(|entries| {
                    for op in patch.ops {
                        if let Err(error) = fields::write(entries, op) {
                            tracing::error!(callback = %name, ?hook, %error, "the patch was not applied");
                        }
                    }
                });
                Ok(None)
            }
        }
    }
}

impl CallHooks for Call {
    async fn pre_call(&self, body: Map<String, Value>) -> Result<Map<String, Value>, HookError> {
        let mut unheld = Some(body);
        record::write(|entries| {
            entries.charged = self.charged.clone();
            if let Some(body) = unheld.take() {
                entries.hold(body);
            }
        });
        let ran = self.run(Point::PreCall).await;
        let body = record::write(CallState::release)
            .or(unheld)
            .unwrap_or_default();
        if ran.is_ok() {
            record::write(|entries| entries.api_call_start = Some(std::time::SystemTime::now()));
        }
        ran.map(|()| body)
    }

    async fn post_call_success(&self) -> Result<(), HookError> {
        self.run(Point::PostCallSuccess).await
    }

    async fn response_headers(&self) {
        let _ = self.run(Point::ResponseHeaders).await;
    }

    async fn succeeded(&self, served: ServedCall) {
        record::write(|entries| entries.serve(served));
        let _ = self.run(Point::AttemptEnded).await;
    }

    async fn failed(&self) {
        record::write(CallState::fail);
        let _ = self.run(Point::PostCallFailure).await;
        let _ = self.run(Point::AttemptEnded).await;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use futures_util::future::BoxFuture;
    use litellm_callbacks::{
        Callback, Field, FieldSet, HookInput, HookOutput, Manifest, Patch, RenderOptions,
        Selection, typed,
    };
    use parking_lot::Mutex;
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    type Log = Arc<Mutex<Vec<String>>>;

    /// Runs `hooks` as one call, with its own record.
    async fn in_call<T>(hooks: impl std::future::Future<Output = T>) -> T {
        record::scope(record::CallRecord::new(), hooks).await
    }
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

    fn registered(callback: Arc<Scripted>) -> Registered {
        Registered::from(callback as Arc<dyn ErasedCallback>)
    }

    async fn call(callbacks: Vec<Arc<Scripted>>) -> Call {
        let callbacks = callbacks.into_iter().map(registered).collect();
        Arc::new(Callbacks::new(callbacks).unwrap()).begin(&crate::test_support::master())
    }

    fn body() -> Map<String, Value> {
        Map::from_iter([("model".to_owned(), json!("claude"))])
    }

    fn served() -> ServedCall {
        ServedCall {
            model_group: "claude".to_owned(),
            deployment: None,
            response_id: None,
            usage: None,
            first_event: None,
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
        ])
        .await;
        let admitted = in_call(async {
            let admitted = call.pre_call(body()).await;
            call.succeeded(served()).await;
            admitted
        })
        .await;

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
        ])
        .await;
        let refused = in_call(call.pre_call(body())).await;

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
        ])
        .await;

        in_call(call.failed()).await;

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
            Callbacks::new(vec![registered(scripted(
                "counting",
                &log,
                &[ended()],
                silent,
            ))])
            .unwrap(),
        );
        let caller = crate::test_support::master();
        in_call(callbacks.begin(&caller).failed()).await;
        in_call(callbacks.begin(&caller).failed()).await;

        assert_eq!(
            log.lock().clone(),
            vec!["counting:AttemptEnded:1:-", "counting:AttemptEnded:1:-"]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_callbacks_state_for_a_request_can_be_made_from_its_caller() {
        let log = Log::default();
        let seeded =
            Registered::for_caller(scripted("seeded", &log, &[ended()], silent), |identity| {
                AtomicU32::new(
                    u32::from(
                        identity.credential == litellm_gateway_auth_core::Credential::MasterKey,
                    ) * 6,
                )
            });
        let callbacks = Arc::new(Callbacks::new(vec![seeded]).unwrap());
        let caller = crate::test_support::master();
        let started_at = 7;

        in_call(callbacks.begin(&caller).failed()).await;

        assert_eq!(
            log.lock().clone(),
            vec![format!("seeded:AttemptEnded:{started_at}:-")]
        );
    }

    /// A callback that writes the payload it is handed on `AttemptEnded`, rendered, and the log
    /// id beside it to the log.
    struct Rendering {
        manifest: Manifest,
        log: Log,
    }

    fn rendering(name: &str, log: &Log) -> Arc<Rendering> {
        Arc::new(Rendering {
            manifest: Manifest {
                selection: Selection::named(name),
                subscriptions: vec![Subscription {
                    reads: FROM_THE_PAYLOAD,
                    ..Subscription::on(Hook::AttemptEnded)
                }],
                has_request_state: false,
            },
            log: log.clone(),
        })
    }

    impl Callback for Rendering {
        type State = ();

        fn manifest(&self) -> &Manifest {
            &self.manifest
        }

        fn handle<'a>(
            &'a self,
            input: &'a HookInput,
            _: Option<&'a ()>,
        ) -> BoxFuture<'a, HookOutput> {
            Box::pin(async move {
                let rendered = match input.fields.payload()? {
                    Some(payload) => payload
                        .render(RenderOptions::default())
                        .await
                        .map_err(CallbackError::Failed)?,
                    None => b"-".to_vec(),
                };
                let id = input.fields.get::<typed::LogId>()?.unwrap_or_default();
                self.log
                    .lock()
                    .push(format!("{id} {}", String::from_utf8_lossy(&rendered)));
                Ok(None)
            })
        }
    }

    #[rstest]
    #[tokio::test]
    async fn the_callbacks_that_read_the_payload_on_a_hook_render_the_one_built_for_it() {
        let log = Log::default();
        let callbacks = Arc::new(
            Callbacks::new(vec![
                Registered::from(rendering("first", &log) as Arc<dyn ErasedCallback>),
                Registered::from(rendering("second", &log) as Arc<dyn ErasedCallback>),
            ])
            .unwrap(),
        );
        let call = callbacks.begin(&crate::test_support::master());
        in_call(async {
            record::write(|entries| {
                entries.call_type = Some(litellm_gateway_inference::CallType::AnthropicMessages);
                entries.call_id = Some("call-1".to_owned());
            });
            call.pre_call(body()).await.unwrap();
            call.failed().await;
        })
        .await;

        let logged = log.lock().clone();
        let documents: Vec<Value> = logged
            .iter()
            .map(|line| serde_json::from_str(line.split_once(' ').unwrap().1).unwrap())
            .collect();
        assert_eq!(
            (
                logged
                    .iter()
                    .map(|line| line.split_once(' ').unwrap().0)
                    .collect::<Vec<_>>(),
                documents[0]["status"].clone(),
                documents[0] == documents[1],
                callbacks.payloads.host().held(),
            ),
            (vec!["call-1", "call-1"], json!("failure"), true, 0)
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_hook_of_a_call_that_has_not_ended_hands_no_payload() {
        let log = Log::default();
        let reading = Arc::new(Rendering {
            manifest: Manifest {
                selection: Selection::named("early"),
                subscriptions: vec![Subscription {
                    reads: FieldSet::of(&[Field::StandardLoggingPayloadRef, Field::LogId]),
                    ..Subscription::on(Hook::PreCall)
                }],
                has_request_state: false,
            },
            log: log.clone(),
        });
        let callbacks = Arc::new(
            Callbacks::new(vec![Registered::from(reading as Arc<dyn ErasedCallback>)]).unwrap(),
        );
        in_call(
            callbacks
                .begin(&crate::test_support::master())
                .pre_call(body()),
        )
        .await
        .unwrap();

        assert_eq!(log.lock().clone(), vec![" -".to_owned()]);
    }

    #[rstest]
    fn a_callback_whose_manifest_its_hook_point_cannot_honor_is_never_registered() {
        let log = Log::default();
        let awaited_and_aborting = Subscription {
            awaited: true,
            ..pre_call()
        };
        let registered = Callbacks::new(vec![registered(scripted(
            "wrong",
            &log,
            &[awaited_and_aborting],
            silent,
        ))]);
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
        let registered = Callbacks::new(vec![registered(scripted(
            "wrong",
            &log,
            &[aborting],
            silent,
        ))]);
        assert_eq!(
            registered.err(),
            Some(Unregistrable::Abort {
                name: "wrong".to_owned(),
                hook: Hook::PostCallFailure
            })
        );
    }
}
