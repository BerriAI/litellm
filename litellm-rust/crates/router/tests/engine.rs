use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
};

use litellm_router::{
    engine::{Engine, Override, Resume, RouteError, RouterCall},
    failure::{Classified, ExceptionClass, Raised, StreamFailure},
    host::{Attempt, Invoked, Op, RouterHost, Target},
    operation::Operation,
    pyrepr::PyNumber,
    random::PythonRandom,
    settings::{FallbackEntry, Settings},
    snapshot::{RoutedDeployment, Snapshot},
    store::{Store, SystemClock},
};
use rstest::rstest;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Scripted {
    Ok,
    Fail(i64, ExceptionClass),
    Stream(StreamFailure),
}

/// Answers each deployment's attempts from its own script; an unscripted attempt succeeds.
#[derive(Default)]
struct ScriptedHost {
    scripts: Mutex<HashMap<String, VecDeque<Scripted>>>,
    attempts: Mutex<Vec<String>>,
    sleeps: Mutex<Vec<f64>>,
    ops: Mutex<Vec<String>>,
}

impl ScriptedHost {
    fn script(self, id: &str, outcomes: &[Scripted]) -> Self {
        self.scripts
            .lock()
            .unwrap()
            .insert(id.into(), outcomes.iter().cloned().collect());
        self
    }

    fn attempts(&self) -> Vec<String> {
        self.attempts.lock().unwrap().clone()
    }
}

fn op_name(op: &Op<String>) -> String {
    match op {
        Op::LogRetry { model, error, .. } => format!("log_retry({model}, {})", raised(error)),
        Op::OpenBucket { id, copy_of, stamp } => {
            format!(
                "open_bucket({id}<-{copy_of}, {}@{})",
                stamp.model_group, stamp.attempted_fallbacks
            )
        }
        Op::StampRetries {
            error, num_retries, ..
        } => format!("stamp({}, {num_retries})", raised(error)),
        Op::MissingTypedFallbacks { error, .. } => format!("missing_typed({})", raised(error)),
        Op::NoFallbackGroup { error, .. } => format!("no_fallback_group({})", raised(error)),
        Op::FallbackOutcome {
            error,
            attempted,
            last,
            ..
        } => format!(
            "outcome({}, {attempted:?}, {})",
            raised(error),
            last.as_ref().map_or("-".into(), raised)
        ),
    }
}

fn raised(error: &Raised<String>) -> String {
    match error {
        Raised::Host(error) => error.clone(),
        Raised::Router { rejection, .. } => format!("{rejection:?}"),
    }
}

impl RouterHost for ScriptedHost {
    type Response = String;
    type Error = String;
    type Fault = ();

    async fn invoke(&self, attempt: Attempt<String>) -> Result<Invoked<String, String>, ()> {
        self.ops
            .lock()
            .unwrap()
            .extend(attempt.ops.iter().map(op_name));
        let Target::Deployment(id) = attempt.target else {
            return Ok(Invoked::Failure {
                error: "mock".into(),
                classified: classified(500, ExceptionClass::InternalServer),
            });
        };
        let index = {
            let mut attempts = self.attempts.lock().unwrap();
            attempts.push(id.clone());
            attempts.len()
        };
        let next = self
            .scripts
            .lock()
            .unwrap()
            .get_mut(&id)
            .and_then(VecDeque::pop_front)
            .unwrap_or(Scripted::Ok);
        Ok(match next {
            Scripted::Ok => Invoked::Success(format!("{id} answered")),
            Scripted::Fail(status, class) => Invoked::Failure {
                error: format!("{id}#{index}"),
                classified: classified(status, class),
            },
            Scripted::Stream(failure) => Invoked::Failure {
                error: format!("{id}#{index}"),
                classified: Classified {
                    stream_failure: Some(failure),
                    ..classified(503, ExceptionClass::ServiceUnavailable)
                },
            },
        })
    }

    async fn sleep(&self, seconds: f64) -> Result<(), ()> {
        self.sleeps.lock().unwrap().push(seconds);
        Ok(())
    }
}

fn classified(status: i64, class: ExceptionClass) -> Classified {
    Classified {
        classes: vec![class],
        type_name: format!("{class:?}Error"),
        message: format!("status {status}"),
        status_code: Some(status),
        sleep_retry_after: -1,
        callbacks_ran: true,
        exact_litellm_type: true,
        ..Classified::default()
    }
}

fn deployment(id: &str, group: &str) -> RoutedDeployment {
    RoutedDeployment {
        id: id.into(),
        model_name: group.into(),
        model: format!("openai/{id}"),
        weight: None,
        cooldown_time: None,
        num_retries: None,
        supports_web_search: None,
    }
}

fn engine(deployments: Vec<RoutedDeployment>, settings: Settings) -> Engine {
    Engine::new(
        Snapshot::new(deployments, settings, vec!["openai".into()]),
        Store::new(Arc::new(SystemClock), None, 1.0),
        PythonRandom::seeded(0),
    )
}

fn settings() -> Settings {
    Settings {
        num_retries: 2,
        max_fallbacks: 5,
        cooldown_time: PyNumber::Int(5),
        model_group_retry_policy: Some(HashMap::new()),
        tunables: litellm_router::settings::Tunables {
            jitter: 0.0,
            ..Default::default()
        },
        ..Settings::default()
    }
}

fn chain(key: &str, targets: &[&str]) -> FallbackEntry {
    FallbackEntry::Keyed {
        key: key.into(),
        targets: targets.iter().map(|target| (*target).to_owned()).collect(),
    }
}

fn call(model: &str) -> RouterCall {
    RouterCall::new(Operation::Completion, model)
}

const SERVER: Scripted = Scripted::Fail(500, ExceptionClass::InternalServer);
const BAD_REQUEST: Scripted = Scripted::Fail(400, ExceptionClass::BadRequest);
const RATE_LIMIT: Scripted = Scripted::Fail(429, ExceptionClass::RateLimit);

#[rstest]
#[tokio::test]
async fn a_retryable_failure_retries_in_the_group_without_waiting() {
    let engine = engine(vec![deployment("a", "g"), deployment("b", "g")], settings());
    let host = ScriptedHost::default()
        .script("a", &[SERVER])
        .script("b", &[SERVER]);

    let routed = engine.route(&host, call("g")).await.ok().unwrap();

    let retries = host.attempts().len() - 1;
    assert!(retries >= 1);
    assert_eq!(routed.outcome.attempted_retries as usize, retries);
    assert_eq!(routed.outcome.max_retries, Some(2));
    assert_eq!(*host.sleeps.lock().unwrap(), vec![0.0; retries]);
}

#[rstest]
#[tokio::test]
async fn a_single_deployment_backs_off_exponentially() {
    let engine = engine(vec![deployment("a", "g")], settings());
    let host = ScriptedHost::default().script("a", &[SERVER, SERVER]);

    let routed = engine.route(&host, call("g")).await.ok().unwrap();

    assert_eq!(routed.outcome.attempted_retries, 2);
    assert_eq!(*host.sleeps.lock().unwrap(), [0.5, 1.0]);
}

#[rstest]
#[tokio::test]
async fn a_non_retryable_failure_falls_back_at_once() {
    let mut settings = settings();
    settings.fallbacks = Some(vec![chain("g", &["h"])]);
    let engine = engine(vec![deployment("a", "g"), deployment("b", "h")], settings);
    let host = ScriptedHost::default().script("a", &[BAD_REQUEST]);

    let routed = engine.route(&host, call("g")).await.ok().unwrap();

    assert_eq!(host.attempts(), ["a", "b"]);
    assert_eq!(routed.outcome.model_group, "h");
    assert_eq!(routed.outcome.attempted_fallbacks, 1);
    assert_eq!(
        *host.ops.lock().unwrap(),
        ["log_retry(g, a#1)", "open_bucket(1<-0, h@1)"]
    );
}

#[rstest]
#[tokio::test]
async fn failed_fallbacks_raise_the_primary_error_and_report_the_last_one() {
    let mut settings = settings();
    settings.fallbacks = Some(vec![chain("g", &["h"])]);
    settings.num_retries = 0;
    let engine = engine(vec![deployment("a", "g"), deployment("b", "h")], settings);
    let host = ScriptedHost::default()
        .script("a", &[SERVER])
        .script("b", &[SERVER]);

    let Err(RouteError::Failed(failed)) = engine.route(&host, call("g")).await else {
        panic!("every group failed");
    };
    let (error, ops) = (failed.error, failed.ops);

    assert_eq!(raised(&error), "a#1");
    assert_eq!(
        ops.iter().map(op_name).collect::<Vec<_>>(),
        ["outcome(a#1, Some([\"h\"]), b#2)"]
    );
}

#[rstest]
#[tokio::test]
async fn no_matching_chain_explains_itself_on_the_top_level_error() {
    let mut settings = settings();
    settings.fallbacks = Some(vec![chain("other", &["h"])]);
    let engine = engine(vec![deployment("a", "g")], settings);
    let host = ScriptedHost::default().script("a", &[BAD_REQUEST]);

    let Err(RouteError::Failed(failed)) = engine.route(&host, call("g")).await else {
        panic!("no fallback applies");
    };
    let (error, ops) = (failed.error, failed.ops);

    assert_eq!(raised(&error), "a#1");
    assert_eq!(
        ops.iter().map(op_name).collect::<Vec<_>>(),
        ["no_fallback_group(a#1)"]
    );
}

#[rstest]
#[tokio::test]
async fn a_rate_limited_deployment_cools_down_and_is_not_picked_again() {
    let engine = engine(vec![deployment("a", "g"), deployment("b", "g")], settings());
    let host = ScriptedHost::default()
        .script("a", &[RATE_LIMIT])
        .script("b", &[RATE_LIMIT]);

    let Err(RouteError::Failed(failed)) = engine.route(&host, call("g")).await else {
        panic!("both deployments cool down");
    };
    let error = failed.error;

    let attempts = host.attempts();
    assert_eq!(attempts.len(), 2);
    assert_ne!(attempts[0], attempts[1]);
    assert!(matches!(error, Raised::Host(_)));
    assert_eq!(engine.store().active_cooldowns(&["a", "b"]).await.len(), 2);
}

#[rstest]
#[tokio::test]
async fn context_window_errors_skip_retries_for_their_own_chain() {
    let mut settings = settings();
    settings.context_window_fallbacks = Some(vec![chain("g", &["big"])]);
    let engine = engine(
        vec![
            deployment("a", "g"),
            deployment("b", "g"),
            deployment("c", "big"),
        ],
        settings,
    );
    let host = ScriptedHost::default()
        .script(
            "a",
            &[Scripted::Fail(400, ExceptionClass::ContextWindowExceeded)],
        )
        .script(
            "b",
            &[Scripted::Fail(400, ExceptionClass::ContextWindowExceeded)],
        );

    let routed = engine.route(&host, call("g")).await.ok().unwrap();

    assert_eq!(host.attempts().len(), 2);
    assert_eq!(routed.outcome.model_group, "big");
}

#[rstest]
#[tokio::test]
async fn a_non_retryable_failure_is_not_retried_without_a_policy() {
    let engine = engine(vec![deployment("a", "g")], settings());
    let host = ScriptedHost::default().script("a", &[BAD_REQUEST]);

    assert!(engine.route(&host, call("g")).await.is_err());
    assert_eq!(host.attempts(), ["a"]);
}

#[rstest]
#[tokio::test]
async fn a_retry_policy_overrides_the_retry_rules() {
    let mut settings = settings();
    settings.retry_policy = Some(litellm_router::settings::RetryPolicy {
        bad_request_error_retries: Some(1),
        ..Default::default()
    });
    let engine = engine(vec![deployment("a", "g")], settings);
    let host = ScriptedHost::default().script("a", &[BAD_REQUEST]);

    let routed = engine.route(&host, call("g")).await.ok().unwrap();

    assert_eq!(host.attempts(), ["a", "a"]);
    assert_eq!(routed.outcome.max_retries, Some(1));
}

#[rstest]
#[tokio::test]
async fn a_shared_logging_object_counts_only_the_first_failure() {
    let engine = engine(
        vec![
            deployment("a", "g"),
            deployment("b", "g"),
            deployment("c", "g"),
        ],
        settings(),
    );
    let host = ScriptedHost::default()
        .script("a", &[RATE_LIMIT])
        .script("b", &[RATE_LIMIT])
        .script("c", &[RATE_LIMIT]);
    let mut shared = call("g");
    shared.shared_logging = true;

    let _ = engine.route(&host, shared).await;

    assert_eq!(
        engine
            .store()
            .active_cooldowns(&["a", "b", "c"])
            .await
            .len(),
        1
    );
}

#[rstest]
#[tokio::test]
async fn disabled_fallbacks_raise_without_an_outcome() {
    let mut settings = settings();
    settings.fallbacks = Some(vec![chain("g", &["h"])]);
    settings.num_retries = 0;
    let engine = engine(vec![deployment("a", "g"), deployment("b", "h")], settings);
    let host = ScriptedHost::default().script("a", &[SERVER]);
    let mut request = call("g");
    request.disable_fallbacks = true;
    request.fallbacks = Override::Inherit;

    let Err(RouteError::Failed(failed)) = engine.route(&host, request).await else {
        panic!("fallbacks are disabled");
    };

    assert_eq!(host.attempts(), ["a"]);
    assert!(failed.ops.is_empty());
}

#[rstest]
#[tokio::test]
async fn exhausted_retries_stamp_the_latest_error() {
    let engine = engine(vec![deployment("a", "g"), deployment("b", "g")], settings());
    let host = ScriptedHost::default()
        .script("a", &[SERVER, SERVER, SERVER])
        .script("b", &[SERVER, SERVER, SERVER]);

    let Err(RouteError::Failed(failed)) = engine.route(&host, call("g")).await else {
        panic!("every attempt fails");
    };
    let (error, ops) = (failed.error, failed.ops);

    assert_eq!(raised(&error), "?#3".replace('?', &host.attempts()[2]));
    assert!(
        ops.iter()
            .map(op_name)
            .any(|op| op == format!("stamp({}, 2)", raised(&error)))
    );
}

#[rstest]
#[tokio::test]
async fn a_stream_failing_before_content_falls_back_without_retrying_the_group() {
    let mut settings = settings();
    settings.fallbacks = Some(vec![chain("g", &["h"])]);
    let engine = engine(
        vec![
            deployment("a", "g"),
            deployment("b", "g"),
            deployment("c", "h"),
        ],
        settings,
    );
    let before_content = Scripted::Stream(StreamFailure::BeforeContent);
    let host = ScriptedHost::default()
        .script("a", &[before_content])
        .script("b", &[before_content]);

    let routed = engine.route(&host, call("g")).await.ok().unwrap();

    assert_eq!(host.attempts().len(), 2);
    assert_eq!(host.attempts()[1], "c");
    assert_eq!(routed.outcome.model_group, "h");
    assert!(host.sleeps.lock().unwrap().is_empty());
}

#[rstest]
#[tokio::test]
async fn a_terminal_stream_failure_is_raised_without_retries_or_fallbacks() {
    let mut settings = settings();
    settings.fallbacks = Some(vec![chain("g", &["h"])]);
    let engine = engine(vec![deployment("a", "g"), deployment("c", "h")], settings);
    let host = ScriptedHost::default().script("a", &[Scripted::Stream(StreamFailure::Terminal)]);

    let Err(RouteError::Failed(failed)) = engine.route(&host, call("g")).await else {
        panic!("a terminal stream failure is raised");
    };

    assert_eq!(host.attempts(), ["a"]);
    assert_eq!(raised(&failed.error), "a#1");
    assert!(failed.ops.is_empty());
}

#[rstest]
#[tokio::test]
async fn a_terminal_stream_failure_in_a_fallback_hop_ends_the_call() {
    let mut settings = settings();
    settings.fallbacks = Some(vec![chain("g", &["h", "i"])]);
    let engine = engine(
        vec![
            deployment("a", "g"),
            deployment("c", "h"),
            deployment("d", "i"),
        ],
        settings,
    );
    let host = ScriptedHost::default()
        .script("a", &[BAD_REQUEST])
        .script("c", &[Scripted::Stream(StreamFailure::Terminal)]);

    let Err(RouteError::Failed(failed)) = engine.route(&host, call("g")).await else {
        panic!("the fallback's stream error is raised");
    };

    assert_eq!(host.attempts(), ["a", "c"]);
    assert_eq!(raised(&failed.error), "c#2");
}

#[rstest]
#[tokio::test]
async fn a_fallback_hop_stream_falls_back_on_its_own_chain_and_reports_its_depth() {
    let mut settings = settings();
    settings.fallbacks = Some(vec![chain("g", &["h"]), chain("h", &["i"])]);
    let engine = engine(
        vec![
            deployment("a", "g"),
            deployment("c", "h"),
            deployment("d", "i"),
        ],
        settings,
    );
    let host = ScriptedHost::default()
        .script("a", &[BAD_REQUEST])
        .script("c", &[Scripted::Stream(StreamFailure::BeforeContent)]);

    let routed = engine.route(&host, call("g")).await.ok().unwrap();

    assert_eq!(host.attempts(), ["a", "c", "d"]);
    assert_eq!(
        (
            routed.outcome.model_group.as_str(),
            routed.outcome.attempted_fallbacks
        ),
        ("i", 2)
    );
}

#[rstest]
#[tokio::test]
async fn a_resumed_stream_falls_back_on_its_hops_chain_skipping_attempted_groups() {
    let mut settings = settings();
    settings.fallbacks = Some(vec![chain("g", &["h", "i"])]);
    let engine = engine(
        vec![
            deployment("a", "g"),
            deployment("c", "h"),
            deployment("d", "i"),
        ],
        settings,
    );
    let host = ScriptedHost::default();
    let resume = Resume {
        group: "h".into(),
        depth: 1,
        original_group: "g".into(),
        attempted_targets: vec!["g".into(), "h".into()],
        error: "c#1".into(),
        classified: Classified {
            stream_failure: Some(StreamFailure::BeforeContent),
            ..classified(503, ExceptionClass::ServiceUnavailable)
        },
        deployment_id: Some("c".into()),
    };

    let routed = engine.resume(&host, call("g"), resume).await.ok().unwrap();

    assert_eq!(host.attempts(), ["d"]);
    assert_eq!(
        (
            routed.outcome.model_group.as_str(),
            routed.outcome.attempted_fallbacks
        ),
        ("i", 2)
    );
}

#[rstest]
#[case::router_budget(None, 2)]
#[case::request_budget(Some(1), 1)]
#[case::request_opts_out(Some(0), 0)]
#[tokio::test]
async fn an_anthropic_stream_failing_before_content_retries_in_the_group_then_falls_back(
    #[case] request_retries: Option<u32>,
    #[case] retries: usize,
) {
    let mut settings = settings();
    settings.fallbacks = Some(vec![chain("g", &["h"])]);
    let engine = engine(vec![deployment("a", "g"), deployment("c", "h")], settings);
    let before_content = Scripted::Stream(StreamFailure::BeforeContent);
    let host = ScriptedHost::default().script("a", &[before_content; 3]);
    let mut call = RouterCall::new(Operation::AnthropicMessages, "g");
    call.num_retries = request_retries;

    let routed = engine.route(&host, call).await.ok().unwrap();

    let mut expected = vec!["a"; 1 + retries];
    expected.push("c");
    assert_eq!(host.attempts(), expected);
    assert_eq!(routed.outcome.model_group, "h");
    assert_eq!(host.sleeps.lock().unwrap().len(), retries);
}

#[rstest]
#[tokio::test]
async fn an_anthropic_stream_retry_that_opens_reports_its_retry_count() {
    let engine = engine(vec![deployment("a", "g")], settings());
    let host =
        ScriptedHost::default().script("a", &[Scripted::Stream(StreamFailure::BeforeContent)]);

    let routed = engine
        .route(&host, RouterCall::new(Operation::AnthropicMessages, "g"))
        .await
        .ok()
        .unwrap();

    assert_eq!(host.attempts(), ["a", "a"]);
    assert_eq!(
        (routed.outcome.attempted_retries, routed.outcome.max_retries),
        (1, Some(2))
    );
}

/// Replaces the registry's snapshot during the first attempt, as a runtime model list change
/// landing while a call is running would.
struct ReplacingHost {
    engine: Arc<Engine>,
    replacement: Mutex<Option<Snapshot>>,
    inner: ScriptedHost,
}

impl RouterHost for ReplacingHost {
    type Response = String;
    type Error = String;
    type Fault = ();

    async fn invoke(&self, attempt: Attempt<String>) -> Result<Invoked<String, String>, ()> {
        if let Some(snapshot) = self.replacement.lock().unwrap().take() {
            self.engine.registry().replace(snapshot);
        }
        self.inner.invoke(attempt).await
    }

    async fn sleep(&self, seconds: f64) -> Result<(), ()> {
        self.inner.sleep(seconds).await
    }
}

#[rstest]
#[tokio::test]
async fn a_running_call_keeps_its_snapshot_while_the_next_call_sees_the_new_one() {
    let engine = Arc::new(engine(vec![deployment("a", "g")], settings()));
    let host = ReplacingHost {
        engine: Arc::clone(&engine),
        replacement: Mutex::new(Some(Snapshot::new(
            vec![deployment("b", "g")],
            settings(),
            vec!["openai".into()],
        ))),
        inner: ScriptedHost::default().script("a", &[SERVER]),
    };

    let running = engine.route(&host, call("g")).await.ok().unwrap();
    let next = engine.route(&host, call("g")).await.ok().unwrap();

    assert_eq!(host.inner.attempts(), ["a", "a", "b"]);
    assert_eq!(
        (running.outcome.deployment_id, next.outcome.deployment_id),
        ("a".to_owned(), "b".to_owned())
    );
}
