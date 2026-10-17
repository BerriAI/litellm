use litellm_host::lifecycle::ExecutionEvent;
use std::{
    ops::ControlFlow,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use futures_util::{StreamExt, stream};
use litellm_host::{
    call::{CallOutput, HostedCompletion, hosted_call},
    interceptors::{Interceptors, RawResponse, RequestContext, WireRequest},
    lifecycle::{CallEvent, CallObserver},
    machine::{CallMachine, HostFailure, Interrupted, Machine, MachineFault, Step},
    protocol::{Protocol, Reply},
};
use litellm_host_native::{
    Boundary, Driver,
    in_process::{Host, StreamConsumer, run, run_hosted},
    services::HostCallHandler,
};
use rstest::{fixture, rstest};
use serde_json::json;

#[derive(Clone, Debug, PartialEq)]
enum TestError {
    Provider,
    Service,
    Hook,
    Consumer,
    Machine,
}

impl From<MachineFault> for TestError {
    fn from(_: MachineFault) -> Self {
        Self::Machine
    }
}

struct TestProtocol;

impl Protocol for TestProtocol {
    type Request = &'static str;
    type Response = String;
    type Error = TestError;
    type HostCall = Reply<&'static str>;
    type Chunk = usize;
    type StreamHead = &'static str;
}

type TestMachine = litellm_host::call::HostedMachine<TestProtocol>;

struct Services {
    reject: bool,
}

impl HostCallHandler<TestProtocol> for Services {
    async fn handle_host_call(&self, reply: Reply<&'static str>) -> Result<(), TestError> {
        if self.reject {
            return Err(TestError::Service);
        }
        reply.send("custom");
        Ok(())
    }
}

#[derive(Default)]
struct Observer(Observations);
struct Observations {
    sender: litellm_host::observation::ObservationSender,
    receiver: Mutex<tokio::sync::mpsc::Receiver<CallEvent>>,
    recorded: Mutex<Vec<CallEvent>>,
}

impl Default for Observations {
    fn default() -> Self {
        let (sender, receiver) = litellm_host::observation::observation_channel(
            std::num::NonZeroUsize::new(128).unwrap(),
        );
        Self {
            sender,
            receiver: Mutex::new(receiver),
            recorded: Mutex::new(Vec::new()),
        }
    }
}

impl Observations {
    fn lock(&self) -> std::sync::LockResult<std::sync::MutexGuard<'_, Vec<CallEvent>>> {
        let mut events = self.recorded.lock()?;
        let mut receiver = self.receiver.lock().unwrap();
        while let Ok(event) = receiver.try_recv() {
            events.push(event);
        }
        Ok(events)
    }
}

impl CallObserver for Observer {
    fn observe(&self, event: CallEvent) {
        self.0.sender.emit(event);
    }
}

struct Hooks {
    observer: Arc<Observer>,
    reject: bool,
}

impl Interceptors<TestError> for Hooks {
    async fn before_provider_request(
        &self,
        wire: WireRequest,
        context: RequestContext,
    ) -> Result<WireRequest, TestError> {
        if self.reject {
            return Err(TestError::Hook);
        }
        Ok(WireRequest {
            url: format!("{}/{}", wire.url, context.model),
            ..wire
        })
    }

    async fn after_provider_response(&self, raw: RawResponse) -> Result<(), TestError> {
        if self.reject {
            return Err(TestError::Hook);
        }
        self.observer.observe(CallEvent::Execution(
            litellm_host::lifecycle::ExecutionEvent::ProviderResponseReceived { raw },
        ));
        Ok(())
    }
}

#[fixture]
fn observer() -> Arc<Observer> {
    Arc::new(Observer::default())
}

#[fixture]
fn interceptors(observer: Arc<Observer>) -> Hooks {
    Hooks {
        observer,
        reject: false,
    }
}

fn dispatching_call() -> TestMachine {
    hosted_call::<TestProtocol, _, _>(
        "projected",
        None,
        |request, services, route_hooks, _observations| async move {
            let custom = services.call(|reply| reply).await?;
            let wire = route_hooks
                .before_provider_request(
                    WireRequest {
                        url: request.into(),
                        headers: Vec::new(),
                        body: json!({}),
                    },
                    RequestContext {
                        model: custom.into(),
                        custom_llm_provider: "test".into(),
                        optional_params: json!({}),
                        secret_fields: Vec::new(),
                        api_key: None,
                    },
                )
                .await?;
            route_hooks
                .after_provider_response(RawResponse {
                    body: wire.url.clone(),
                })
                .await?;
            Ok(CallOutput::Complete(wire.url))
        },
    )
}

struct Release(Arc<AtomicBool>);

impl Drop for Release {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

struct Streaming {
    polls: Arc<AtomicUsize>,
    released: Arc<AtomicBool>,
    machine: TestMachine,
}

fn streaming_call(chunks: Vec<Result<usize, TestError>>) -> Streaming {
    let polls = Arc::new(AtomicUsize::new(0));
    let released = Arc::new(AtomicBool::new(false));
    let provider_polls = polls.clone();
    let release = Release(released.clone());
    let machine = hosted_call::<TestProtocol, _, _>(
        "input",
        None,
        move |_, _, _, _observations| async move {
            let chunks = stream::iter(chunks)
                .inspect(move |_| {
                    let _held = &release;
                    provider_polls.fetch_add(1, Ordering::SeqCst);
                })
                .boxed();
            Ok(CallOutput::Stream {
                head: "headers",
                chunks,
            })
        },
    );
    Streaming {
        polls,
        released,
        machine,
    }
}

#[rstest]
#[tokio::test]
async fn services_and_hooks_answer_the_machine_inline(interceptors: Hooks) {
    let observer = interceptors.observer.clone();
    let mut driver = Driver::new(dispatching_call(), Services { reject: false }, interceptors);
    let Boundary::Complete(HostedCompletion::Complete(response)) = driver.advance().await.unwrap()
    else {
        panic!("a unary call completes at the first boundary")
    };
    assert_eq!(response, "projected/custom");
    assert!(matches!(
        observer.0.lock().unwrap().as_slice(),
        [CallEvent::Execution(ExecutionEvent::ProviderResponseReceived { raw })] if raw.body == "projected/custom"
    ));
}

#[rstest]
#[case::service(true, false, TestError::Service)]
#[case::hook(false, true, TestError::Hook)]
#[tokio::test]
async fn handler_failures_interrupt_the_machine(
    observer: Arc<Observer>,
    #[case] reject_service: bool,
    #[case] reject_hook: bool,
    #[case] expected: TestError,
) {
    let mut driver = Driver::new(
        dispatching_call(),
        Services {
            reject: reject_service,
        },
        Hooks {
            observer: observer.clone(),
            reject: reject_hook,
        },
    );
    assert!(matches!(driver.advance().await, Err(error) if error == expected));
    assert!(observer.0.lock().unwrap().is_empty());
}

#[rstest]
#[tokio::test]
async fn advancing_delivers_one_chunk_per_demand() {
    let Streaming { polls, machine, .. } = streaming_call(vec![Ok(0), Ok(1)]);
    let mut driver = Driver::new(machine, Services { reject: false }, ());
    assert!(matches!(
        driver.advance().await,
        Ok(Boundary::Open("headers"))
    ));
    assert_eq!(polls.load(Ordering::SeqCst), 0);
    for index in 0..2 {
        assert!(matches!(driver.advance().await, Ok(Boundary::Chunk(chunk)) if chunk == index));
        assert_eq!(polls.load(Ordering::SeqCst), index + 1);
    }
    assert!(matches!(
        driver.advance().await,
        Ok(Boundary::Complete(HostedCompletion::StreamEnded))
    ));
    assert_eq!(polls.load(Ordering::SeqCst), 2);
}

#[rstest]
#[tokio::test]
async fn provider_stream_errors_surface_at_the_failing_chunk() {
    let Streaming { polls, machine, .. } =
        streaming_call(vec![Ok(0), Err(TestError::Provider), Ok(2)]);
    let mut driver = Driver::new(machine, Services { reject: false }, ());
    assert!(matches!(driver.advance().await, Ok(Boundary::Open(_))));
    assert!(matches!(driver.advance().await, Ok(Boundary::Chunk(0))));
    assert_eq!(driver.advance().await.err(), Some(TestError::Provider));
    assert_eq!(polls.load(Ordering::SeqCst), 2);
}

#[rstest]
#[tokio::test]
async fn detaching_completes_without_pulling_more_chunks() {
    let Streaming { polls, machine, .. } = streaming_call(vec![Ok(0), Ok(1)]);
    let mut driver = Driver::new(machine, Services { reject: false }, ());
    assert!(matches!(driver.advance().await, Ok(Boundary::Open(_))));
    assert!(matches!(driver.advance().await, Ok(Boundary::Chunk(0))));
    assert!(matches!(
        driver.detach().await,
        Ok(Boundary::Complete(HostedCompletion::Detached))
    ));
    assert_eq!(polls.load(Ordering::SeqCst), 1);
}

#[rstest]
#[case::at_open(0)]
#[case::after_chunk(1)]
#[tokio::test]
async fn dropping_the_driver_drops_the_call(#[case] chunks_before_drop: usize) {
    let Streaming {
        polls,
        released,
        machine,
    } = streaming_call(vec![Ok(0), Ok(1)]);
    let mut driver = Driver::new(machine, Services { reject: false }, ());
    assert!(matches!(driver.advance().await, Ok(Boundary::Open(_))));
    for _ in 0..chunks_before_drop {
        assert!(matches!(driver.advance().await, Ok(Boundary::Chunk(_))));
    }
    assert!(!released.load(Ordering::SeqCst));
    drop(driver);
    assert!(released.load(Ordering::SeqCst));
    assert_eq!(polls.load(Ordering::SeqCst), chunks_before_drop);
}

struct Interruptible {
    inner: TestMachine,
    interrupted: Arc<Mutex<Vec<HostFailure<TestError>>>>,
}

impl Machine for Interruptible {
    type Protocol = TestProtocol;
    type Complete = HostedCompletion<String>;

    fn resume(&mut self) -> Step<'_, Self> {
        self.inner.resume()
    }

    fn interrupt(&mut self, failure: HostFailure<TestError>) -> Interrupted<'_, Self> {
        self.interrupted.lock().unwrap().push(failure.clone());
        self.inner.interrupt(failure)
    }
}

struct Consumer {
    detach_after: Option<usize>,
    fail_after: Option<usize>,
    delivered: Mutex<Vec<usize>>,
}

impl Consumer {
    fn demand_after(&self, delivered: usize) -> Result<ControlFlow<()>, TestError> {
        if self.fail_after == Some(delivered) {
            return Err(TestError::Consumer);
        }
        Ok(if self.detach_after == Some(delivered) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        })
    }
}

impl StreamConsumer<TestProtocol> for Consumer {
    async fn open_stream(&self, head: &'static str) -> Result<ControlFlow<()>, TestError> {
        assert_eq!(head, "headers");
        self.demand_after(0)
    }

    async fn send_chunk(&self, chunk: usize) -> Result<ControlFlow<()>, TestError> {
        let mut delivered = self.delivered.lock().unwrap();
        delivered.push(chunk);
        self.demand_after(delivered.len())
    }
}

#[rstest]
#[case::consumed(None, None, Ok(HostedCompletion::StreamEnded), 3, 3)]
#[case::detach_at_open(Some(0), None, Ok(HostedCompletion::Detached), 0, 0)]
#[case::detach_after_chunk(Some(1), None, Ok(HostedCompletion::Detached), 1, 1)]
#[case::consumer_fails(None, Some(1), Err(TestError::Consumer), 1, 1)]
#[tokio::test]
async fn in_process_runner_follows_consumer_demand(
    observer: Arc<Observer>,
    #[case] detach_after: Option<usize>,
    #[case] fail_after: Option<usize>,
    #[case] expected: Result<HostedCompletion<String>, TestError>,
    #[case] expected_polls: usize,
    #[case] expected_delivered: usize,
) {
    let Streaming {
        polls,
        released,
        machine,
    } = streaming_call(vec![Ok(0), Ok(1), Ok(2)]);
    let consumer = Consumer {
        detach_after,
        fail_after,
        delivered: Mutex::new(Vec::new()),
    };
    let outcome = run_hosted(
        machine,
        Host {
            services: &Services { reject: false },
            interceptors: &(),
            stream: &consumer,
            observers: Some(&observer.0.sender),
        },
    )
    .await;
    assert_eq!(outcome, expected);
    assert_eq!(polls.load(Ordering::SeqCst), expected_polls);
    assert_eq!(
        *consumer.delivered.lock().unwrap(),
        (0..expected_delivered).collect::<Vec<_>>()
    );
    assert!(released.load(Ordering::SeqCst));
    let events = observer.0.lock().unwrap();
    assert_eq!(events.len(), 2);
    assert!(matches!(events[0], CallEvent::Started { .. }));
    match &expected {
        Ok(HostedCompletion::Detached) => {
            assert!(matches!(events[1], CallEvent::Cancelled { .. }))
        }
        Ok(_) => assert!(matches!(events[1], CallEvent::Succeeded { .. })),
        Err(_) => assert!(matches!(events[1], CallEvent::Failed { .. })),
    }
}

#[rstest]
#[case::at_open(0)]
#[case::after_chunk(1)]
#[tokio::test]
async fn consumer_failures_interrupt_the_machine(#[case] fail_after: usize) {
    let Streaming { polls, machine, .. } = streaming_call(vec![Ok(0), Ok(1), Ok(2)]);
    let interrupted = Arc::new(Mutex::new(Vec::new()));
    let consumer = Consumer {
        detach_after: None,
        fail_after: Some(fail_after),
        delivered: Mutex::new(Vec::new()),
    };
    let outcome = run(
        Interruptible {
            inner: machine,
            interrupted: interrupted.clone(),
        },
        Host {
            services: &Services { reject: false },
            interceptors: &(),
            stream: &consumer,
            observers: None,
        },
    )
    .await;
    assert_eq!(outcome, Err(TestError::Consumer));
    assert_eq!(
        *interrupted.lock().unwrap(),
        [HostFailure::Error(TestError::Consumer)]
    );
    assert_eq!(polls.load(Ordering::SeqCst), fail_after);
}

struct Recording {
    ops: &'static [&'static str],
    calls: AtomicUsize,
    seen: Mutex<Vec<String>>,
    fail: Option<&'static str>,
    events: Observations,
}

impl Recording {
    fn runtime(&self) -> Host<'_, Self, (), ()> {
        Host {
            services: self,
            interceptors: &(),
            stream: &(),
            observers: Some(&self.events.sender),
        }
    }
}

impl HostCallHandler<TestProtocol> for Recording {
    async fn handle_host_call(&self, reply: Reply<&'static str>) -> Result<(), TestError> {
        let op = self.ops[self.calls.fetch_add(1, Ordering::SeqCst)];
        self.seen.lock().unwrap().push(format!("op:{op}"));
        if self.fail == Some(op) {
            return Err(TestError::Service);
        }
        reply.send(op);
        Ok(())
    }
}

impl CallObserver for Recording {
    fn observe(&self, event: CallEvent) {
        self.seen.lock().unwrap().push(match event {
            CallEvent::Started { .. } => "started".into(),
            CallEvent::Succeeded { .. } => "succeeded".into(),
            CallEvent::Failed { .. } => "failed".into(),
            other => format!("{other:?}"),
        });
    }
}

fn scripted(
    ops: &'static [&'static str],
    outcome: Result<(), TestError>,
) -> CallMachine<TestProtocol, ()> {
    CallMachine::new(None, move |host| {
        Box::pin(async move {
            for op in ops {
                let answered = host.services.call(|reply| reply).await?;
                assert_eq!(answered, *op);
            }
            outcome
        })
    })
}

#[rstest]
#[case::succeeds(&["sign", "send"], Ok(()), None, Ok(()), &["started", "op:sign", "op:send", "succeeded"])]
#[case::call_fails(&[], Err(TestError::Provider), None, Err(TestError::Provider), &["started", "failed"])]
#[case::service_fails(&["sign", "send", "never"], Ok(()), Some("send"), Err(TestError::Service), &["started", "op:sign", "op:send", "failed"])]
#[tokio::test]
async fn generic_runner_forwards_ops_and_emits_one_terminal(
    #[case] ops: &'static [&'static str],
    #[case] call_outcome: Result<(), TestError>,
    #[case] fail: Option<&'static str>,
    #[case] expected: Result<(), TestError>,
    #[case] seen: &[&str],
) {
    let host = Recording {
        ops,
        calls: AtomicUsize::new(0),
        seen: Mutex::new(Vec::new()),
        fail,
        events: Observations::default(),
    };
    let outcome = run(scripted(ops, call_outcome), host.runtime()).await;
    assert_eq!(outcome, expected);
    assert_eq!(
        *host.seen.lock().unwrap(),
        seen.iter()
            .filter(|item| item.starts_with("op:"))
            .copied()
            .collect::<Vec<_>>()
    );
    let events = host.events.lock().unwrap();
    assert!(matches!(events.as_slice(), [CallEvent::Started { .. }, _]));
    assert_eq!(
        matches!(events[1], CallEvent::Failed { .. }),
        expected.is_err()
    );
    assert_eq!(
        matches!(events[1], CallEvent::Succeeded { .. }),
        expected.is_ok()
    );
}

#[rstest]
#[tokio::test]
async fn generic_runner_success_keeps_the_start_time(observer: Arc<Observer>) {
    let services = Recording {
        ops: &["send"],
        calls: AtomicUsize::new(0),
        seen: Mutex::new(Vec::new()),
        fail: None,
        events: Observations::default(),
    };
    let outcome = run(
        scripted(&["send"], Ok(())),
        Host {
            services: &services,
            interceptors: &(),
            stream: &(),
            observers: Some(&observer.0.sender),
        },
    )
    .await;
    assert_eq!(outcome, Ok(()));
    let events = observer.0.lock().unwrap();
    let [
        CallEvent::Started { start_time },
        CallEvent::Succeeded { timing, .. },
    ] = events.as_slice()
    else {
        panic!("unexpected events {events:?}");
    };
    assert_eq!(*start_time, timing.start_time);
}

#[rstest]
#[tokio::test]
async fn in_process_runner_dispatches_services_and_hooks(interceptors: Hooks) {
    let observer = interceptors.observer.clone();
    let completion = run_hosted(
        dispatching_call(),
        Host {
            services: &Services { reject: false },
            interceptors: &interceptors,
            stream: &(),
            observers: Some(&observer.0.sender),
        },
    )
    .await
    .unwrap();
    assert_eq!(
        completion,
        HostedCompletion::Complete("projected/custom".into())
    );
    assert!(matches!(
        observer.0.lock().unwrap().as_slice(),
        [
            CallEvent::Started { .. },
            CallEvent::Execution(ExecutionEvent::ProviderResponseReceived { .. }),
            CallEvent::Succeeded { .. },
        ]
    ));
}

struct ResponseGate {
    ready: tokio::sync::Notify,
    reject: bool,
}

impl Interceptors<TestError> for ResponseGate {
    async fn before_provider_request(
        &self,
        wire: WireRequest,
        _: RequestContext,
    ) -> Result<WireRequest, TestError> {
        Ok(wire)
    }

    async fn after_provider_response(&self, raw: RawResponse) -> Result<(), TestError> {
        assert_eq!(raw.body, "provider response");
        self.ready.notified().await;
        if self.reject {
            Err(TestError::Hook)
        } else {
            Ok(())
        }
    }
}

#[rstest]
#[case::accept(false)]
#[case::reject(true)]
#[tokio::test]
async fn response_interception_waits_and_can_reject_after_observation(#[case] reject: bool) {
    let (sender, mut receiver) =
        litellm_host::observation::observation_channel(std::num::NonZeroUsize::new(1).unwrap());
    let machine = hosted_call::<TestProtocol, _, _>(
        "input",
        Some(sender),
        |_, _, interceptors, observers| async move {
            let raw = RawResponse {
                body: "provider response".into(),
            };
            observers.unwrap().emit(CallEvent::Execution(
                ExecutionEvent::ProviderResponseReceived { raw: raw.clone() },
            ));
            interceptors.after_provider_response(raw).await?;
            Ok(CallOutput::Complete("accepted".into()))
        },
    );
    let interceptor = ResponseGate {
        ready: tokio::sync::Notify::new(),
        reject,
    };
    let mut driver = Driver::new(machine, Services { reject: false }, &interceptor);
    let mut advance = Box::pin(driver.advance());
    assert!(futures_util::poll!(&mut advance).is_pending());
    assert!(
        matches!(receiver.try_recv(), Ok(CallEvent::Execution(ExecutionEvent::ProviderResponseReceived { raw })) if raw.body == "provider response")
    );
    interceptor.ready.notify_one();
    match advance.await {
        Err(error) => {
            assert!(reject);
            assert_eq!(error, TestError::Hook);
        }
        Ok(Boundary::Complete(HostedCompletion::Complete(value))) => {
            assert!(!reject);
            assert_eq!(value, "accepted");
        }
        _ => panic!("expected completion or rejection"),
    }
}
