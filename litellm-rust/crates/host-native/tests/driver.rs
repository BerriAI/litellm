use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use futures_util::{StreamExt, stream};
use litellm_host::{
    call::{CallOutput, HostedCompletion, hosted_call},
    event::{CallEvent, MachineEvent, RawResponse, RequestContext, WireRequest},
    hooks::RouteHooks,
    lifecycle::CallObserver,
    machine::MachineFault,
    protocol::{Demand, Protocol, Reply},
    services::HostCallHandler,
};
use litellm_host_native::{
    Boundary, Driver,
    in_process::{Host, StreamConsumer, run_hosted},
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
struct Observer(Mutex<Vec<CallEvent>>);

impl CallObserver for Observer {
    fn observe(&self, event: CallEvent) {
        self.0.lock().unwrap().push(event);
    }
}

struct Hooks {
    observer: Arc<Observer>,
    reject: bool,
}

impl RouteHooks<TestError> for Hooks {
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

    async fn on_event(&self, event: MachineEvent) -> Result<(), TestError> {
        if self.reject {
            return Err(TestError::Hook);
        }
        self.observer.observe(CallEvent::Machine(event));
        Ok(())
    }
}

#[fixture]
fn observer() -> Arc<Observer> {
    Arc::new(Observer::default())
}

#[fixture]
fn hooks(observer: Arc<Observer>) -> Hooks {
    Hooks {
        observer,
        reject: false,
    }
}

fn dispatching_call() -> TestMachine {
    hosted_call::<TestProtocol, _, _>("projected", |request, services, route_hooks| async move {
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
            .on_event(MachineEvent::ResponseReceived {
                raw: RawResponse {
                    body: wire.url.clone(),
                },
            })
            .await?;
        Ok(CallOutput::Complete(wire.url))
    })
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
    let machine = hosted_call::<TestProtocol, _, _>("input", move |_, _, _| async move {
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
    });
    Streaming {
        polls,
        released,
        machine,
    }
}

#[rstest]
#[tokio::test]
async fn services_and_hooks_answer_the_machine_inline(hooks: Hooks) {
    let observer = hooks.observer.clone();
    let mut driver = Driver::new(dispatching_call(), Services { reject: false }, hooks);
    let Boundary::Complete(HostedCompletion::Complete(response)) = driver.advance().await.unwrap()
    else {
        panic!("a unary call completes at the first boundary")
    };
    assert_eq!(response, "projected/custom");
    assert!(matches!(
        observer.0.lock().unwrap().as_slice(),
        [CallEvent::Machine(MachineEvent::ResponseReceived { raw })] if raw.body == "projected/custom"
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

struct Consumer {
    detach_after: Option<usize>,
    fail_after: Option<usize>,
    delivered: Mutex<Vec<usize>>,
}

impl Consumer {
    fn demand_after(&self, delivered: usize) -> Result<Demand, TestError> {
        if self.fail_after == Some(delivered) {
            return Err(TestError::Consumer);
        }
        Ok(if self.detach_after == Some(delivered) {
            Demand::Detached
        } else {
            Demand::More
        })
    }
}

impl StreamConsumer<TestProtocol> for Consumer {
    async fn open_stream(&self, head: &'static str) -> Result<Demand, TestError> {
        assert_eq!(head, "headers");
        self.demand_after(0)
    }

    async fn send_chunk(&self, chunk: usize) -> Result<Demand, TestError> {
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
            hooks: &(),
            stream: &consumer,
            observer: Some(observer.as_ref()),
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
#[tokio::test]
async fn in_process_runner_dispatches_services_and_hooks(hooks: Hooks) {
    let observer = hooks.observer.clone();
    let completion = run_hosted(
        dispatching_call(),
        Host {
            services: &Services { reject: false },
            hooks: &hooks,
            stream: &(),
            observer: Some(observer.as_ref()),
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
            CallEvent::Machine(MachineEvent::ResponseReceived { .. }),
            CallEvent::Succeeded { .. },
        ]
    ));
}
