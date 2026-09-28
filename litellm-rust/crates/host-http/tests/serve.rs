use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use axum::{
    Json,
    body::to_bytes,
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures_util::{StreamExt, stream};
use http::{StatusCode, header::CONTENT_TYPE};
use litellm_host::{
    call::{CallOutput, hosted_call},
    event::{CallEvent, MachineEvent, RawResponse, RequestContext, WireRequest},
    hooks::RouteHooks,
    lifecycle::CallObserver,
    machine::MachineFault,
    protocol::Protocol,
    protocol::Reply,
};
use litellm_host_http::{Error, ResponseEncoder, StreamEncoder, Unary, serve, serve_unary};
use rstest::{fixture, rstest};
use serde_json::json;

#[derive(Clone, Debug, PartialEq)]
enum TestError {
    Provider,
    Adapter,
    Hook,
    Machine,
}

impl From<MachineFault> for TestError {
    fn from(_: MachineFault) -> Self {
        Self::Machine
    }
}

struct TestProtocol;

impl Protocol for TestProtocol {
    type Response = Bytes;
    type Error = TestError;
    type Request = &'static str;
    type HostCall = Reply<&'static str>;
    type Chunk = Bytes;
    type StreamHead = &'static str;
}

#[derive(Clone, Copy, PartialEq)]
enum Rejection {
    None,
    Head,
    Chunk,
    Custom,
    Complete,
}

struct Adapter(Rejection);

impl ResponseEncoder for Adapter {
    type Protocol = TestProtocol;

    fn encode_response(&self, value: Bytes) -> Result<Response, TestError> {
        if self.0 == Rejection::Complete {
            return Err(TestError::Adapter);
        }
        Ok((StatusCode::CREATED, [("x-converted", "yes")], value).into_response())
    }
}

impl litellm_host::services::HostCallHandler<TestProtocol> for Adapter {
    async fn handle_host_call(&self, reply: Reply<&'static str>) -> Result<(), TestError> {
        if self.0 == Rejection::Custom {
            return Err(TestError::Adapter);
        }
        reply.send("custom");
        Ok(())
    }
}

impl StreamEncoder for Adapter {
    fn encode_stream_head(
        &self,
        content_type: &'static str,
    ) -> Result<http::Response<()>, TestError> {
        if self.0 == Rejection::Head {
            return Err(TestError::Adapter);
        }
        Ok(http::Response::builder()
            .status(StatusCode::ACCEPTED)
            .header(CONTENT_TYPE, content_type)
            .body(())
            .unwrap())
    }

    fn encode_chunk(&self, chunk: Bytes) -> Result<Bytes, TestError> {
        if self.0 == Rejection::Chunk {
            return Err(TestError::Adapter);
        }
        Ok(chunk)
    }

    fn encode_stream_error(&self, error: Error<TestError>) -> Bytes {
        Bytes::from(format!("event: error\ndata: {error:?}\n\n"))
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
    fn observer(&self) -> Option<Arc<dyn CallObserver>> {
        Some(self.observer.clone())
    }

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

#[rstest]
#[tokio::test]
async fn projection_custom_operations_and_hooks_feed_the_http_response(hooks: Hooks) {
    let observer = hooks.observer.clone();
    let machine = hosted_call::<TestProtocol, _, _>(
        "projected",
        |request, services, route_hooks| async move {
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
            Ok(CallOutput::Complete(Bytes::from(wire.url)))
        },
    );
    let response = serve(
        machine,
        Adapter(Rejection::None),
        hooks,
        Adapter(Rejection::None),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);
    assert_eq!(response.headers()["x-converted"], "yes");
    assert_eq!(
        to_bytes(response.into_body(), 1024).await.unwrap(),
        "projected/custom"
    );
    let events = observer.0.lock().unwrap();
    assert!(matches!(events.as_slice(), [
        CallEvent::Started { .. },
        CallEvent::Machine(MachineEvent::ResponseReceived { raw }),
        CallEvent::Succeeded { .. },
    ] if raw.body == "projected/custom"));
}

struct Release(Arc<AtomicBool>);

impl Drop for Release {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[rstest]
#[case::consumed(None)]
#[case::dropped_at_open(Some(0))]
#[case::dropped_after_chunk(Some(1))]
#[case::dropped_before_eof(Some(2))]
#[tokio::test]
async fn body_demand_controls_polling_and_lifecycle(
    hooks: Hooks,
    #[case] drop_after: Option<usize>,
) {
    let observer = hooks.observer.clone();
    let polls = Arc::new(AtomicUsize::new(0));
    let released = Arc::new(AtomicBool::new(false));
    let provider_polls = polls.clone();
    let release = Release(released.clone());
    let machine = hosted_call::<TestProtocol, _, _>("input", move |_, _, _| async move {
        let chunks = stream::unfold((0, release), move |(index, release)| {
            provider_polls.fetch_add(1, Ordering::SeqCst);
            async move {
                (index < 2).then(|| (Ok(Bytes::from(index.to_string())), (index + 1, release)))
            }
        })
        .boxed();
        Ok(CallOutput::Stream {
            head: "text/event-stream",
            chunks,
        })
    });
    let response = serve(
        machine,
        Adapter(Rejection::None),
        hooks,
        Adapter(Rejection::None),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    assert_eq!(response.headers()[CONTENT_TYPE], "text/event-stream");
    assert_eq!(polls.load(Ordering::SeqCst), 0);
    assert!(matches!(
        observer.0.lock().unwrap().as_slice(),
        [CallEvent::Started { .. }]
    ));
    let mut body = response.into_body().into_data_stream();
    for index in 0..drop_after.unwrap_or(2) {
        assert_eq!(body.next().await.unwrap().unwrap(), index.to_string());
        assert_eq!(polls.load(Ordering::SeqCst), index + 1);
        assert_eq!(observer.0.lock().unwrap().len(), 1);
    }
    if drop_after.is_none() {
        assert!(body.next().await.is_none());
        assert_eq!(polls.load(Ordering::SeqCst), 3);
    }
    drop(body);
    assert!(released.load(Ordering::SeqCst));
    let events = observer.0.lock().unwrap();
    assert_eq!(events.len(), 2);
    assert_eq!(
        matches!(events[1], CallEvent::Cancelled { .. }),
        drop_after.is_some()
    );
    assert_eq!(
        matches!(events[1], CallEvent::Succeeded { .. }),
        drop_after.is_none()
    );
}

#[rstest]
#[case::provider(Rejection::None, TestError::Provider, 2)]
#[case::encoding(Rejection::Chunk, TestError::Adapter, 1)]
#[tokio::test]
async fn stream_failure_emits_one_error_frame_and_stops(
    hooks: Hooks,
    #[case] rejection: Rejection,
    #[case] expected: TestError,
    #[case] expected_polls: usize,
) {
    let observer = hooks.observer.clone();
    let polls = Arc::new(AtomicUsize::new(0));
    let provider_polls = polls.clone();
    let machine = hosted_call::<TestProtocol, _, _>("input", move |_, _, _| async move {
        let chunks = stream::iter([
            Ok(Bytes::from_static(b"first")),
            Err(TestError::Provider),
            Ok(Bytes::from_static(b"must not be delivered")),
        ])
        .inspect(move |_| {
            provider_polls.fetch_add(1, Ordering::SeqCst);
        })
        .boxed();
        Ok(CallOutput::Stream {
            head: "text/event-stream",
            chunks,
        })
    });
    let response = serve(machine, Adapter(rejection), hooks, Adapter(rejection))
        .await
        .unwrap();
    let body = to_bytes(response.into_body(), 1024).await.unwrap();
    let prefix = if rejection == Rejection::Chunk {
        ""
    } else {
        "first"
    };
    assert_eq!(
        body,
        format!(
            "{prefix}event: error\ndata: {:?}\n\n",
            Error::Call(expected)
        )
    );
    assert_eq!(polls.load(Ordering::SeqCst), expected_polls);
    assert!(matches!(
        observer.0.lock().unwrap().as_slice(),
        [CallEvent::Started { .. }, CallEvent::Failed { .. },]
    ));
}

#[rstest]
#[case::provider(Rejection::None)]
#[case::headers(Rejection::Head)]
#[case::custom_operation(Rejection::Custom)]
#[case::response_conversion(Rejection::Complete)]
#[tokio::test]
async fn failures_before_open_return_an_error(hooks: Hooks, #[case] rejection: Rejection) {
    let observer = hooks.observer.clone();
    let machine = hosted_call::<TestProtocol, _, _>("input", move |_, services, _| async move {
        services.call(|reply| reply).await?;
        match rejection {
            Rejection::Head => Ok(CallOutput::Stream {
                head: "text/event-stream",
                chunks: stream::pending().boxed(),
            }),
            Rejection::None => Err(TestError::Provider),
            _ => Ok(CallOutput::Complete(Bytes::new())),
        }
    });
    let expected = if rejection == Rejection::None {
        TestError::Provider
    } else {
        TestError::Adapter
    };
    assert_eq!(
        serve(machine, Adapter(rejection), hooks, Adapter(rejection))
            .await
            .unwrap_err(),
        Error::Call(expected)
    );
    assert!(matches!(
        observer.0.lock().unwrap().as_slice(),
        [CallEvent::Started { .. }, CallEvent::Failed { .. },]
    ));
}

#[rstest]
#[case::before_headers(false)]
#[case::awaiting_chunk(true)]
#[tokio::test]
async fn cancelling_pending_work_releases_the_machine(hooks: Hooks, #[case] streaming: bool) {
    let observer = hooks.observer.clone();
    let released = Arc::new(AtomicBool::new(false));
    let release = Release(released.clone());
    let machine = hosted_call::<TestProtocol, _, _>("input", move |_, _, _| async move {
        if !streaming {
            let _release = release;
            return std::future::pending().await;
        }
        let chunks = stream::once(async move {
            let _release = release;
            std::future::pending().await
        })
        .boxed();
        Ok(CallOutput::Stream {
            head: "text/event-stream",
            chunks,
        })
    });
    let mut response = Box::pin(serve(
        machine,
        Adapter(Rejection::None),
        hooks,
        Adapter(Rejection::None),
    ));
    if streaming {
        let mut body = response.await.unwrap().into_body().into_data_stream();
        assert!(futures_util::poll!(body.next()).is_pending());
        assert!(!released.load(Ordering::SeqCst));
        drop(body);
    } else {
        assert!(futures_util::poll!(&mut response).is_pending());
        assert!(!released.load(Ordering::SeqCst));
        drop(response);
    }
    assert!(released.load(Ordering::SeqCst));
    assert!(matches!(
        observer.0.lock().unwrap().as_slice(),
        [CallEvent::Started { .. }, CallEvent::Cancelled { .. },]
    ));
}

#[rstest]
#[case::before_provider_request(false)]
#[case::event(true)]
#[tokio::test]
async fn hook_rejection_stops_execution_and_is_reported_once(
    observer: Arc<Observer>,
    #[case] event: bool,
) {
    let continued = Arc::new(AtomicBool::new(false));
    let executed = continued.clone();
    let machine = hosted_call::<TestProtocol, _, _>("input", move |_, _, route_hooks| async move {
        if event {
            route_hooks
                .on_event(MachineEvent::ResponseReceived {
                    raw: RawResponse {
                        body: "response".into(),
                    },
                })
                .await?;
        } else {
            route_hooks
                .before_provider_request(
                    WireRequest {
                        url: "url".into(),
                        headers: Vec::new(),
                        body: json!({}),
                    },
                    RequestContext {
                        model: "model".into(),
                        custom_llm_provider: "provider".into(),
                        optional_params: json!({}),
                        secret_fields: Vec::new(),
                        api_key: None,
                    },
                )
                .await?;
        }
        executed.store(true, Ordering::SeqCst);
        Ok(CallOutput::Complete(Bytes::new()))
    });
    let hooks = Hooks {
        observer: observer.clone(),
        reject: true,
    };
    assert_eq!(
        serve(
            machine,
            Adapter(Rejection::None),
            hooks,
            Adapter(Rejection::None)
        )
        .await
        .unwrap_err(),
        Error::Call(TestError::Hook)
    );
    assert!(!continued.load(Ordering::SeqCst));
    assert!(matches!(
        observer.0.lock().unwrap().as_slice(),
        [CallEvent::Started { .. }, CallEvent::Failed { .. },]
    ));
}

#[derive(Clone, Copy)]
enum InvalidFlow {
    DeliverBeforeOpen,
    OpenTwice,
}

#[rstest]
#[case::deliver_before_open(InvalidFlow::DeliverBeforeOpen)]
#[case::open_twice(InvalidFlow::OpenTwice)]
#[tokio::test]
async fn invalid_host_operations_fail_without_panicking(hooks: Hooks, #[case] flow: InvalidFlow) {
    use litellm_host::{call::HostedCompletion, machine::CallMachine};

    let observer = hooks.observer.clone();
    let machine = CallMachine::<TestProtocol, HostedCompletion<Bytes>>::new(move |host| {
        Box::pin(async move {
            match flow {
                InvalidFlow::DeliverBeforeOpen => {
                    host.stream.send_chunk(Bytes::new()).await?;
                }
                InvalidFlow::OpenTwice => {
                    host.stream.open_stream("text/event-stream").await?;
                    host.stream.open_stream("text/event-stream").await?;
                }
            }
            Ok(HostedCompletion::StreamEnded)
        })
    });
    let result = serve(
        machine,
        Adapter(Rejection::None),
        hooks,
        Adapter(Rejection::None),
    )
    .await;
    if matches!(flow, InvalidFlow::OpenTwice) {
        let body = to_bytes(result.unwrap().into_body(), 1024).await.unwrap();
        assert_eq!(body, "event: error\ndata: Protocol\n\n");
    } else {
        assert_eq!(result.unwrap_err(), Error::Protocol);
    }
    assert!(matches!(
        observer.0.lock().unwrap().as_slice(),
        [CallEvent::Started { .. }, CallEvent::Failed { .. },]
    ));
}

struct UnaryProtocol;

impl Protocol for UnaryProtocol {
    type Response = serde_json::Value;
    type Error = TestError;
    type Request = &'static str;
    type HostCall = std::convert::Infallible;
    type Chunk = std::convert::Infallible;
    type StreamHead = std::convert::Infallible;
}

#[rstest]
#[tokio::test]
async fn unary_calls_use_into_response_after_hooks_and_before_success(hooks: Hooks) {
    let observer = hooks.observer.clone();
    let machine =
        hosted_call::<UnaryProtocol, _, _>("projected", |request, _, route_hooks| async move {
            let wire = route_hooks
                .before_provider_request(
                    WireRequest {
                        url: request.into(),
                        headers: Vec::new(),
                        body: json!({}),
                    },
                    RequestContext {
                        model: "rewritten".into(),
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
            Ok(CallOutput::Complete(json!({"url": wire.url})))
        });
    let response = serve_unary(
        machine,
        (),
        hooks,
        Unary::new(|value| {
            assert!(matches!(
                observer.0.lock().unwrap().as_slice(),
                [CallEvent::Started { .. }, CallEvent::Machine(_),]
            ));
            (StatusCode::CREATED, [("x-converted", "yes")], Json(value))
        }),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);
    assert_eq!(response.headers()["x-converted"], "yes");
    assert_eq!(response.headers()[CONTENT_TYPE], "application/json");
    let body: serde_json::Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 1024).await.unwrap()).unwrap();
    assert_eq!(body, json!({"url": "projected/rewritten"}));
    assert!(matches!(
        observer.0.lock().unwrap().as_slice(),
        [
            CallEvent::Started { .. },
            CallEvent::Machine(_),
            CallEvent::Succeeded { .. },
        ]
    ));
}

#[rstest]
#[case::provider(false, TestError::Provider)]
#[case::hook(true, TestError::Hook)]
#[tokio::test]
async fn unary_failure_preserves_the_error_without_converting(
    observer: Arc<Observer>,
    #[case] reject_hook: bool,
    #[case] expected: TestError,
) {
    let machine = hosted_call::<UnaryProtocol, _, _>("input", |_, _, route_hooks| async move {
        route_hooks
            .on_event(MachineEvent::ResponseReceived {
                raw: RawResponse { body: "raw".into() },
            })
            .await?;
        Err(TestError::Provider)
    });
    let hooks = Hooks {
        observer: observer.clone(),
        reject: reject_hook,
    };
    let converted = AtomicBool::new(false);
    let result = serve_unary(
        machine,
        (),
        hooks,
        Unary::new(|value| {
            converted.store(true, Ordering::SeqCst);
            Json(value)
        }),
    )
    .await;
    assert_eq!(result.unwrap_err(), Error::Call(expected));
    assert!(!converted.load(Ordering::SeqCst));
    let events = observer.0.lock().unwrap();
    assert!(matches!(events.first(), Some(CallEvent::Started { .. })));
    assert!(matches!(events.last(), Some(CallEvent::Failed { .. })));
    assert_eq!(events.len(), if reject_hook { 2 } else { 3 });
}

#[rstest]
#[tokio::test]
async fn cancelling_unary_execution_releases_work_without_converting(hooks: Hooks) {
    let observer = hooks.observer.clone();
    let released = Arc::new(AtomicBool::new(false));
    let release = Release(released.clone());
    let machine = hosted_call::<UnaryProtocol, _, _>("input", move |_, _, _| async move {
        let _release = release;
        std::future::pending().await
    });
    let converted = AtomicBool::new(false);
    let mut call = Box::pin(serve_unary(
        machine,
        (),
        hooks,
        Unary::new(|value| {
            converted.store(true, Ordering::SeqCst);
            Json(value)
        }),
    ));
    assert!(futures_util::poll!(&mut call).is_pending());
    assert!(!released.load(Ordering::SeqCst));
    drop(call);
    assert!(released.load(Ordering::SeqCst));
    assert!(!converted.load(Ordering::SeqCst));
    assert!(matches!(
        observer.0.lock().unwrap().as_slice(),
        [CallEvent::Started { .. }, CallEvent::Cancelled { .. },]
    ));
}

struct CustomUnaryProtocol;

impl Protocol for CustomUnaryProtocol {
    type Response = Bytes;
    type Error = TestError;
    type Request = &'static str;
    type HostCall = Reply<&'static str>;
    type Chunk = std::convert::Infallible;
    type StreamHead = std::convert::Infallible;
}

struct CustomUnaryAdapter {
    reject_response: bool,
    converted: Arc<AtomicBool>,
}

impl ResponseEncoder for CustomUnaryAdapter {
    type Protocol = CustomUnaryProtocol;

    fn encode_response(&self, response: Bytes) -> Result<Response, TestError> {
        self.converted.store(true, Ordering::SeqCst);
        if self.reject_response {
            return Err(TestError::Adapter);
        }
        Ok(response.into_response())
    }
}

struct Credentials(bool);

impl litellm_host::services::HostCallHandler<CustomUnaryProtocol> for Credentials {
    async fn handle_host_call(&self, reply: Reply<&'static str>) -> Result<(), TestError> {
        if self.0 {
            return Err(TestError::Adapter);
        }
        reply.send("credential");
        Ok(())
    }
}

#[rstest]
#[case::success(false, false)]
#[case::custom_operation_fails(true, false)]
#[case::response_conversion_fails(false, true)]
#[tokio::test]
async fn unary_custom_operations_and_conversion_finish_before_terminal_observation(
    hooks: Hooks,
    #[case] reject_op: bool,
    #[case] reject_response: bool,
) {
    let observer = hooks.observer.clone();
    let continued = Arc::new(AtomicBool::new(false));
    let executed = continued.clone();
    let released = Arc::new(AtomicBool::new(false));
    let release = Release(released.clone());
    let converted = Arc::new(AtomicBool::new(false));
    let machine = hosted_call::<CustomUnaryProtocol, _, _>(
        "request",
        move |request, services, _| async move {
            let _release = release;
            let credential = services.call(|reply| reply).await?;
            executed.store(true, Ordering::SeqCst);
            Ok(CallOutput::Complete(Bytes::from(format!(
                "{request}/{credential}"
            ))))
        },
    );
    let result = serve_unary(
        machine,
        Credentials(reject_op),
        hooks,
        CustomUnaryAdapter {
            reject_response,
            converted: converted.clone(),
        },
    )
    .await;
    assert_eq!(continued.load(Ordering::SeqCst), !reject_op);
    assert_eq!(converted.load(Ordering::SeqCst), !reject_op);
    assert!(released.load(Ordering::SeqCst));
    if reject_op || reject_response {
        assert_eq!(result.unwrap_err(), Error::Call(TestError::Adapter));
        assert!(matches!(
            observer.0.lock().unwrap().as_slice(),
            [CallEvent::Started { .. }, CallEvent::Failed { .. }]
        ));
    } else {
        let body = to_bytes(result.unwrap().into_body(), 1024).await.unwrap();
        assert_eq!(body, "request/credential");
        assert!(matches!(
            observer.0.lock().unwrap().as_slice(),
            [CallEvent::Started { .. }, CallEvent::Succeeded { .. }]
        ));
    }
}
