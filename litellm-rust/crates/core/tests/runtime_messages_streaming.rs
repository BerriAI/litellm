use litellm_core::messages::types::ProviderMessagesRequest;
use std::future::{Ready, ready};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_util::{Stream, StreamExt, stream};
use litellm_core::Error;
use litellm_core::integrations::custom_logger::{LogError, LogFuture};
use litellm_core::lifecycle::{
    ActionResult, CallLifecycleContext, CallbackFuture, Clock, DeploymentFailureHooks,
    DeploymentPreHooks, DeploymentSuccessHooks, ModerationHooks, PreCallHooks,
    TerminalClassification, TerminalDispatcher, TerminalRecord,
};
use litellm_core::messages::lifecycle::Options;
use litellm_core::messages::types::MessagesRequest;
use litellm_core::providers::auth::Environment;
use litellm_core::runtime::{
    CallServices, HttpFuture, HttpRequest, HttpStreamFuture, HttpStreamResponse, HttpTransport,
    LiteLlm, MessagesRuntimeServices,
};
use serde_json::json;

macro_rules! messages_body {
    ($($json:tt)+) => {
        serde_json::from_value(json!($($json)+)).expect("messages body")
    };
}

#[derive(Default)]
struct Tracking {
    opened: AtomicUsize,
    session_drops: AtomicUsize,
    service_drops: AtomicUsize,
    terminals: Mutex<Vec<TerminalRecord>>,
    deployment_events: Mutex<Vec<&'static str>>,
    reject_deployment: bool,
    drain: Option<litellm_core::lifecycle::StreamDrainPolicy>,
}

impl Tracking {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
}

struct Session {
    tracking: Arc<Tracking>,
}

impl Drop for Session {
    fn drop(&mut self) {
        self.tracking.session_drops.fetch_add(1, Ordering::Relaxed);
    }
}

impl litellm_core::lifecycle::StreamDrain for Session {
    fn stream_drain_policy(&self) -> litellm_core::lifecycle::StreamDrainPolicy {
        self.tracking.drain.clone().unwrap_or_else(|| {
            litellm_core::lifecycle::StreamDrainPolicy::new(
                100,
                std::time::Duration::from_millis(50),
            )
        })
    }
}

impl Clock for Session {
    fn now(&self) -> f64 {
        10.0
    }
}

impl PreCallHooks<MessagesRequest> for Session {
    type PreCallFuture<'a> = Ready<ActionResult<MessagesRequest, Error>>;
    fn async_pre_call_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: MessagesRequest,
    ) -> Self::PreCallFuture<'a> {
        ready(ActionResult::Continue(request))
    }
}

impl ModerationHooks<ProviderMessagesRequest> for Session {
    type ModerationFuture<'a> = Ready<ActionResult<ProviderMessagesRequest, Error>>;

    fn async_moderation_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: ProviderMessagesRequest,
    ) -> Self::ModerationFuture<'a> {
        ready(ActionResult::Continue(request))
    }
}

impl DeploymentPreHooks<MessagesRequest> for Session {
    fn async_pre_call_deployment_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: MessagesRequest,
    ) -> CallbackFuture<'a, ActionResult<MessagesRequest, Error>>
    where
        MessagesRequest: 'a,
    {
        Box::pin(async move {
            self.tracking.deployment_events.lock().unwrap().push("pre");
            if self.tracking.reject_deployment {
                return ActionResult::Reject(Error::InvalidRequest("deployment rejected".into()));
            }
            ActionResult::Replace(MessagesRequest {
                body: litellm_core::messages::types::AnthropicMessagesRequest {
                    max_tokens: Some(17),
                    ..request.body
                },
                ..request
            })
        })
    }
}
impl DeploymentSuccessHooks<litellm_core::messages::types::AnthropicMessagesResponse> for Session {}
impl DeploymentFailureHooks for Session {
    fn async_post_call_failure_deployment_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        _: &'a Error,
    ) -> CallbackFuture<'a, Result<(), Error>> {
        Box::pin(async move {
            self.tracking
                .deployment_events
                .lock()
                .unwrap()
                .push("failure");
            Err(Error::InvalidRequest("deployment observer failed".into()))
        })
    }
}

impl TerminalDispatcher for Session {
    fn dispatch<'a>(&'a self, terminal: &'a TerminalRecord) -> LogFuture<'a> {
        Box::pin(async move {
            self.tracking
                .deployment_events
                .lock()
                .unwrap()
                .push("terminal");
            self.tracking
                .terminals
                .lock()
                .unwrap()
                .push(terminal.clone());
            Ok::<(), LogError>(())
        })
    }
}

#[derive(Clone)]
struct Calls {
    tracking: Arc<Tracking>,
}

impl CallServices for Calls {
    type Bindings = u64;
    type Session = Session;
    type OpenFuture<'a> = Ready<Result<Session, Error>>;

    fn open<'a>(&'a self, _: CallLifecycleContext, _: Self::Bindings) -> Self::OpenFuture<'a> {
        self.tracking.opened.fetch_add(1, Ordering::Relaxed);
        ready(Ok(Session {
            tracking: self.tracking.clone(),
        }))
    }
}

struct Transport {
    response: Mutex<Option<Result<HttpStreamResponse, Error>>>,
    requests: Mutex<Vec<HttpRequest>>,
}

impl HttpTransport for Transport {
    fn execute(&self, _: HttpRequest) -> HttpFuture<'_> {
        Box::pin(ready(Err(Error::Unsupported("buffered test transport"))))
    }

    fn execute_stream(&self, request: HttpRequest) -> HttpStreamFuture<'_> {
        self.requests.lock().unwrap().push(request);
        let response = self.response.lock().unwrap().take().unwrap();
        Box::pin(ready(response))
    }
}

struct Services {
    transport: Transport,
    calls: Calls,
    tracking: Arc<Tracking>,
}

impl Drop for Services {
    fn drop(&mut self) {
        self.tracking.service_drops.fetch_add(1, Ordering::Relaxed);
    }
}

impl Environment for Services {
    fn environment(&self, _: &str) -> Option<String> {
        None
    }
}

impl MessagesRuntimeServices for Services {
    type Transport = Transport;
    type Calls = Calls;

    fn transport(&self) -> &Self::Transport {
        &self.transport
    }

    fn calls(&self) -> &Self::Calls {
        &self.calls
    }
}

fn request() -> MessagesRequest {
    MessagesRequest {
        model: "anthropic/claude-test".to_string(),
        body: messages_body!({
            "model": "claude-test",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true
        }),
        api_key: Some("test-key".to_string()),
        api_base: Some("https://recording.invalid/v1/messages".to_string()),
        custom_llm_provider: None,
        extra_headers: None,
        timeout: None,
    }
}

fn context() -> CallLifecycleContext {
    CallLifecycleContext::new("messages", "claude-test", "anthropic", "call-1")
}

fn client(
    response: Result<HttpStreamResponse, Error>,
    tracking: Arc<Tracking>,
) -> LiteLlm<Services> {
    LiteLlm::from_services(Services {
        transport: Transport {
            response: Mutex::new(Some(response)),
            requests: Mutex::new(Vec::new()),
        },
        calls: Calls {
            tracking: tracking.clone(),
        },
        tracking,
    })
}

fn response(stream: litellm_core::lifecycle::BytesStream) -> HttpStreamResponse {
    HttpStreamResponse {
        status: 200,
        content_type: Some("text/event-stream".to_string()),
        cache_control: Some("no-cache".to_string()),
        stream,
    }
}

#[rstest::rstest]
#[case::network("network")]
#[case::truncated("truncated")]
#[case::cancelled("cancelled")]
#[tokio::test]
async fn established_stream_failure_only_dispatches_terminal(#[case] ending: &str) {
    let tracking = Tracking::new();
    let source: litellm_core::lifecycle::BytesStream = match ending {
        "network" => Box::pin(stream::iter([Err(Error::Network("stream failed".into()))])),
        "truncated" => Box::pin(stream::empty()),
        "cancelled" => Box::pin(stream::pending()),
        _ => unreachable!(),
    };
    let client = client(Ok(response(source)), tracking.clone());
    let call = client
        .messages_stream_with(
            request(),
            Options {
                asynchronous: true,
                ..Options::default()
            },
            context(),
            1,
        )
        .await
        .unwrap();
    let completion = call.completion.register();
    if ending == "cancelled" {
        drop(call.stream);
    } else {
        assert!(
            call.stream
                .collect::<Vec<_>>()
                .await
                .iter()
                .any(Result::is_err)
        );
    }
    let terminal = completion.await.unwrap();
    let expected = match ending {
        "network" => "NetworkError",
        "truncated" => "InvalidResponse",
        "cancelled" => "NetworkError",
        _ => unreachable!(),
    };
    assert!(
        matches!(terminal.classification, TerminalClassification::Failure { kind, .. } if kind == expected)
    );
    assert_eq!(tracking.terminals.lock().unwrap().len(), 1);
    assert_eq!(
        *tracking.deployment_events.lock().unwrap(),
        ["pre", "terminal"]
    );
}

#[tokio::test]
async fn deployment_pre_replacement_reaches_stream_transport() {
    let tracking = Tracking::new();
    let client = client(Ok(response(Box::pin(stream::pending()))), tracking.clone());
    let call = client
        .messages_stream_with(
            request(),
            Options {
                asynchronous: true,
                ..Options::default()
            },
            context(),
            1,
        )
        .await
        .unwrap();
    let completion = call.completion.register();
    drop(call.stream);
    completion.await.unwrap();

    let requests = client.services().transport.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["max_tokens"], 17);
    assert_eq!(
        tracking
            .deployment_events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| **event == "pre")
            .count(),
        1
    );
}

#[tokio::test]
async fn deployment_pre_rejection_prevents_stream_transport() {
    let tracking = Arc::new(Tracking {
        reject_deployment: true,
        ..Tracking::default()
    });
    let client = client(Ok(response(Box::pin(stream::pending()))), tracking.clone());
    let result = client
        .messages_stream_with(
            request(),
            Options {
                asynchronous: true,
                ..Options::default()
            },
            context(),
            1,
        )
        .await;
    if let Ok(call) = result {
        let completion = call.completion.register();
        drop(call.stream);
        completion.await.unwrap();
        panic!("deployment rejection must prevent opening a provider stream");
    }
    assert!(
        matches!(result, Err(Error::InvalidRequest(message)) if message == "deployment rejected")
    );
    assert!(
        client
            .services()
            .transport
            .requests
            .lock()
            .unwrap()
            .is_empty()
    );
    assert_eq!(tracking.terminals.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn provider_open_failure_notifies_deployment_before_terminal_and_preserves_error() {
    let tracking = Tracking::new();
    let client = client(
        Err(Error::Network("provider unavailable".into())),
        tracking.clone(),
    );
    let result = client
        .messages_stream_with(
            request(),
            Options {
                asynchronous: true,
                ..Options::default()
            },
            context(),
            1,
        )
        .await;

    assert!(matches!(result, Err(Error::Network(message)) if message == "provider unavailable"));
    assert_eq!(
        client.services().transport.requests.lock().unwrap().len(),
        1
    );
    let events = tracking.deployment_events.lock().unwrap();
    assert_eq!(
        events
            .iter()
            .copied()
            .filter(|event| *event != "pre")
            .collect::<Vec<_>>(),
        ["failure", "terminal"]
    );
}

#[tokio::test]
async fn completion_retains_application_and_session_until_terminal_delivery() {
    let tracking = Tracking::new();
    let events = b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":4}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    let client = client(
        Ok(response(Box::pin(stream::iter([Ok(Bytes::from_static(
            events,
        ))])))),
        tracking.clone(),
    );
    let call = client
        .messages_stream_with(
            request(),
            Options {
                asynchronous: true,
                ..Options::default()
            },
            context(),
            1,
        )
        .await
        .unwrap();

    assert_eq!(tracking.opened.load(Ordering::Relaxed), 1);
    drop(client);
    assert_eq!(tracking.service_drops.load(Ordering::Relaxed), 0);
    assert_eq!(tracking.session_drops.load(Ordering::Relaxed), 0);

    let completion = call.completion.register();
    let bytes = call
        .stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .concat();
    assert_eq!(bytes, events);
    let terminal = completion.await.unwrap();

    assert_eq!(terminal.classification, TerminalClassification::Success);
    assert_eq!(terminal.usage.prompt_tokens, 5);
    assert_eq!(terminal.usage.completion_tokens, 4);
    assert_eq!(terminal.usage.total_tokens, 9);
    assert_eq!(tracking.terminals.lock().unwrap().len(), 1);
    assert_eq!(tracking.session_drops.load(Ordering::Relaxed), 1);
    assert_eq!(tracking.service_drops.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn unregistered_completion_still_delivers_terminal_before_cleanup() {
    let tracking = Tracking::new();
    let client = client(
        Ok(response(Box::pin(stream::iter([Ok(Bytes::from_static(
            b"data: {\"type\":\"message_start\",\"message\":{\"content\":[]}}\n\ndata: {\"type\":\"message_stop\"}\n\n",
        ))])))),
        tracking.clone(),
    );
    let call = client
        .messages_stream_with(
            request(),
            Options {
                asynchronous: true,
                ..Options::default()
            },
            context(),
            5,
        )
        .await
        .unwrap();
    drop(client);
    let stream = call.stream;
    drop(call.completion);
    assert!(stream.collect::<Vec<_>>().await.iter().all(Result::is_ok));

    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while tracking.terminals.lock().unwrap().is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();

    assert_eq!(tracking.terminals.lock().unwrap().len(), 1);
    assert_eq!(tracking.session_drops.load(Ordering::Relaxed), 1);
    assert_eq!(tracking.service_drops.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn partial_consumption_and_early_drop_drain_once_and_release_owners() {
    let tracking = Tracking::new();
    let client = client(
        Ok(response(Box::pin(stream::iter([
            Ok(Bytes::from_static(b"data: {\"type\":\"ping\"}\n\n")),
            Ok(Bytes::from_static(b"data: {\"type\":\"message_start\",\"message\":{\"content\":[]}}\n\ndata: {\"type\":\"message_stop\"}\n\n")),
        ])))),
        tracking.clone(),
    );
    let call = client
        .messages_stream_with(
            request(),
            Options {
                asynchronous: true,
                ..Options::default()
            },
            context(),
            2,
        )
        .await
        .unwrap();
    drop(client);
    let completion = call.completion.register();
    let mut stream = call.stream;

    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        "data: {\"type\":\"ping\"}\n\n"
    );
    drop(stream);
    let terminal = completion.await.unwrap();

    assert_eq!(terminal.classification, TerminalClassification::Success);
    assert_eq!(tracking.terminals.lock().unwrap().len(), 1);
    assert_eq!(tracking.session_drops.load(Ordering::Relaxed), 1);
    assert_eq!(tracking.service_drops.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn provider_failure_dispatches_before_releasing_the_session() {
    let tracking = Tracking::new();
    let client = client(
        Ok(HttpStreamResponse {
            status: 500,
            content_type: Some("application/json".to_string()),
            cache_control: None,
            stream: Box::pin(stream::iter([Ok(Bytes::from_static(b"failed"))])),
        }),
        tracking.clone(),
    );
    let result = client
        .messages_stream_with(
            request(),
            Options {
                asynchronous: true,
                ..Options::default()
            },
            context(),
            3,
        )
        .await;
    let error = match result {
        Ok(_) => panic!("provider rejects the stream"),
        Err(error) => error,
    };

    assert!(matches!(error, Error::Http { status: 500, .. }));
    assert_eq!(tracking.terminals.lock().unwrap().len(), 1);
    assert_eq!(tracking.session_drops.load(Ordering::Relaxed), 1);
    drop(client);
    assert_eq!(tracking.service_drops.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn cancelled_consumer_releases_stream_owners() {
    let tracking = Tracking::new();
    let client = client(Ok(response(Box::pin(stream::pending()))), tracking.clone());
    let call = client
        .messages_stream_with(
            request(),
            Options {
                asynchronous: true,
                ..Options::default()
            },
            context(),
            4,
        )
        .await
        .unwrap();
    drop(client);
    let completion = call.completion.register();
    let consumer = tokio::spawn(async move {
        let mut stream = call.stream;
        stream.next().await
    });
    tokio::task::yield_now().await;
    consumer.abort();
    assert!(consumer.await.unwrap_err().is_cancelled());
    let terminal = completion.await.unwrap();

    assert!(matches!(
        terminal.classification,
        TerminalClassification::Failure { ref kind, .. } if kind == "NetworkError"
    ));
    assert_eq!(tracking.terminals.lock().unwrap().len(), 1);
    assert_eq!(tracking.session_drops.load(Ordering::Relaxed), 1);
    assert_eq!(tracking.service_drops.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn stream_failure_dispatches_once_and_releases_owners() {
    let tracking = Tracking::new();
    let client = client(
        Ok(response(Box::pin(stream::iter([Err(Error::Network(
            "stream failed".to_string(),
        ))])))),
        tracking.clone(),
    );
    let call = client
        .messages_stream_with(
            request(),
            Options {
                asynchronous: true,
                ..Options::default()
            },
            context(),
            6,
        )
        .await
        .unwrap();
    drop(client);
    let completion = call.completion.register();
    let chunks = call.stream.collect::<Vec<_>>().await;
    let terminal = completion.await.unwrap();

    assert!(chunks.iter().any(Result::is_err));
    assert!(matches!(
        terminal.classification,
        TerminalClassification::Failure { ref kind, .. } if kind == "NetworkError"
    ));
    assert_eq!(tracking.terminals.lock().unwrap().len(), 1);
    assert_eq!(tracking.session_drops.load(Ordering::Relaxed), 1);
    assert_eq!(tracking.service_drops.load(Ordering::Relaxed), 1);
}

struct TrackedStream {
    inner: litellm_core::lifecycle::BytesStream,
    polls: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
}

impl Stream for TrackedStream {
    type Item = Result<Bytes, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.polls.fetch_add(1, Ordering::Relaxed);
        self.inner.as_mut().poll_next(cx)
    }
}

impl Drop for TrackedStream {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

#[tokio::test]
async fn sse_preserves_raw_chunks_without_read_ahead_and_releases_upstream_at_stop() {
    let tracking = Tracking::new();
    let polls = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let start = Bytes::from(String::from(
        "data: {\"type\":\"message_start\",\"message\":{\"content\":[]}}\n\n",
    ));
    let chunks = vec![
        start.slice(..10),
        Bytes::new(),
        start.slice(10..),
        Bytes::from(String::from("data: {\"type\":\"message_stop\"}\n\n")),
    ];
    let source = TrackedStream {
        inner: Box::pin(stream::iter(
            chunks
                .clone()
                .into_iter()
                .map(Ok)
                .chain([Err(Error::Network("must not be polled".into()))]),
        )),
        polls: polls.clone(),
        drops: drops.clone(),
    };
    let client = client(Ok(response(Box::pin(source))), tracking.clone());
    let call = client
        .messages_stream_with(
            request(),
            Options {
                asynchronous: true,
                ..Options::default()
            },
            context(),
            7,
        )
        .await
        .unwrap();
    let completion = call.completion.register();
    let mut stream = call.stream;
    assert_eq!(polls.load(Ordering::Relaxed), 0);
    for (index, expected) in chunks.iter().enumerate() {
        let received = stream.next().await.unwrap().unwrap();
        assert_eq!(&received, expected);
        assert_eq!(received.as_ptr(), expected.as_ptr());
        assert_eq!(polls.load(Ordering::Relaxed), index + 1);
    }
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    assert!(stream.next().await.is_none());
    assert_eq!(polls.load(Ordering::Relaxed), chunks.len());
    assert_eq!(
        completion.await.unwrap().classification,
        TerminalClassification::Success
    );
    drop(stream);
    assert_eq!(tracking.terminals.lock().unwrap().len(), 1);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn sse_observation_failures_dispatch_once_and_release_upstream() {
    for (tail, truncated) in [
        (b"data: not json\n\n".as_slice(), false),
        (b"data: \xff\n\n", false),
        (
            b"data: {\"type\":\"error\",\"error\":\"overloaded\"}\n\n",
            false,
        ),
        (b"data: {\"type\":\"message_stop\"}\n", true),
    ] {
        let tracking = Tracking::new();
        let polls = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let source = TrackedStream {
            inner: Box::pin(stream::iter([
                Ok(Bytes::from_static(
                    b"data: {\"type\":\"message_start\",\"message\":{\"content\":[]}}\n\n",
                )),
                Ok(Bytes::copy_from_slice(tail)),
            ])),
            polls: polls.clone(),
            drops: drops.clone(),
        };
        let client = client(Ok(response(Box::pin(source))), tracking.clone());
        let call = client
            .messages_stream_with(
                request(),
                Options {
                    asynchronous: true,
                    ..Options::default()
                },
                context(),
                8,
            )
            .await
            .unwrap();
        let completion = call.completion.register();
        let mut stream = call.stream;
        stream.next().await.unwrap().unwrap();
        if truncated {
            assert_eq!(stream.next().await.unwrap().unwrap().as_ref(), tail);
        }
        assert!(matches!(
            stream.next().await,
            Some(Err(Error::InvalidResponse(_)))
        ));
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        let count = polls.load(Ordering::Relaxed);
        assert!(stream.next().await.is_none());
        assert_eq!(polls.load(Ordering::Relaxed), count);
        assert!(matches!(completion.await.unwrap().classification,
            TerminalClassification::Failure { kind, .. } if kind == "InvalidResponse"));
        drop(stream);
        assert_eq!(tracking.terminals.lock().unwrap().len(), 1);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }
}

#[rstest::rstest]
#[case::success(false)]
#[case::failure(true)]
#[tokio::test]
async fn detached_stream_retains_owners_until_gated_upstream_finishes(#[case] fail: bool) {
    let tracking = Tracking::new();
    let (release, gate) = tokio::sync::oneshot::channel::<()>();
    let source = stream::iter([Ok(Bytes::from_static(
        b"data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5}}}\n\n",
    ))]).chain(stream::once(async move {
        gate.await.unwrap();
        if fail {
            Err(Error::Network("original upstream error".into()))
        } else {
            Ok(Bytes::from_static(b"data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":7}}\n\ndata: {\"type\":\"message_stop\"}\n\n"))
        }
    }));
    let client = client(Ok(response(Box::pin(source))), tracking.clone());
    let call = client
        .messages_stream_with(
            request(),
            Options {
                asynchronous: true,
                ..Options::default()
            },
            context(),
            9,
        )
        .await
        .unwrap();
    let completion = call.completion.register();
    let mut stream = call.stream;
    stream.next().await.unwrap().unwrap();
    drop(client);
    drop(stream);
    tokio::task::yield_now().await;
    assert!(tracking.terminals.lock().unwrap().is_empty());
    assert_eq!(tracking.session_drops.load(Ordering::Relaxed), 0);
    assert_eq!(tracking.service_drops.load(Ordering::Relaxed), 0);
    release.send(()).unwrap();
    let terminal = tokio::time::timeout(std::time::Duration::from_secs(1), completion)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(terminal.usage.prompt_tokens, 5);
    if fail {
        assert!(
            matches!(terminal.classification, TerminalClassification::Failure { message, .. } if message.contains("original upstream error"))
        );
    } else {
        assert_eq!(terminal.classification, TerminalClassification::Success);
        assert_eq!(terminal.usage.completion_tokens, 7);
        assert_eq!(terminal.usage.total_tokens, 12);
    }
    assert_eq!(
        *tracking.deployment_events.lock().unwrap(),
        ["pre", "terminal"]
    );
    assert_eq!(tracking.terminals.lock().unwrap().len(), 1);
    assert_eq!(tracking.session_drops.load(Ordering::Relaxed), 1);
    assert_eq!(tracking.service_drops.load(Ordering::Relaxed), 1);
}

#[rstest::rstest]
#[case::capacity_exhausted(0, false)]
#[case::timeout(1, true)]
#[tokio::test]
async fn detached_drain_limits_release_upstream_with_partial_usage(
    #[case] capacity: usize,
    #[case] timeout: bool,
) {
    let tracking = Arc::new(Tracking {
        drain: Some(litellm_core::lifecycle::StreamDrainPolicy::new(
            capacity,
            std::time::Duration::from_millis(10),
        )),
        ..Tracking::default()
    });
    let drops = Arc::new(AtomicUsize::new(0));
    let source = TrackedStream {
        inner: Box::pin(stream::iter([Ok(Bytes::from_static(
            b"data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5}}}\n\n",
        ))]).chain(stream::pending())),
        polls: Arc::new(AtomicUsize::new(0)),
        drops: drops.clone(),
    };
    let client = client(Ok(response(Box::pin(source))), tracking.clone());
    let call = client
        .messages_stream_with(
            request(),
            Options {
                asynchronous: true,
                ..Options::default()
            },
            context(),
            10,
        )
        .await
        .unwrap();
    let completion = call.completion.register();
    let mut stream = call.stream;
    stream.next().await.unwrap().unwrap();
    drop(stream);
    drop(client);
    let terminal = tokio::time::timeout(std::time::Duration::from_secs(1), completion)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(terminal.usage.prompt_tokens, 5);
    if timeout {
        assert!(
            matches!(terminal.classification, TerminalClassification::Failure { message, .. } if message.contains("timed out"))
        );
    } else {
        assert!(matches!(
            terminal.classification,
            TerminalClassification::Incomplete { .. }
        ));
    }
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    assert_eq!(
        *tracking.deployment_events.lock().unwrap(),
        ["pre", "terminal"]
    );
    assert_eq!(tracking.session_drops.load(Ordering::Relaxed), 1);
    assert_eq!(tracking.service_drops.load(Ordering::Relaxed), 1);
}
