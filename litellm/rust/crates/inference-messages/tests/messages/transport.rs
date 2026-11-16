use std::collections::VecDeque;

use bytes::Bytes;
use futures_util::{StreamExt, TryStreamExt, future::BoxFuture};
use litellm_inference_messages::{
    Delivered, MessagesCallResponse, Transport, TransportRequest, TransportResponse,
};
use litellm_inference_testing::RecordingSecrets;
use rstest::rstest;

use super::*;

const INVALID_SIGNATURE: &str = "messages.1.content.0: Invalid `signature` in `thinking` block";
const UNREACHABLE_BASE: &str = "http://127.0.0.1:9";

#[derive(Debug, thiserror::Error)]
#[error("raised by the host client")]
struct Raised;

enum Scripted {
    Complete(Value),
    Stream(&'static [&'static str]),
    Rejected(u16, &'static str),
    Failed,
}

struct Sent {
    url: String,
    headers: Vec<(String, String)>,
    body: Value,
    raw_body: Bytes,
    stream: bool,
}

#[derive(Default)]
struct ScriptedTransport {
    script: Mutex<VecDeque<Scripted>>,
    sent: Mutex<Vec<Sent>>,
}

impl ScriptedTransport {
    fn new(script: impl IntoIterator<Item = Scripted>) -> Arc<Self> {
        Arc::new(Self {
            script: Mutex::new(script.into_iter().collect()),
            sent: Mutex::default(),
        })
    }

    fn sent(&self) -> std::sync::MutexGuard<'_, Vec<Sent>> {
        self.sent.lock().unwrap()
    }
}

impl Transport for ScriptedTransport {
    fn send(
        &self,
        request: TransportRequest,
    ) -> BoxFuture<'static, Result<TransportResponse, Error>> {
        self.sent.lock().unwrap().push(Sent {
            url: request.url,
            headers: request.headers,
            body: serde_json::from_slice(&request.body).expect("the host is handed JSON bytes"),
            raw_body: request.body,
            stream: request.stream,
        });
        let next = self
            .script
            .lock()
            .unwrap()
            .pop_front()
            .expect("a scripted answer");
        Box::pin(async move {
            let (status, delivered) = match next {
                Scripted::Complete(body) => (200, Delivered::Complete(body.to_string())),
                Scripted::Stream(frames) => (
                    200,
                    Delivered::Stream(
                        futures_util::stream::iter(
                            frames.iter().map(|frame| Ok(Bytes::from_static(frame.as_bytes()))),
                        )
                        .boxed(),
                    ),
                ),
                Scripted::Rejected(status, message) => (
                    status,
                    Delivered::Rejected {
                        text: json!({"type": "error", "error": {"type": "invalid_request_error", "message": message}})
                            .to_string(),
                        error: Error::transport(Raised),
                    },
                ),
                Scripted::Failed => return Err(Error::transport(Raised)),
            };
            Ok(TransportResponse {
                status,
                headers: vec![("request-id".into(), "req_1".into())],
                delivered,
            })
        })
    }
}

fn hosted_call(call: MessagesCall, fields: Value) -> MessagesCall {
    MessagesCall {
        api_key: Some("sk-ant".into()),
        api_base: Some(UNREACHABLE_BASE.into()),
        ..with_fields(call, fields)
    }
}

async fn execute(
    transport: Arc<ScriptedTransport>,
    call: MessagesCall,
) -> Result<(MessagesCallResponse, Vec<Bytes>), Error> {
    execute_recorded(transport, &RecordingCall::<Messages>::new(call)).await
}

async fn execute_recorded(
    transport: Arc<ScriptedTransport>,
    host: &RecordingCall<Messages>,
) -> Result<(MessagesCallResponse, Vec<Bytes>), Error> {
    let route = messages_route(Arc::new(RecordingSecrets::empty())).with_transport(transport);
    match route
        .execute(host.request()?, host, Some(host.events.0.sender.clone()))
        .await?
    {
        MessagesCallResponse::Stream { head, chunks } => Ok((
            MessagesCallResponse::Stream {
                head,
                chunks: Box::pin(futures_util::stream::empty()),
            },
            chunks.try_collect().await?,
        )),
        complete => Ok((complete, Vec::new())),
    }
}

#[rstest]
#[tokio::test]
async fn the_host_transport_sends_the_wire_request_instead_of_the_native_client(
    call: MessagesCall,
) {
    let transport = ScriptedTransport::new([Scripted::Complete(message_body())]);

    let (response, _) = execute(transport.clone(), hosted_call(call, json!({})))
        .await
        .expect("the host transport answers");

    let MessagesCallResponse::Complete(MessagesBody::Message(message)) = response else {
        panic!("a complete message");
    };
    assert_eq!(message.id, "msg_1");
    let sent = transport.sent();
    let [request] = sent.as_slice() else {
        panic!("one request");
    };
    assert_eq!(request.url, format!("{UNREACHABLE_BASE}/v1/messages"));
    assert!(!request.stream);
    assert_eq!(request.body["model"], MODEL);
    assert!(
        request
            .headers
            .iter()
            .any(|(name, value)| name == "x-api-key" && value == "sk-ant"),
        "{:?}",
        request.headers
    );
}

#[rstest]
#[tokio::test]
async fn a_host_stream_reaches_the_caller_chunk_by_chunk(call: MessagesCall) {
    let frames: &'static [&'static str] = &[
        "event: message_start\ndata: {\"type\":\"message_start\"}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ];
    let transport = ScriptedTransport::new([Scripted::Stream(frames)]);

    let (response, chunks) = execute(
        transport.clone(),
        hosted_call(call, json!({"stream": true})),
    )
    .await
    .expect("the host stream opens");

    let MessagesCallResponse::Stream { head, .. } = response else {
        panic!("a stream");
    };
    assert_eq!(
        head.headers,
        vec![("request-id".to_string(), "req_1".to_string())]
    );
    assert_eq!(
        chunks,
        frames
            .iter()
            .map(|frame| Bytes::from_static(frame.as_bytes()))
            .collect::<Vec<_>>()
    );
    assert!(transport.sent()[0].stream);
}

fn replayed_thinking() -> Value {
    json!({
        "max_tokens": 4096,
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "plan", "signature": "sig"},
                {"type": "text", "text": "hello"}
            ]},
            {"role": "user", "content": "again"}
        ]
    })
}

#[rstest]
#[tokio::test]
async fn a_recoverable_rejection_is_retried_through_the_host_transport(call: MessagesCall) {
    let transport = ScriptedTransport::new([
        Scripted::Rejected(400, INVALID_SIGNATURE),
        Scripted::Complete(message_body()),
    ]);

    execute(transport.clone(), hosted_call(call, replayed_thinking()))
        .await
        .expect("the retry succeeds");

    let sent = transport.sent();
    let [first, second] = sent.as_slice() else {
        panic!("two requests");
    };
    assert!(first.body.get("thinking").is_some());
    assert!(second.body.get("thinking").is_none());
}

#[rstest]
#[case::anthropic("https://api.anthropic.com", false)]
#[case::anthropic_compatible_host("https://api.deepseek.com/anthropic", true)]
#[tokio::test]
async fn unsigned_thinking_is_sent_only_to_a_host_that_does_not_check_claude_signatures(
    call: MessagesCall,
    #[case] api_base: &str,
    #[case] kept: bool,
) {
    let transport = ScriptedTransport::new([Scripted::Complete(message_body())]);
    let unsigned = json!({"type": "thinking", "thinking": "Gemini's plan", "signature": ""});
    let replayed = json!({
        "max_tokens": 4096,
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [unsigned.clone(), {"type": "text", "text": "hello"}]},
            {"role": "user", "content": "again"}
        ]
    });

    execute(
        transport.clone(),
        MessagesCall {
            api_base: Some(api_base.into()),
            ..hosted_call(call, replayed)
        },
    )
    .await
    .expect("the first call succeeds");

    let sent = transport.sent();
    let [request] = sent.as_slice() else {
        panic!("one request");
    };
    assert_eq!(request.url, format!("{api_base}/v1/messages"));
    assert_eq!(
        request.body["messages"][1]["content"]
            .as_array()
            .expect("the assistant turn is sent")
            .contains(&unsigned),
        kept
    );
    assert!(request.body.get("thinking").is_some());
}

#[rstest]
#[tokio::test]
async fn the_raw_head_carries_the_bytes_the_host_transport_last_sent(
    call: MessagesCall,
    #[values(false, true)] stream: bool,
) {
    let frames: &'static [&'static str] =
        &["event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"];
    let success = if stream {
        Scripted::Stream(frames)
    } else {
        Scripted::Complete(message_body())
    };
    let transport = ScriptedTransport::new([Scripted::Rejected(400, INVALID_SIGNATURE), success]);
    let fields = json!({"stream": stream, "temperature": 1e-7, "system": "caf\u{e9}"});
    let host =
        RecordingCall::<Messages>::new(hosted_call(with_fields(call, replayed_thinking()), fields));

    execute_recorded(transport.clone(), &host)
        .await
        .expect("the retry succeeds");

    let sent = transport.sent();
    let [_, retried] = sent.as_slice() else {
        panic!("two requests");
    };
    let [raw] = <[_; 1]>::try_from(host.events.0.raw_heads()).unwrap();
    let head = raw.head.expect("the head is forwarded");
    assert_eq!(head.sent_body, retried.raw_body);
    assert_eq!(head.streamed, stream);
}

#[rstest]
#[case::final_rejection(vec![Scripted::Rejected(429, "slow down")])]
#[case::spent_retry(vec![
    Scripted::Rejected(400, INVALID_SIGNATURE),
    Scripted::Rejected(400, INVALID_SIGNATURE),
])]
#[case::client_failure(vec![Scripted::Failed])]
#[tokio::test]
async fn the_caller_gets_the_error_the_host_transport_raised(
    call: MessagesCall,
    #[case] script: Vec<Scripted>,
) {
    let attempts = script.len();
    let transport = ScriptedTransport::new(script);

    let error = execute(transport.clone(), hosted_call(call, replayed_thinking()))
        .await
        .err()
        .expect("the call fails");

    assert!(litellm_error::find::<Raised>(&error).is_some(), "{error:?}");
    assert_eq!(error.upstream_response(), None);
    assert_eq!(transport.sent().len(), attempts);
}
