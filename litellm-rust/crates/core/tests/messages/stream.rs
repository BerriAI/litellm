use std::{convert::Infallible, sync::Mutex};

use bytes::Bytes;
use litellm_core::messages::route::{Messages, MessagesStreamHead};
use litellm_host::host::{Demand, Host};
use rstest::rstest;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

use super::*;

const UPSTREAM_HEADERS: [(&str, &str); 2] = [
    ("request-id", "req_upstream_123"),
    ("anthropic-ratelimit-requests-remaining", "41"),
];

const SSE_BODY: &str = "event: message_start\ndata: {\"type\":\"message_start\"}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";

enum Seen {
    Open(Vec<(String, String)>),
    Deliver(Bytes),
}

/// Projects like `LocalMessagesHost`, records every stream op in the order the route
/// performs it, and detaches after `detach_after` ops.
struct RecordingStreamHost {
    call: LocalMessagesHost,
    detach_after: usize,
    seen: Mutex<Vec<Seen>>,
}

impl RecordingStreamHost {
    fn new(call: MessagesCall, detach_after: usize) -> Self {
        Self {
            call: LocalMessagesHost::new(call),
            detach_after,
            seen: Mutex::new(Vec::new()),
        }
    }

    fn record(&self, op: Seen) -> Demand {
        let mut seen = self.seen.lock().unwrap();
        seen.push(op);
        match seen.len() < self.detach_after {
            true => Demand::More,
            false => Demand::Detached,
        }
    }
}

impl Host<Messages> for RecordingStreamHost {
    async fn project(&self) -> Result<MessagesCall, Error> {
        self.call.project().await
    }

    async fn custom_op(&self, op: Infallible) -> Result<(), Error> {
        match op {}
    }

    async fn open(&self, head: MessagesStreamHead) -> Result<Demand, Error> {
        Ok(self.record(Seen::Open(head.headers)))
    }

    async fn deliver(&self, chunk: Bytes) -> Result<Demand, Error> {
        Ok(self.record(Seen::Deliver(chunk)))
    }
}

fn streaming(call: MessagesCall, api_base: String) -> MessagesCall {
    MessagesCall {
        api_key: Some("sk-ant".into()),
        api_base: Some(api_base),
        ..with_fields(call, json!({"stream": true}))
    }
}

fn sse_response() -> ResponseTemplate {
    UPSTREAM_HEADERS.iter().fold(
        ResponseTemplate::new(200).set_body_raw(SSE_BODY, "text/event-stream"),
        |response, (name, value)| response.insert_header(*name, *value),
    )
}

async fn stream_through(host: &RecordingStreamHost) -> Result<MessagesOutput, Error> {
    litellm_host::run::run(machine(Arc::new(RecordingSecrets::empty())), host).await
}

#[rstest]
#[tokio::test]
async fn upstream_headers_are_on_the_stream_head_before_the_first_chunk(call: MessagesCall) {
    let upstream = upstream([sse_response()]).await;
    let host = RecordingStreamHost::new(streaming(call, upstream.uri()), usize::MAX);

    let outcome = stream_through(&host).await.expect("streamed call succeeds");

    assert!(matches!(outcome, MessagesOutput::Streamed));
    let seen = host.seen.into_inner().unwrap();
    let [Seen::Open(headers), chunks @ ..] = seen.as_slice() else {
        panic!("the stream opens before any chunk is delivered");
    };
    let surfaced: Vec<(&str, &str)> = headers
        .iter()
        .filter(|(name, _)| {
            UPSTREAM_HEADERS
                .iter()
                .any(|(upstream, _)| upstream == name)
        })
        .map(|(name, value)| (name.as_str(), value.as_str()))
        .collect();
    assert_eq!(surfaced, UPSTREAM_HEADERS);
    let delivered: Vec<u8> = chunks
        .iter()
        .flat_map(|step| match step {
            Seen::Deliver(chunk) => chunk.to_vec(),
            Seen::Open(_) => panic!("the stream opens exactly once"),
        })
        .collect();
    assert_eq!(delivered, SSE_BODY.as_bytes());
}

#[rstest]
#[case::at_open(1)]
#[case::after_the_first_chunk(2)]
#[tokio::test]
async fn a_detached_caller_receives_nothing_more(call: MessagesCall, #[case] detach_after: usize) {
    let upstream = upstream([sse_response()]).await;
    let host = RecordingStreamHost::new(streaming(call, upstream.uri()), detach_after);

    let outcome = stream_through(&host)
        .await
        .expect("a detached stream still completes");

    assert!(matches!(outcome, MessagesOutput::Streamed));
    assert_eq!(host.seen.into_inner().unwrap().len(), detach_after);
}

#[rstest]
#[case::text_body(ResponseTemplate::new(429).set_body_string("slow down"), "slow down")]
#[case::json_envelope(
    status_response(429, json!({"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}})),
    r#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#
)]
#[tokio::test]
async fn an_upstream_error_fails_the_call_without_opening_the_stream(
    call: MessagesCall,
    #[case] response: ResponseTemplate,
    #[case] body: &str,
) {
    let upstream = upstream([response]).await;
    let host = RecordingStreamHost::new(streaming(call, upstream.uri()), usize::MAX);

    let error = stream_through(&host)
        .await
        .err()
        .expect("upstream error propagates");

    assert_eq!(
        error,
        Error::Transport(litellm_http::transport::Error::Http {
            status: 429,
            body: body.into()
        })
    );
    assert!(host.seen.into_inner().unwrap().is_empty());
}

/// The native route relays bytes as they are. Python's synthetic `api_error` for a stream
/// that never reaches `message_stop` lives in its SSE wrapper, above this route.
#[rstest]
#[tokio::test]
async fn a_stream_that_ends_without_message_stop_is_relayed_as_is(call: MessagesCall) {
    const INCOMPLETE: &str = "event: message_start\ndata: {\"type\":\"message_start\"}\n\n";
    let upstream =
        upstream([ResponseTemplate::new(200).set_body_raw(INCOMPLETE, "text/event-stream")]).await;
    let host = RecordingStreamHost::new(streaming(call, upstream.uri()), usize::MAX);

    stream_through(&host).await.expect("streamed call succeeds");

    let delivered: Vec<u8> = host
        .seen
        .into_inner()
        .unwrap()
        .iter()
        .flat_map(|step| match step {
            Seen::Deliver(chunk) => chunk.to_vec(),
            Seen::Open(_) => Vec::new(),
        })
        .collect();
    assert_eq!(delivered, INCOMPLETE.as_bytes());
}

/// Serves one SSE chunk and then holds the connection open without ever finishing.
async fn stalling_upstream() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = vec![0; 4096];
        let _ = socket.read(&mut request).await;
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n\
                  1f\r\nevent: message_start\ndata: {}\n\n\r\n",
            )
            .await
            .unwrap();
        std::future::pending::<()>().await;
    });
    base
}

#[rstest]
#[tokio::test]
async fn the_timeout_covers_a_stalled_stream_body(call: MessagesCall) {
    let base = stalling_upstream().await;
    let host = RecordingStreamHost::new(
        MessagesCall {
            timeout: Some(Duration::from_millis(300)),
            ..streaming(call, base)
        },
        usize::MAX,
    );

    let error = tokio::time::timeout(Duration::from_secs(5), stream_through(&host))
        .await
        .expect("the stalled stream gives up within the timeout")
        .err()
        .expect("a stalled body fails the call");

    assert!(matches!(error, Error::Transport(_)), "{error:?}");
    let seen = host.seen.into_inner().unwrap();
    assert!(
        matches!(seen.as_slice(), [Seen::Open(_), Seen::Deliver(chunk)] if chunk.as_ref() == b"event: message_start\ndata: {}\n\n"),
        "the chunk before the stall reached the caller, saw {} ops",
        seen.len()
    );
}

#[rstest]
#[tokio::test]
async fn a_host_on_anthropic_sse_is_relayed_byte_for_byte(call: MessagesCall) {
    let upstream = upstream([sse_response()]).await;
    let host = RecordingStreamHost::new(
        MessagesCall {
            custom_llm_provider: Some("azure_ai".into()),
            ..streaming(call, upstream.uri())
        },
        usize::MAX,
    );

    let outcome = stream_through(&host).await.expect("azure streams");

    assert!(matches!(outcome, MessagesOutput::Streamed));
    let seen = host.seen.into_inner().unwrap();
    let delivered: Vec<u8> = seen
        .iter()
        .filter_map(|step| match step {
            Seen::Deliver(chunk) => Some(chunk.to_vec()),
            Seen::Open(_) => None,
        })
        .flatten()
        .collect();
    assert_eq!(delivered, SSE_BODY.as_bytes());
}
