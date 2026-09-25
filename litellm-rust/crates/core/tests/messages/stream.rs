use std::{convert::Infallible, sync::Mutex};

use bytes::Bytes;
use litellm_core::messages::route::{Messages, MessagesStreamHead};
use litellm_host::host::{Demand, Host};
use rstest::rstest;

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
    let mut body = call.body.clone();
    body.insert("stream".into(), json!(true));
    MessagesCall {
        api_key: Some("sk-ant".into()),
        api_base: Some(api_base),
        body,
        ..call
    }
}

fn sse_response() -> ResponseTemplate {
    UPSTREAM_HEADERS.iter().fold(
        ResponseTemplate::new(200).set_body_raw(SSE_BODY, "text/event-stream"),
        |response, (name, value)| response.insert_header(*name, *value),
    )
}

async fn stream_through(host: &RecordingStreamHost) -> Result<MessagesOutput, Error> {
    litellm_host::run::run(messages_machine(Arc::new(RecordingSecrets::empty())), host).await
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
#[tokio::test]
async fn an_upstream_error_fails_the_call_without_opening_the_stream(call: MessagesCall) {
    let upstream = upstream([ResponseTemplate::new(429).set_body_string("slow down")]).await;
    let host = RecordingStreamHost::new(streaming(call, upstream.uri()), usize::MAX);

    let error = stream_through(&host)
        .await
        .err()
        .expect("upstream error propagates");

    assert!(
        matches!(
            error,
            Error::Transport(litellm_http::transport::Error::Http { status: 429, .. })
        ),
        "{error:?}"
    );
    assert!(host.seen.into_inner().unwrap().is_empty());
}

#[rstest]
#[tokio::test]
async fn streaming_is_refused_for_providers_that_cannot_stream(call: MessagesCall) {
    let upstream = upstream([sse_response()]).await;
    let host = RecordingStreamHost::new(
        MessagesCall {
            custom_llm_provider: Some("azure_ai".into()),
            ..streaming(call, upstream.uri())
        },
        usize::MAX,
    );

    let error = stream_through(&host)
        .await
        .err()
        .expect("azure streaming is refused");

    assert_eq!(
        error,
        Error::Unsupported("streaming messages for this provider")
    );
    assert!(received(&upstream).await.is_empty());
}
