use std::{
    convert::Infallible,
    sync::Arc,
    sync::atomic::{AtomicUsize, Ordering},
};

use axum::body::to_bytes;
use bytes::Bytes;
use futures_util::{StreamExt, stream};
use http::{StatusCode, header::CONTENT_TYPE};
use litellm_host::{
    call::{CallOutput, hosted_call},
    machine::MachineFault,
    protocol::Protocol,
};
use litellm_host_http::{Sse, serve};
use rstest::rstest;

#[derive(Clone, Debug)]
enum TestError {
    Upstream,
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
    type Request = ();
    type HostCall = Infallible;
    type Chunk = Bytes;
    type StreamHead = ();
}

#[rstest]
#[case::complete(false)]
#[case::failed(true)]
#[tokio::test]
async fn sse_preserves_encoded_chunks_and_uses_the_supplied_error_format(#[case] fail: bool) {
    let first = Bytes::from_static(b"event: custom\ndata: first\n\n");
    let last = Bytes::from_static(b"data: [DONE]\n\n");
    let machine = hosted_call::<TestProtocol, _, _>((), move |(), _, _| async move {
        let chunks = stream::iter([
            Ok(first),
            if fail {
                Err(TestError::Upstream)
            } else {
                Ok(last)
            },
        ])
        .boxed();
        Ok(CallOutput::Stream { head: (), chunks })
    });
    let errors = Arc::new(AtomicUsize::new(0));
    let formatted_errors = errors.clone();
    let adapter = Sse::new(std::convert::identity, move |error| {
        formatted_errors.fetch_add(1, Ordering::SeqCst);
        Bytes::from(format!("event: custom_error\ndata: {error:?}\n\n"))
    });
    let response = serve(machine, (), (), adapter).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[CONTENT_TYPE], "text/event-stream");
    assert_eq!(errors.load(Ordering::SeqCst), 0);
    let body = to_bytes(response.into_body(), 1024).await.unwrap();
    assert_eq!(
        body,
        if fail {
            "event: custom\ndata: first\n\nevent: custom_error\ndata: Call(Upstream)\n\n"
        } else {
            "event: custom\ndata: first\n\ndata: [DONE]\n\n"
        }
    );
    assert_eq!(errors.load(Ordering::SeqCst), usize::from(fail));
}

#[rstest]
#[tokio::test]
async fn completed_calls_use_the_response_converter_without_sse_headers() {
    use axum::response::IntoResponse;

    let machine = hosted_call::<TestProtocol, _, _>((), |(), _, _| async {
        Ok(CallOutput::Complete(Bytes::from_static(b"completed")))
    });
    let adapter = Sse::new(
        |response| (StatusCode::CREATED, [("x-converted", "yes")], response).into_response(),
        |_| panic!("a completed call cannot format a stream error"),
    );
    let response = serve(machine, (), (), adapter).await.unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);
    assert_eq!(response.headers()["x-converted"], "yes");
    assert_ne!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .map(|value| value.as_bytes()),
        Some(b"text/event-stream".as_slice())
    );
    assert_eq!(
        to_bytes(response.into_body(), 1024).await.unwrap(),
        "completed"
    );
}
