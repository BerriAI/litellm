#![cfg(feature = "sse")]

use std::io;

use futures_util::{StreamExt, TryStreamExt};
use litellm_framing::sse::{SseFrame, SseFramer};
use litellm_framing::{Error, Framer};
use rstest::rstest;

async fn collect_sse(chunks: &[&[u8]]) -> Result<Vec<SseFrame>, Error> {
    SseFramer
        .frame(futures_util::stream::iter(
            chunks.iter().copied().map(Ok::<_, io::Error>),
        ))
        .try_collect()
        .await
}

#[rstest]
#[case(
    &[&b":ping\r\nevent: delta\r\nid: 7\r\nretry: 10\r\ndata: \xe2"[..], &b"\x82"[..], &b"\xac\r"[..], &b"\ndata: next\r\n\r"[..], &b"\ndata: [DONE]\n\n"[..]],
    vec![
        SseFrame {
            event: Some("delta".into()),
            data: Some("€\nnext".into()),
            id: Some("7".into()),
            retry: Some(10),
        },
        SseFrame {
            event: None,
            data: Some("[DONE]".into()),
            id: None,
            retry: None,
        },
    ]
)]
#[tokio::test]
async fn fragmented_utf8_crlf_and_multiline_data_retain_metadata_and_sentinel(
    #[case] chunks: &[&[u8]],
    #[case] expected: Vec<SseFrame>,
) {
    assert_eq!(collect_sse(chunks).await.unwrap(), expected);
}

#[tokio::test]
async fn eof_does_not_dispatch_an_unterminated_frame() {
    assert!(collect_sse(&[b"data: partial\n"]).await.unwrap().is_empty());
}

#[rstest]
#[case(io::ErrorKind::ConnectionReset)]
#[case(io::ErrorKind::UnexpectedEof)]
#[tokio::test]
async fn framing_errors_terminate_and_preserve_input_error_causes(#[case] kind: io::ErrorKind) {
    let mut frames = Box::pin(SseFramer.frame(futures_util::stream::iter([
        Err(io::Error::new(kind, "reset")),
        Ok(&b"data: later\n\n"[..]),
    ])));
    let error = frames.next().await.unwrap().unwrap_err();
    assert!(matches!(
        error,
        Error::Sse(sse_stream::Error::Body(ref cause))
            if cause.downcast_ref::<io::Error>().unwrap().kind() == kind
    ));
    assert!(frames.next().await.is_none());
    assert!(frames.next().await.is_none());
}
