#![cfg(feature = "sse")]

use std::io;

use bytes::Bytes;
use futures_util::{StreamExt, TryStreamExt};
use litellm_framing::sse::{SseFrame, frames};
use litellm_framing::{Error, MAX_FRAME_BYTES};
use rstest::rstest;

fn message(data: &str) -> SseFrame {
    SseFrame {
        event: None,
        data: data.into(),
        id: None,
        retry: None,
    }
}

async fn collect(chunks: &[&[u8]]) -> Result<Vec<SseFrame>, Error> {
    frames(futures_util::stream::iter(
        chunks.iter().copied().map(Ok::<_, io::Error>),
    ))
    .try_collect()
    .await
}

#[rstest]
#[case::fragmented_utf8_crlf_and_multiline_data(
    &[&b":ping\r\nevent: delta\r\nid: 7\r\nretry: 10\r\ndata: \xe2"[..], &b"\x82"[..], &b"\xac\r"[..], &b"\ndata: next\r\n\r"[..], &b"\ndata: [DONE]\n\n"[..]],
    vec![
        SseFrame {
            event: Some("delta".into()),
            data: "€\nnext".into(),
            id: Some("7".into()),
            retry: Some(10),
        },
        message("[DONE]"),
    ]
)]
#[case::bom_comments_and_leading_blank_lines(&[&b"\xEF\xBB\xBF\n\n:ping\ndata: x\n\n"[..]], vec![message("x")])]
#[case::unknown_fields_ignored_and_colonless_line_is_an_empty_field(&[&b"foo: bar\nignored\ndata\n\n"[..]], vec![message("")])]
#[case::last_event_wins_and_bad_retry_or_nul_id_are_ignored(
    &[&b"event: a\nevent: b\nretry: 5x\nid: a\0b\ndata: x\n\n"[..]],
    vec![SseFrame { event: Some("b".into()), data: "x".into(), id: None, retry: None }]
)]
#[case::blocks_without_data_do_not_dispatch(&[&b"event: a\nid: 1\n\ndata: x\n\n"[..]], vec![message("x")])]
#[case::lone_cr_terminators(&[&b"data: a\rdata: b\r\r"[..]], vec![message("a\nb")])]
#[case::lone_cr_at_eof_ends_the_blank_line(&[&b"data: a\r\n\r"[..]], vec![message("a")])]
#[case::only_one_leading_space_is_stripped(&[&b"data:  two\ndata:none\n\n"[..]], vec![message(" two\nnone")])]
#[tokio::test]
async fn parses_events_per_the_whatwg_stream_grammar(
    #[case] chunks: &[&[u8]],
    #[case] expected: Vec<SseFrame>,
) {
    assert_eq!(collect(chunks).await.unwrap(), expected);
}

#[tokio::test]
async fn eof_does_not_dispatch_an_unterminated_frame() {
    assert!(
        collect(&[&b"data: partial\n"[..]])
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn rejects_invalid_utf8() {
    assert!(matches!(
        collect(&[&b"data: \xff\n\n"[..]]).await,
        Err(Error::InvalidUtf8(_))
    ));
}

#[tokio::test]
async fn an_endless_unterminated_event_fails_at_the_cap_instead_of_buffering_forever() {
    let chunk = Bytes::from(vec![b'a'; MAX_FRAME_BYTES / 16]);
    let mut frames = Box::pin(frames(futures_util::stream::repeat_with(move || {
        Ok::<_, io::Error>(chunk.clone())
    })));
    assert!(matches!(
        frames.next().await.unwrap(),
        Err(Error::FrameTooLarge)
    ));
    assert!(frames.next().await.is_none());
}

#[rstest]
#[case(io::ErrorKind::ConnectionReset)]
#[case(io::ErrorKind::UnexpectedEof)]
#[tokio::test]
async fn body_errors_terminate_and_preserve_the_cause(#[case] kind: io::ErrorKind) {
    let mut frames = Box::pin(frames(futures_util::stream::iter([
        Err(io::Error::new(kind, "reset")),
        Ok(&b"data: later\n\n"[..]),
    ])));
    let error = frames.next().await.unwrap().unwrap_err();
    assert!(matches!(
        error,
        Error::Body(ref cause) if cause.downcast_ref::<io::Error>().unwrap().kind() == kind
    ));
    assert!(frames.next().await.is_none());
    assert!(frames.next().await.is_none());
}
