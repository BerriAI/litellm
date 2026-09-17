#![cfg(all(feature = "aws", feature = "sse"))]

mod support;

use std::io;

use futures_util::TryStreamExt;
use litellm_framing::sse::SseFrame;
use litellm_framing::{aws_event_stream, sse};

use support::encode;

#[tokio::test]
async fn hosting_payloads_feed_the_same_sse_parser_across_envelope_boundaries() {
    let bytes = [encode(b"event: delta\ndata: hel"), encode(b"lo\nid: 7\n\n")].concat();
    let envelopes = aws_event_stream::frames(futures_util::stream::iter(
        bytes.chunks(3).map(Ok::<_, io::Error>),
    ));
    let frames = sse::frames(envelopes.map_ok(|message| message.payload().clone()))
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert_eq!(
        frames,
        vec![SseFrame {
            event: Some("delta".into()),
            data: "hello".into(),
            id: Some("7".into()),
            retry: None,
        }]
    );
}
