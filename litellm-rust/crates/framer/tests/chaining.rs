#![cfg(all(feature = "aws", feature = "sse"))]

mod support;

use std::io;

use futures_util::TryStreamExt;
use litellm_framing::Framer;
use litellm_framing::aws_event_stream::{AwsEventStreamFrame, AwsEventStreamFramer};
use litellm_framing::sse::SseFramer;

use support::encode;

#[tokio::test]
async fn hosting_payloads_feed_the_same_sse_framer_across_envelope_boundaries() {
    let bytes = [encode(b"event: delta\ndata: hel"), encode(b"lo\nid: 7\n\n")].concat();
    let envelopes = AwsEventStreamFramer.frame(futures_util::stream::iter(
        bytes.chunks(3).map(Ok::<_, io::Error>),
    ));
    let frames = SseFramer
        .frame(envelopes.map_ok(|frame: AwsEventStreamFrame| frame.payload))
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].event.as_deref(), Some("delta"));
    assert_eq!(frames[0].data.as_deref(), Some("hello"));
    assert_eq!(frames[0].id.as_deref(), Some("7"));
}
