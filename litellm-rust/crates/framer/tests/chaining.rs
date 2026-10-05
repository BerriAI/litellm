#![cfg(all(feature = "aws", feature = "sse"))]

mod support;

use bytes::Bytes;
use futures_util::{StreamExt, TryStreamExt};
use litellm_framing::{
    EventStreamError, SseError,
    aws_event_stream::{AwsEventStreamCodec, Message},
    frames,
    sse::{SseCodec, SseEvent},
};
use proptest::prelude::*;
use support::{body_cause, cut_at, encode_all, every, input, runtime};

fn delta(data: &str) -> SseEvent {
    SseEvent {
        event: Some("delta".into()),
        data: data.into(),
        id: Some("7".into()),
        retry: None,
    }
}

fn envelopes(payloads: Vec<Bytes>) -> Vec<u8> {
    encode_all(AwsEventStreamCodec, payloads.into_iter().map(Message::new))
}

proptest! {
    #[test]
    fn an_sse_event_cut_anywhere_across_envelopes_is_reassembled(cut in 0_usize..64, chunk in 1_usize..8) {
        let sse = encode_all(SseCodec::default(), [delta("hello")]);
        let wire = envelopes(cut_at(&sse, [cut.min(sse.len())]));
        let events = runtime().block_on(async {
            let payloads = frames(input(every(&wire, chunk)), AwsEventStreamCodec)
                .map_ok(|message| message.payload().clone());
            frames(payloads, SseCodec::default()).try_collect::<Vec<_>>().await
        })
        .unwrap();
        prop_assert_eq!(events, vec![delta("hello")]);
    }
}

#[tokio::test]
async fn a_truncated_envelope_after_an_sse_event_keeps_the_event_and_its_cause() {
    let complete = encode_all(SseCodec::default(), [delta("complete")]);
    let incomplete = encode_all(SseCodec::default(), [delta("incomplete")]);
    let wire = envelopes(vec![complete.into(), incomplete.into()]);
    let payloads = frames(
        input(every(&wire[..wire.len() - 1], 3)),
        AwsEventStreamCodec,
    )
    .map_ok(|message| message.payload().clone());
    let mut events = Box::pin(frames(payloads, SseCodec::default()));

    assert_eq!(events.next().await.unwrap().unwrap(), delta("complete"));
    let Some(Err(SseError::Body(body))) = events.next().await else {
        panic!("the envelope error surfaces through the SSE layer");
    };
    assert!(matches!(
        body_cause::<EventStreamError>(&body),
        Some(EventStreamError::Truncated)
    ));
    assert!(events.next().await.is_none());
}
