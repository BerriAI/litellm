#![cfg(feature = "aws")]

mod support;

use std::io;

use bytes::Bytes;
use futures_util::{StreamExt, TryStreamExt, stream};
use litellm_framing::{
    EventStreamError,
    aws_event_stream::{AwsEventStreamCodec, Header, HeaderValue, Message},
    frames,
};
use proptest::prelude::*;
use rstest::{fixture, rstest};
use support::{body_cause, cut_at, encode_all, every, input, runtime};

async fn collect(pieces: Vec<Bytes>) -> Result<Vec<Message>, EventStreamError> {
    frames(input(pieces), AwsEventStreamCodec)
        .try_collect()
        .await
}

fn message(payload: &[u8]) -> Message {
    Message::new(Bytes::copy_from_slice(payload))
        .add_header(Header::new(
            ":event-type",
            HeaderValue::String("payload".into()),
        ))
        .add_header(Header::new("sequence", HeaderValue::Int32(7)))
}

#[fixture]
fn payload_frame() -> Vec<u8> {
    encode_all(AwsEventStreamCodec, [message(b"payload")])
}

fn header_value() -> impl Strategy<Value = HeaderValue> {
    prop_oneof![
        "[a-z]{0,8}".prop_map(|text| HeaderValue::String(text.into())),
        any::<i32>().prop_map(HeaderValue::Int32),
        any::<bool>().prop_map(HeaderValue::Bool),
        proptest::collection::vec(any::<u8>(), 0..8)
            .prop_map(|bytes| HeaderValue::ByteArray(bytes.into())),
    ]
}

fn arbitrary_message() -> impl Strategy<Value = Message> {
    (
        proptest::collection::vec(("[a-z:-]{1,12}", header_value()), 0..3),
        proptest::collection::vec(any::<u8>(), 0..32),
    )
        .prop_map(|(headers, payload)| {
            headers.into_iter().fold(
                Message::new(Bytes::from(payload)),
                |message, (name, value)| message.add_header(Header::new(name, value)),
            )
        })
}

proptest! {
    #[test]
    fn any_messages_survive_a_round_trip_through_any_cuts(
        messages in proptest::collection::vec(arbitrary_message(), 1..4),
        cuts in proptest::collection::vec(0_usize..512, 0..4),
    ) {
        let wire = encode_all(AwsEventStreamCodec, messages.clone());
        let decoded = runtime().block_on(collect(cut_at(&wire, cuts))).unwrap();
        prop_assert_eq!(decoded, messages);
    }
}

#[rstest]
#[case::prelude_crc(8)]
#[case::message_crc(usize::MAX)]
#[tokio::test]
async fn a_corrupt_crc_is_malformed(payload_frame: Vec<u8>, #[case] index: usize) {
    let mut corrupt = payload_frame;
    let flipped = index.min(corrupt.len() - 1);
    corrupt[flipped] ^= 1;
    assert!(matches!(
        collect(every(&corrupt, 3)).await,
        Err(EventStreamError::Malformed(_))
    ));
}

#[rstest]
#[case::zero(0)]
#[case::below_minimum(15)]
#[case::above_maximum(16 * 1024 * 1024 + 1)]
#[case::u32_max(u32::MAX)]
#[tokio::test]
async fn a_length_outside_the_frame_bounds_fails_before_buffering(#[case] length: u32) {
    assert!(matches!(
        collect(every(&length.to_be_bytes(), 1)).await,
        Err(EventStreamError::InvalidLength(seen)) if seen == length as usize
    ));
}

#[rstest]
#[case::before_the_length(1)]
#[case::inside_the_prelude(5)]
#[case::one_byte_short(usize::MAX)]
#[tokio::test]
async fn eof_inside_a_frame_is_truncation(payload_frame: Vec<u8>, #[case] end: usize) {
    let end = end.min(payload_frame.len() - 1);
    assert!(matches!(
        collect(every(&payload_frame[..end], 1)).await,
        Err(EventStreamError::Truncated)
    ));
}

const FRAME_OVERHEAD_BYTES: usize = 16;
const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

#[tokio::test]
async fn a_frame_at_exactly_the_maximum_length_decodes() {
    let largest = Message::new(vec![0xAB; MAX_FRAME_BYTES - FRAME_OVERHEAD_BYTES]);
    let wire = encode_all(AwsEventStreamCodec, [largest.clone()]);
    assert_eq!(wire.len(), MAX_FRAME_BYTES);
    assert_eq!(collect(every(&wire, 1 << 20)).await.unwrap(), vec![largest]);
}

#[tokio::test]
async fn a_frame_one_byte_over_the_maximum_length_is_rejected_by_its_prelude() {
    let oversized = Message::new(vec![0xAB; MAX_FRAME_BYTES - FRAME_OVERHEAD_BYTES + 1]);
    let wire = encode_all(AwsEventStreamCodec, [oversized]);
    assert!(matches!(
        collect(every(&wire[..4], 1)).await,
        Err(EventStreamError::InvalidLength(length)) if length == MAX_FRAME_BYTES + 1
    ));
}

#[tokio::test]
async fn an_empty_body_yields_nothing() {
    assert_eq!(collect(vec![]).await.unwrap(), vec![]);
}

#[tokio::test]
async fn a_complete_frame_precedes_a_truncated_following_frame() {
    let wire = encode_all(AwsEventStreamCodec, [message(b"first"), message(b"second")]);
    let mut messages = Box::pin(frames(
        input(every(&wire[..wire.len() - 1], 3)),
        AwsEventStreamCodec,
    ));

    assert_eq!(messages.next().await.unwrap().unwrap(), message(b"first"));
    assert!(matches!(
        messages.next().await,
        Some(Err(EventStreamError::Truncated))
    ));
    assert!(messages.next().await.is_none());
}

#[tokio::test]
async fn a_body_error_after_a_complete_frame_preserves_its_cause() {
    let first = encode_all(AwsEventStreamCodec, [message(b"first")]);
    let mut messages = Box::pin(frames(
        stream::iter([
            Ok(cut_at(&first, [5])[0].clone()),
            Ok(cut_at(&first, [5])[1].clone()),
            Ok(Bytes::from_static(b"\0\0\0")),
            Err(io::Error::new(io::ErrorKind::ConnectionReset, "reset")),
        ]),
        AwsEventStreamCodec,
    ));

    assert_eq!(messages.next().await.unwrap().unwrap(), message(b"first"));
    let Some(Err(EventStreamError::Body(body))) = messages.next().await else {
        panic!("the body error surfaces");
    };
    assert_eq!(
        body_cause::<io::Error>(&body).unwrap().kind(),
        io::ErrorKind::ConnectionReset
    );
    assert!(messages.next().await.is_none());
}
