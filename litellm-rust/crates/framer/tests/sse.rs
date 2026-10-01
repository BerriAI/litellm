#![cfg(feature = "sse")]

mod support;

use std::io;

use bytes::Bytes;
use futures_util::{StreamExt, TryStreamExt, stream};
use litellm_framing::{
    SseError, frames,
    sse::{SseCodec, SseEvent},
};
use proptest::prelude::*;
use rstest::rstest;
use support::{body_cause, cut_at, encode_all, every, input, runtime};

async fn collect(pieces: Vec<Bytes>) -> Result<Vec<SseEvent>, SseError> {
    frames(input(pieces), SseCodec::default())
        .try_collect()
        .await
}

fn event(name: Option<&str>, data: &str) -> SseEvent {
    SseEvent {
        event: name.map(str::to_owned),
        data: data.to_owned(),
        id: None,
        retry: None,
    }
}

fn sse_event() -> impl Strategy<Value = SseEvent> {
    (
        proptest::option::of("[^\r\n\0]{0,8}"),
        "[^\r\0]{0,16}",
        proptest::option::of("[^\r\n\0]{0,8}"),
        proptest::option::of(any::<u64>()),
    )
        .prop_map(|(event, data, id, retry)| SseEvent {
            event,
            data,
            id,
            retry,
        })
}

fn terminators() -> impl Strategy<Value = &'static [u8]> {
    prop_oneof![Just(&b"\n"[..]), Just(&b"\r\n"[..]), Just(&b"\r"[..])]
}

proptest! {
    #[test]
    fn any_events_survive_a_round_trip_through_any_terminator_and_any_cuts(
        events in proptest::collection::vec(sse_event(), 1..4),
        terminator in terminators(),
        cuts in proptest::collection::vec(0_usize..256, 0..4),
        bom in any::<bool>(),
    ) {
        let lf_wire = encode_all(SseCodec::default(), events.clone());
        let body: Vec<u8> = lf_wire
            .iter()
            .flat_map(|byte| if *byte == b'\n' { terminator.to_vec() } else { vec![*byte] })
            .collect();
        let wire = if bom { [&b"\xEF\xBB\xBF"[..], &body].concat() } else { body };
        let decoded = runtime().block_on(collect(cut_at(&wire, cuts))).unwrap();
        prop_assert_eq!(decoded, events);
    }
}

#[rstest]
#[case::comment(b":ping\ndata: x\n\n")]
#[case::unknown_field(b"vendor: 1\ndata: x\n\n")]
#[case::field_without_colon(b"garbage\ndata: x\n\n")]
#[case::retry_with_non_digits(b"retry: soon\ndata: x\n\n")]
#[case::retry_with_a_sign(b"retry: +5\ndata: x\n\n")]
#[case::retry_without_a_value(b"retry:\ndata: x\n\n")]
#[case::id_with_nul(b"id: a\0b\ndata: x\n\n")]
#[tokio::test]
async fn lines_the_spec_ignores_do_not_change_the_event(#[case] wire: &[u8]) {
    assert_eq!(
        collect(every(wire, 1)).await.unwrap(),
        vec![event(None, "x")]
    );
}

#[rstest]
#[case::no_data_at_all(b"event: ping\nid: 1\n\ndata: x\n\n", vec![event(None, "x")])]
#[case::empty_data_field(b"data:\n\n", vec![event(None, "")])]
#[case::one_leading_space_stripped(b"data:  x\n\n", vec![event(None, " x")])]
#[case::multiline_data(b"data: a\ndata: b\ndata:\n\n", vec![event(None, "a\nb\n")])]
#[case::last_event_name_wins(b"event: a\nevent: b\ndata: x\n\n", vec![event(Some("b"), "x")])]
#[case::last_retry_wins(b"retry: 1\nretry: 2\ndata: x\n\n", vec![SseEvent { retry: Some(2), ..event(None, "x") }])]
#[case::split_utf8_across_lines_is_not_joined(b"data: \xe2\x82\xac\ndata: \xe2\x82\xac\n\n", vec![event(None, "€\n€")])]
#[tokio::test]
async fn dispatch_follows_the_data_buffer(#[case] wire: &[u8], #[case] expected: Vec<SseEvent>) {
    assert_eq!(collect(every(wire, 1)).await.unwrap(), expected);
}

#[rstest]
#[case::unterminated_single(b"data: partial\n", vec![])]
#[case::unterminated_tail_after_complete(b"data: complete\n\ndata: unfinished\n", vec![event(None, "complete")])]
#[case::lone_cr_terminates_at_eof(b"data: x\r\r", vec![event(None, "x")])]
#[case::lone_cr_line_then_eof(b"data: x\r", vec![])]
#[tokio::test]
async fn eof_dispatches_only_terminated_events(
    #[case] wire: &[u8],
    #[case] expected: Vec<SseEvent>,
) {
    assert_eq!(
        collect(vec![Bytes::copy_from_slice(wire)]).await.unwrap(),
        expected
    );
}

#[rstest]
#[case::inside_the_first_line(vec![&b"data: a\r"[..], &b"\ndata: b\r\n\r\n"[..]])]
#[case::inside_the_blank_line(vec![&b"data: a\r\ndata: b\r\n\r"[..], &b"\n"[..]])]
#[tokio::test]
async fn a_crlf_split_across_chunks_is_one_terminator(#[case] pieces: Vec<&[u8]>) {
    let pieces = pieces.into_iter().map(Bytes::copy_from_slice).collect();
    assert_eq!(collect(pieces).await.unwrap(), vec![event(None, "a\nb")]);
}

#[tokio::test]
async fn a_bom_is_stripped_only_at_the_start_of_the_stream() {
    let wire = b"\xEF\xBB\xBFdata: a\n\n\xEF\xBB\xBFdata: b\ndata: c\n\n";
    let decoded = collect(every(wire, 2)).await.unwrap();
    assert_eq!(decoded, vec![event(None, "a"), event(None, "c")]);
}

#[tokio::test]
async fn invalid_utf8_in_a_field_fails_after_earlier_events_and_terminates() {
    let mut events = Box::pin(frames(
        input(every(b"data: ok\n\ndata: \xff\n\n", 3)),
        SseCodec::default(),
    ));

    assert_eq!(events.next().await.unwrap().unwrap(), event(None, "ok"));
    assert!(matches!(
        events.next().await,
        Some(Err(SseError::InvalidUtf8(_)))
    ));
    assert!(events.next().await.is_none());
}

#[rstest]
#[case(io::ErrorKind::ConnectionReset)]
#[case(io::ErrorKind::UnexpectedEof)]
#[tokio::test]
async fn a_body_error_keeps_earlier_events_and_its_cause_then_terminates(
    #[case] kind: io::ErrorKind,
) {
    let mut events = Box::pin(frames(
        stream::iter([
            Ok(&b"data: first\n\ndata: partial"[..]),
            Err(io::Error::new(kind, "reset")),
            Ok(&b"\n\n"[..]),
        ]),
        SseCodec::default(),
    ));

    assert_eq!(events.next().await.unwrap().unwrap(), event(None, "first"));
    let Some(Err(SseError::Body(body))) = events.next().await else {
        panic!("the body error surfaces");
    };
    assert_eq!(body_cause::<io::Error>(&body).unwrap().kind(), kind);
    assert!(events.next().await.is_none());
    assert!(events.next().await.is_none());
}
