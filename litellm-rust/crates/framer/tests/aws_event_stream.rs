#![cfg(feature = "aws")]

mod support;

use std::io;

use futures_util::{StreamExt, TryStreamExt};
use litellm_framing::aws_event_stream::{Message, frames};
use litellm_framing::{Error, MAX_FRAME_BYTES};
use rstest::{fixture, rstest};

use support::encode;

async fn collect(bytes: &[u8], chunk_size: usize) -> Result<Vec<Message>, Error> {
    frames(futures_util::stream::iter(
        bytes.chunks(chunk_size).map(Ok::<_, io::Error>),
    ))
    .try_collect()
    .await
}

#[fixture]
fn two_frames() -> Vec<u8> {
    [encode(b"\xff\x00"), encode(b"second")].concat()
}

#[fixture]
fn payload_frame() -> Vec<u8> {
    encode(b"payload")
}

#[rstest]
#[case(1)]
#[case(3)]
#[case(12)]
#[case(usize::MAX)]
#[tokio::test]
async fn fragmented_and_coalesced_frames_preserve_typed_headers_and_binary_payloads(
    two_frames: Vec<u8>,
    #[case] chunk_size: usize,
) {
    let chunk_size = chunk_size.min(two_frames.len());
    let frames = collect(&two_frames, chunk_size).await.unwrap();
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].payload().as_ref(), b"\xff\x00");
    assert_eq!(frames[1].payload().as_ref(), b"second");
    assert_eq!(
        frames[0].headers()[0].value().as_string().unwrap().as_str(),
        "payload"
    );
    assert_eq!(frames[0].headers()[1].value().as_int32(), Ok(7));
}

#[rstest]
#[case(8)]
#[case(usize::MAX)]
#[tokio::test]
async fn rejects_corrupt_crcs(payload_frame: Vec<u8>, #[case] index: usize) {
    let corrupt_index = index.min(payload_frame.len() - 1);
    let mut corrupt = payload_frame;
    corrupt[corrupt_index] ^= 1;
    assert!(matches!(collect(&corrupt, 3).await, Err(Error::Aws(_))));
}

#[rstest]
#[case(0_u32)]
#[case(15)]
#[tokio::test]
async fn rejects_lengths_below_the_prelude_and_crc(#[case] length: u32) {
    assert!(matches!(
        collect(&length.to_be_bytes(), 1).await,
        Err(Error::InvalidLength(_))
    ));
}

#[rstest]
#[case(MAX_FRAME_BYTES as u32 + 1)]
#[case(u32::MAX)]
#[tokio::test]
async fn rejects_oversized_lengths_from_the_prefix_alone(#[case] length: u32) {
    assert!(matches!(
        collect(&length.to_be_bytes(), 1).await,
        Err(Error::FrameTooLarge)
    ));
}

#[tokio::test]
async fn accepts_a_length_at_the_cap() {
    assert!(matches!(
        collect(&(MAX_FRAME_BYTES as u32).to_be_bytes(), 1).await,
        Err(Error::Truncated)
    ));
}

#[rstest]
#[case(1)]
#[case(3)]
#[case(5)]
#[tokio::test]
async fn rejects_truncation(payload_frame: Vec<u8>, #[case] end: usize) {
    assert!(matches!(
        collect(&payload_frame[..end], 1).await,
        Err(Error::Truncated)
    ));
}

#[rstest]
#[tokio::test]
async fn body_errors_terminate_and_preserve_the_cause(payload_frame: Vec<u8>) {
    let mut frames = Box::pin(frames(futures_util::stream::iter([
        Err(io::Error::new(io::ErrorKind::ConnectionReset, "reset")),
        Ok(&payload_frame[..]),
    ])));
    let error = frames.next().await.unwrap().unwrap_err();
    assert!(matches!(
        error,
        Error::Body(ref cause) if cause.downcast_ref::<io::Error>().unwrap().kind() == io::ErrorKind::ConnectionReset
    ));
    assert!(frames.next().await.is_none());
    assert!(frames.next().await.is_none());
}
