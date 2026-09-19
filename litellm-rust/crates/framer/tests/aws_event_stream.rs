#![cfg(feature = "aws")]

mod support;

use std::io;

use futures_util::TryStreamExt;
use litellm_framing::aws_event_stream::{AwsEventStreamFrame, AwsEventStreamFramer};
use litellm_framing::{Error, Framer};
use rstest::{fixture, rstest};

use support::encode;

async fn collect_aws(bytes: &[u8], chunk_size: usize) -> Result<Vec<AwsEventStreamFrame>, Error> {
    AwsEventStreamFramer
        .frame(futures_util::stream::iter(
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
    let frames = collect_aws(&two_frames, chunk_size).await.unwrap();
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].payload, &b"\xff\x00"[..]);
    assert_eq!(frames[1].payload, "second");
    assert_eq!(
        frames[0].headers[0].value().as_string().unwrap().as_str(),
        "payload"
    );
    assert_eq!(frames[0].headers[1].value().as_int32(), Ok(7));
}

#[rstest]
#[case(8)]
#[case(0)]
#[tokio::test]
async fn rejects_corrupt_crcs(payload_frame: Vec<u8>, #[case] index: usize) {
    let corrupt_index = if index == 0 {
        payload_frame.len() - 1
    } else {
        index
    };
    let mut corrupt = payload_frame;
    corrupt[corrupt_index] ^= 1;
    assert!(matches!(collect_aws(&corrupt, 3).await, Err(Error::Aws(_))));
}

#[rstest]
#[case(0_u32)]
#[case(15)]
#[case(u32::MAX)]
#[tokio::test]
async fn rejects_invalid_lengths(#[case] length: u32) {
    assert!(matches!(
        collect_aws(&length.to_be_bytes(), 1).await,
        Err(Error::InvalidLength(_))
    ));
}

#[rstest]
#[case(1)]
#[case(3)]
#[case(5)]
#[tokio::test]
async fn rejects_truncation(payload_frame: Vec<u8>, #[case] end: usize) {
    assert!(matches!(
        collect_aws(&payload_frame[..end], 1).await,
        Err(Error::Truncated)
    ));
}
