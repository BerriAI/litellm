//! A stream held back while another attempt can still answer in its place. Until a stream sends
//! generated content its client has nothing to lose, so its chunks are kept instead of sent on,
//! and the stream settles one of two ways: content begins and every chunk goes out as the
//! provider sent it, or the stream fails first and `routing::call` may hand the request to a
//! retry or a fallback, as Python's `wrap_anthropic_stream` does.

use futures_util::{
    StreamExt,
    stream::{self, BoxStream},
};
use litellm_framer::sse::SseBlock;
use litellm_router_core::AttemptError;

use crate::stream_event::{Kind, Read};

/// `MAX_BUFFERED_PRE_CONTENT_ANTHROPIC_CHUNKS` in `litellm/router_rust/invoke/streaming.py`: a
/// stream that sends this many chunks without content goes out as it is.
const MAX_HELD_CHUNKS: usize = 200;

/// A stream as the SSE blocks it was read in.
pub type Chunks<E> = BoxStream<'static, Result<SseBlock, E>>;

/// How a stream told its reader that it failed.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Cause {
    /// The provider sent its error as an event of a stream that otherwise went well. `status`
    /// and `body` are the answer the provider gives the same error before any stream.
    ErrorEvent { status: u16, body: String },
    /// The stream broke under the reader.
    Broke,
}

/// A stream's failure before any content, as router-core reads it.
pub(crate) struct Failed {
    pub cause: Cause,
    pub failure: AttemptError,
}

/// How a held stream settled. Either way the stream is handed back from its first byte.
pub(crate) enum Held<E> {
    /// Content began, the stream ended, or it sent more than is held.
    Committed(Chunks<E>),
    /// It failed before any content, and `sent` still carries the failure.
    Failed { failed: Failed, sent: Chunks<E> },
}

enum Settled {
    Content,
    Failed(Failed),
}

/// Hold `chunks` until the stream settles. `read` reads the events of the stream's format, and
/// `broke` a stream that broke under the reader.
pub(crate) async fn hold<E: Send + 'static>(
    chunks: Chunks<E>,
    read: Read,
    broke: fn(&E) -> AttemptError,
) -> Held<E> {
    // A stream read to its end is handed back too, and may not be polled past it.
    let mut chunks = chunks.fuse();
    let mut kept = Vec::new();
    while kept.len() < MAX_HELD_CHUNKS {
        let Some(chunk) = chunks.next().await else {
            break;
        };
        let settled = match &chunk {
            Ok(block) => block
                .event
                .as_ref()
                .and_then(|event| match read(&event.data).kind {
                    Kind::Lifecycle => None,
                    Kind::Content => Some(Settled::Content),
                    Kind::Failed { status, answer } => Some(Settled::Failed(Failed {
                        failure: AttemptError::from_provider(status, &answer, None),
                        cause: Cause::ErrorEvent {
                            status,
                            body: answer,
                        },
                    })),
                }),
            Err(error) => Some(Settled::Failed(Failed {
                cause: Cause::Broke,
                failure: broke(error),
            })),
        };
        kept.push(chunk);
        match settled {
            None => {}
            Some(Settled::Content) => break,
            Some(Settled::Failed(failed)) => {
                return Held::Failed {
                    failed,
                    sent: resent(kept, chunks),
                };
            }
        }
    }
    Held::Committed(resent(kept, chunks))
}

fn resent<E: Send + 'static>(
    kept: Vec<Result<SseBlock, E>>,
    rest: stream::Fuse<Chunks<E>>,
) -> Chunks<E> {
    stream::iter(kept).chain(rest).boxed()
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use litellm_framer::sse;
    use litellm_router_core::FailureKind;
    use rstest::rstest;

    use super::*;
    use crate::stream_event::messages_event;

    const START: &str =
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{}}\n\n";
    const PING: &str = "event: ping\ndata: {\"type\":\"ping\"}\n\n";
    const BLOCK: &str =
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0}\n\n";
    const DELTA: &str = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n";
    const OVERLOADED: &str = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}\n\n";
    const UNTYPED: &str = "event: error\ndata: {\"type\":\"error\"}\n\n";

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Dropped;

    fn dropped(_: &Dropped) -> AttemptError {
        AttemptError::new(FailureKind::ApiConnection, None)
    }

    type Sent = Vec<Result<Bytes, Dropped>>;

    fn bytes(chunks: &[Result<&'static str, Dropped>]) -> Sent {
        chunks
            .iter()
            .map(|chunk| chunk.map(|text| Bytes::from_static(text.as_bytes())))
            .collect()
    }

    /// How a failed stream failed: whether it broke, and the status and kind of its failure.
    type Failure = (bool, Option<u16>, FailureKind);

    /// How a stream of `chunks` settled, as its failure when it failed, and the bytes of every
    /// block it hands back.
    async fn settled(chunks: &[Result<&'static str, Dropped>]) -> (Option<Failure>, Sent) {
        // Like a provider's stream, this one panics when it is polled past its end.
        let chunks = stream::unfold(bytes(chunks).into_iter(), |mut left| async {
            left.next().map(|chunk| (chunk, left))
        });
        let chunks = sse::blocks(chunks).boxed();
        let (failure, sent) = match hold(chunks, messages_event, dropped).await {
            Held::Committed(sent) => (None, sent),
            Held::Failed { failed, sent } => (
                Some((
                    failed.cause == Cause::Broke,
                    failed.failure.status,
                    failed.failure.kind,
                )),
                sent,
            ),
        };
        let sent = sent.map(|block| block.map(|block| block.bytes));
        (failure, sent.collect().await)
    }

    #[rstest]
    #[case::content_commits_the_stream(vec![Ok(START), Ok(PING), Ok(BLOCK), Ok(DELTA), Ok(OVERLOADED)], None)]
    #[case::a_stream_that_ends_without_content_goes_out(vec![Ok(START), Ok(PING)], None)]
    #[case::an_error_event_without_a_type_is_no_failure(vec![Ok(START), Ok(UNTYPED), Ok(DELTA)], None)]
    #[case::an_overloaded_provider_fails_the_stream(
        vec![Ok(START), Ok(PING), Ok(OVERLOADED), Ok(DELTA)],
        Some((false, Some(529), FailureKind::InternalServer))
    )]
    #[case::a_break_fails_the_stream(
        vec![Ok(START), Err(Dropped), Ok(DELTA)],
        Some((true, None, FailureKind::ApiConnection))
    )]
    #[tokio::test]
    async fn a_held_stream_settles_on_content_or_on_its_failure_and_is_handed_back_whole(
        #[case] chunks: Vec<Result<&'static str, Dropped>>,
        #[case] failure: Option<Failure>,
    ) {
        assert_eq!(settled(&chunks).await, (failure, bytes(&chunks)));
    }

    /// A transport splits a stream where it likes, and a host frames its events its own way.
    /// The hold reads whole events however their bytes arrive, and hands every byte back.
    #[rstest]
    #[case::an_error_event_split_across_chunks(
        vec![START, &OVERLOADED[..40], &OVERLOADED[40..]],
        Some(529)
    )]
    #[case::an_error_event_with_crlf_line_endings(
        vec![START, "event: error\r\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}\r\n\r\n"],
        Some(529)
    )]
    #[case::a_stream_that_ends_inside_an_event(
        vec![START, "event: content_block_delta\ndata: {\"type\":\"content_bl"],
        None
    )]
    #[tokio::test]
    async fn a_held_stream_is_read_by_whole_events_and_handed_back_byte_for_byte(
        #[case] chunks: Vec<&'static str>,
        #[case] failed_with: Option<u16>,
    ) {
        let whole = chunks.concat();
        let chunks: Vec<_> = chunks.into_iter().map(Ok).collect();

        let (failure, sent) = settled(&chunks).await;

        let sent: Vec<u8> = sent.into_iter().flat_map(Result::unwrap).collect();
        assert_eq!(
            (failure.and_then(|(_, status, _)| status), sent),
            (failed_with, whole.into_bytes())
        );
    }

    #[tokio::test]
    async fn a_stream_that_sends_more_than_is_held_goes_out_unread_past_it() {
        let chunks: Vec<Result<&'static str, Dropped>> =
            std::iter::repeat_n(Ok(PING), MAX_HELD_CHUNKS)
                .chain([Ok(OVERLOADED)])
                .collect();

        let (failure, sent) = settled(&chunks).await;

        assert_eq!((failure, sent.len()), (None, MAX_HELD_CHUNKS + 1));
    }
}
