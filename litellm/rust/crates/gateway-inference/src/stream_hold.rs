//! A stream held back while another attempt can still answer in its place. Until a stream sends
//! generated content its client has nothing to lose, so its blocks are kept instead of sent on,
//! and the stream settles one of two ways: it commits and every block goes out as the provider
//! sent it, or it fails first and `routing::call` decides what answers the request, as Python's
//! `wrap_anthropic_stream` does, except that Python also sends `message_start` on at once while only
//! a retry can take over and the request has no message `thread`.
//!
//! [`hold`] waits for the stream's next block for as long as the stream takes to send it. The
//! attempt's own timeout, which bounds the whole upstream response, is what ends a stream that
//! sends nothing.

use std::time::SystemTime;

use futures_util::{
    StreamExt,
    stream::{self, BoxStream, Fuse},
};
use litellm_framer::sse::SseBlock;

use crate::stream_event::{Kind, Read};

/// How much of a stream is held while it has sent no content.
#[derive(Clone, Copy)]
pub(crate) struct Budget {
    pub blocks: usize,
    pub bytes: usize,
}

/// `blocks` is `MAX_BUFFERED_PRE_CONTENT_ANTHROPIC_CHUNKS` in
/// `litellm/router_rust/invoke/streaming.py`. Python bounds no bytes, so a provider that sends
/// large events before any content grows its buffer without limit.
pub(crate) const HELD: Budget = Budget {
    blocks: 200,
    bytes: 1 << 20,
};

/// A stream as the SSE blocks it was read in.
pub type Chunks<E> = BoxStream<'static, Result<SseBlock, E>>;

/// How a stream failed before any content.
pub(crate) enum Cause<E> {
    /// The provider sent its error as an event of a stream that otherwise went well. `status`
    /// and `answer` are what the provider answers a request with when it fails the same way
    /// before any stream, and `sent` is the event as it came.
    ErrorEvent {
        status: u16,
        answer: String,
        sent: SseBlock,
    },
    /// The stream broke under the reader, with the error it broke on.
    Broke(E),
}

/// A stream that failed before any content: what it sent first, how it failed, and what is left
/// of it to read.
pub(crate) struct Failed<E> {
    before: Vec<SseBlock>,
    pub cause: Cause<E>,
    rest: Fuse<Chunks<E>>,
}

impl<E: Send + 'static> Failed<E> {
    /// The stream from its first byte, its failure where it came.
    pub(crate) fn replayed(self) -> Chunks<E> {
        let failure = match self.cause {
            Cause::ErrorEvent { sent, .. } => Ok(sent),
            Cause::Broke(error) => Err(error),
        };
        let held = self.before.into_iter().map(Ok).chain([failure]);
        stream::iter(held).chain(self.rest).boxed()
    }

    /// How the stream failed, with the stream itself closed.
    pub(crate) fn closed(self) -> Cause<E> {
        self.cause
    }
}

/// How a held stream settled.
pub(crate) enum Held<E> {
    /// Content began, the stream ended, or it sent more than is held: the stream from its first
    /// byte, which nothing may answer in place of any more, and when its first event arrived.
    Committed(Chunks<E>, Option<SystemTime>),
    Failed(Failed<E>),
}

/// Hold `chunks` until the stream settles: until it sends content, fails, ends, or has sent all
/// that `budget` holds. `read` reads the events of the stream's format.
pub(crate) async fn hold<E: Send + 'static>(
    chunks: Chunks<E>,
    read: Read,
    budget: Budget,
) -> Held<E> {
    // A stream read to its end is handed back too, and may not be polled past it.
    let mut rest = chunks.fuse();
    let mut before = Vec::new();
    let mut bytes = 0;
    let mut first_event = None;
    while before.len() < budget.blocks && bytes < budget.bytes {
        let block = match rest.next().await {
            None => break,
            Some(Ok(block)) => block,
            Some(Err(error)) => {
                let cause = Cause::Broke(error);
                return Held::Failed(Failed {
                    before,
                    cause,
                    rest,
                });
            }
        };
        if block.event.is_some() {
            first_event.get_or_insert_with(SystemTime::now);
        }
        match block.event.as_ref().map(|event| read(&event.data).kind) {
            Some(Kind::Failed { status, answer }) => {
                let cause = Cause::ErrorEvent {
                    status,
                    answer,
                    sent: block,
                };
                return Held::Failed(Failed {
                    before,
                    cause,
                    rest,
                });
            }
            Some(Kind::Content) => {
                before.push(block);
                break;
            }
            Some(Kind::Lifecycle) | None => {
                bytes += block.bytes.len();
                before.push(block);
            }
        }
    }
    let sent = stream::iter(before.into_iter().map(Ok)).chain(rest).boxed();
    Held::Committed(sent, first_event)
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use litellm_framer::sse;
    use rstest::rstest;

    use super::*;
    use crate::messages::read_event;

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

    type Sent = Vec<Result<Bytes, Dropped>>;

    fn bytes(chunks: &[Result<&'static str, Dropped>]) -> Sent {
        chunks
            .iter()
            .map(|chunk| chunk.map(|text| Bytes::from_static(text.as_bytes())))
            .collect()
    }

    /// How a failed stream failed: the status its error event stands for, or that it broke.
    #[derive(Debug, PartialEq)]
    enum Failure {
        ErrorEvent(u16),
        Broke,
    }

    /// How a stream of `chunks` settled under `budget`, as its failure when it failed, and the
    /// bytes of every block it hands back.
    async fn settled(
        chunks: &[Result<&'static str, Dropped>],
        budget: Budget,
    ) -> (Option<Failure>, Sent) {
        // Like a provider's stream, this one panics when it is polled past its end.
        let chunks = stream::unfold(bytes(chunks).into_iter(), |mut left| async {
            left.next().map(|chunk| (chunk, left))
        });
        let chunks = sse::blocks(chunks).boxed();
        let (failure, sent) = match hold(chunks, read_event, budget).await {
            Held::Committed(sent, _) => (None, sent),
            Held::Failed(failed) => {
                let failure = match &failed.cause {
                    Cause::ErrorEvent { status, .. } => Failure::ErrorEvent(*status),
                    Cause::Broke(Dropped) => Failure::Broke,
                };
                (Some(failure), failed.replayed())
            }
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
        Some(Failure::ErrorEvent(529))
    )]
    #[case::a_break_fails_the_stream(vec![Ok(START), Err(Dropped), Ok(DELTA)], Some(Failure::Broke))]
    #[tokio::test]
    async fn a_held_stream_settles_on_content_or_on_its_failure_and_is_handed_back_whole(
        #[case] chunks: Vec<Result<&'static str, Dropped>>,
        #[case] failure: Option<Failure>,
    ) {
        assert_eq!(settled(&chunks, HELD).await, (failure, bytes(&chunks)));
    }

    /// A transport splits a stream where it likes, and a host frames its events its own way.
    /// The hold reads whole events however their bytes arrive, and hands every byte back.
    #[rstest]
    #[case::an_error_event_split_across_chunks(
        vec![START, &OVERLOADED[..40], &OVERLOADED[40..]],
        Some(Failure::ErrorEvent(529))
    )]
    #[case::an_error_event_with_crlf_line_endings(
        vec![START, "event: error\r\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}\r\n\r\n"],
        Some(Failure::ErrorEvent(529))
    )]
    #[case::a_stream_that_ends_inside_an_event(
        vec![START, "event: content_block_delta\ndata: {\"type\":\"content_bl"],
        None
    )]
    #[tokio::test]
    async fn a_held_stream_is_read_by_whole_events_and_handed_back_byte_for_byte(
        #[case] chunks: Vec<&'static str>,
        #[case] failure: Option<Failure>,
    ) {
        let whole = chunks.concat();
        let chunks: Vec<_> = chunks.into_iter().map(Ok).collect();

        let (failed, sent) = settled(&chunks, HELD).await;

        let sent: Vec<u8> = sent.into_iter().flat_map(Result::unwrap).collect();
        assert_eq!((failed, sent), (failure, whole.into_bytes()));
    }

    /// Two pings fill either budget, so the overloaded error after them is never read.
    #[rstest]
    #[case::more_blocks_than_are_held(Budget { blocks: 2, bytes: usize::MAX })]
    #[case::more_bytes_than_are_held(Budget { blocks: usize::MAX, bytes: PING.len() + 1 })]
    #[tokio::test]
    async fn a_stream_that_sends_more_than_is_held_goes_out_unread_past_it(#[case] budget: Budget) {
        let chunks = [Ok(PING), Ok(PING), Ok(OVERLOADED)];

        assert_eq!(settled(&chunks, budget).await, (None, bytes(&chunks)));
    }

    /// A held stream reports when its first event arrived, not when it is replayed, and a
    /// comment that carries no event is no first event.
    #[rstest]
    #[case::an_event(vec![Ok(START), Ok(PING)], true)]
    #[case::only_a_comment(vec![Ok(": keepalive\n\n")], false)]
    #[tokio::test]
    async fn a_held_stream_tells_when_its_first_event_arrived(
        #[case] chunks: Vec<Result<&'static str, Dropped>>,
        #[case] timed: bool,
    ) {
        let before = SystemTime::now();
        let blocks = sse::blocks(stream::iter(bytes(&chunks))).boxed();
        let Held::Committed(_, first_event) = hold(blocks, read_event, HELD).await else {
            panic!("the stream failed");
        };
        assert_eq!(
            first_event.map(|at| at >= before && at <= SystemTime::now()),
            timed.then_some(true)
        );
    }
}
