use futures_util::{Stream, StreamExt};

use crate::{Error, Framer};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SseFrame {
    pub event: Option<String>,
    pub data: Option<String>,
    pub id: Option<String>,
    pub retry: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SseFramer;

impl Framer for SseFramer {
    type Frame = SseFrame;

    fn frame<S, B, E>(self, input: S) -> impl Stream<Item = Result<SseFrame, Error>> + Send
    where
        S: Stream<Item = Result<B, E>> + Send,
        B: bytes::Buf + Send,
        E: std::error::Error + Send + Sync + 'static,
    {
        let frames = Box::pin(sse_stream::SseStream::from_bytes_stream(input));
        futures_util::stream::try_unfold(frames, |mut frames| async move {
            let Some(frame) = frames.next().await else {
                return Ok(None);
            };
            let frame = frame?;
            Ok(Some((
                SseFrame {
                    event: frame.event,
                    data: frame.data,
                    id: frame.id,
                    retry: frame.retry,
                },
                frames,
            )))
        })
        .fuse()
    }
}
