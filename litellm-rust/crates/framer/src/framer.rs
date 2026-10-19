use futures_util::Stream;

use crate::Error;

pub trait Framer: Send {
    type Frame: Send;

    fn frame<S, B, E>(self, input: S) -> impl Stream<Item = Result<Self::Frame, Error>> + Send
    where
        S: Stream<Item = Result<B, E>> + Send,
        B: bytes::Buf + Send,
        E: std::error::Error + Send + Sync + 'static;
}
