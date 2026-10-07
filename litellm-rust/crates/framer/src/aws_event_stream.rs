use bytes::{Buf, Bytes, BytesMut};
use futures_util::{Stream, StreamExt};

use aws_smithy_eventstream::frame::read_message_from;
use aws_smithy_types::event_stream::Header;

use crate::{Error, Framer};

const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq)]
pub struct AwsEventStreamFrame {
    pub headers: Vec<Header>,
    pub payload: Bytes,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AwsEventStreamFramer;

impl Framer for AwsEventStreamFramer {
    type Frame = AwsEventStreamFrame;

    fn frame<S, B, E>(self, input: S) -> impl Stream<Item = Result<Self::Frame, Error>> + Send
    where
        S: Stream<Item = Result<B, E>> + Send,
        B: Buf + Send,
        E: std::error::Error + Send + Sync + 'static,
    {
        futures_util::stream::try_unfold(
            (Box::pin(input), BytesMut::new()),
            |(mut input, mut buffer)| async move {
                loop {
                    if buffer.len() >= 4 {
                        let length = (&buffer[..4]).get_u32() as usize;
                        if !(16..=MAX_FRAME_BYTES).contains(&length) {
                            return Err(Error::InvalidLength(length));
                        }
                        if buffer.len() >= length {
                            let raw = buffer.split_to(length).freeze();
                            let message = read_message_from(raw)?;
                            let frame = AwsEventStreamFrame {
                                headers: message.headers().to_vec(),
                                payload: message.payload().clone(),
                            };
                            return Ok(Some((frame, (input, buffer))));
                        }
                    }
                    match input.next().await {
                        Some(Ok(mut chunk)) => {
                            while chunk.has_remaining() {
                                let bytes = chunk.chunk();
                                buffer.extend_from_slice(bytes);
                                let length = bytes.len();
                                chunk.advance(length);
                            }
                        }
                        Some(Err(error)) => return Err(Error::Body(Box::new(error))),
                        None if buffer.is_empty() => return Ok(None),
                        None => return Err(Error::Truncated),
                    }
                }
            },
        )
        .fuse()
    }
}
