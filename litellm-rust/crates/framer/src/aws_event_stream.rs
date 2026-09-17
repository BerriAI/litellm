use aws_smithy_eventstream::frame::read_message_from;
pub use aws_smithy_types::event_stream::{Header, HeaderValue, Message};
use bytes::{Buf, BufMut, BytesMut};
use futures_util::{Stream, StreamExt};

use crate::{Error, MAX_FRAME_BYTES};

const MIN_FRAME_BYTES: usize = 16;

pub fn frames<S, B, E>(input: S) -> impl Stream<Item = Result<Message, Error>> + Send
where
    S: Stream<Item = Result<B, E>> + Send,
    B: Buf + Send,
    E: std::error::Error + Send + Sync + 'static,
{
    futures_util::stream::try_unfold(
        (Box::pin(input), BytesMut::new()),
        |(mut input, mut buffer)| async move {
            loop {
                if let Some(length) = frame_length(&buffer)?
                    && buffer.len() >= length
                {
                    let message = read_message_from(&buffer[..length])?;
                    buffer.advance(length);
                    return Ok(Some((message, (input, buffer))));
                }
                match input.next().await {
                    Some(Ok(chunk)) => buffer.put(chunk),
                    Some(Err(error)) => return Err(Error::Body(Box::new(error))),
                    None if buffer.is_empty() => return Ok(None),
                    None => return Err(Error::Truncated),
                }
            }
        },
    )
}

fn frame_length(buffer: &[u8]) -> Result<Option<usize>, Error> {
    let Some(prefix) = buffer.first_chunk::<4>() else {
        return Ok(None);
    };
    match u32::from_be_bytes(*prefix) as usize {
        length @ MIN_FRAME_BYTES..=MAX_FRAME_BYTES => Ok(Some(length)),
        length @ ..MIN_FRAME_BYTES => Err(Error::InvalidLength(length)),
        _ => Err(Error::FrameTooLarge),
    }
}
