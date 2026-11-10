use aws_smithy_eventstream::frame::{read_message_from, write_message_to};
pub use aws_smithy_types::event_stream::{Header, HeaderValue, Message};
use bytes::BytesMut;
use tokio_util::codec::{Decoder, Encoder};

use crate::EventStreamError;

const MIN_FRAME_BYTES: usize = 16;
const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Default)]
pub struct AwsEventStreamCodec;

impl Decoder for AwsEventStreamCodec {
    type Item = Message;
    type Error = EventStreamError;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Message>, EventStreamError> {
        let Some(prefix) = src.first_chunk::<4>() else {
            return Ok(None);
        };
        let length = u32::from_be_bytes(*prefix) as usize;
        if !(MIN_FRAME_BYTES..=MAX_FRAME_BYTES).contains(&length) {
            return Err(EventStreamError::InvalidLength(length));
        }
        if src.len() < length {
            return Ok(None);
        }
        Ok(Some(read_message_from(src.split_to(length).freeze())?))
    }

    fn decode_eof(&mut self, src: &mut BytesMut) -> Result<Option<Message>, EventStreamError> {
        match self.decode(src)? {
            Some(message) => Ok(Some(message)),
            None if src.is_empty() => Ok(None),
            None => Err(EventStreamError::Truncated),
        }
    }
}

impl Encoder<Message> for AwsEventStreamCodec {
    type Error = EventStreamError;

    fn encode(&mut self, message: Message, dst: &mut BytesMut) -> Result<(), EventStreamError> {
        Ok(write_message_to(&message, dst)?)
    }
}
