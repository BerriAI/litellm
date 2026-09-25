use aws_smithy_eventstream::frame::write_message_to;
use aws_smithy_types::event_stream::{Header, HeaderValue, Message};
use bytes::Bytes;

pub fn encode(payload: &'static [u8]) -> Vec<u8> {
    let message = Message::new(Bytes::from_static(payload))
        .add_header(Header::new(
            ":event-type",
            HeaderValue::String("payload".into()),
        ))
        .add_header(Header::new("sequence", HeaderValue::Int32(7)));
    let mut bytes = Vec::new();
    write_message_to(&message, &mut bytes).unwrap();
    bytes
}
