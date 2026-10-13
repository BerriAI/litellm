//! SSE decoding and encoding shared by every format's `streaming` module.
//!
//! An event type the format adds later is not malformed: each format's event type keeps it in
//! `Other` and it re-encodes unchanged.
//!
//! DIVERGENCE: decoding is strict. An event whose `data` is not a valid event of the format is
//! `InvalidResponse` and ends the stream, and so is `[DONE]` on a Messages stream. Python
//! never fails a stream on a malformed event:
//! - Chat Completions (`BaseModelResponseIterator`) and Responses
//!   (`BaseResponsesAPIStreamingIterator`) skip the event and keep going
//! - Messages (`BaseAnthropicMessagesStreamingIterator`) forwards the provider's bytes to the
//!   client without parsing them
//! - Anthropic to Chat (`anthropic/chat/handler.py` `ModelResponseIterator`) buffers the invalid
//!   lines on the guess that one payload was split across them
//!
//! Skipping loses content or a tool-call fragment with nothing to show for it, so Rust reports it.

use bytes::Bytes;
use futures_util::{StreamExt, TryStreamExt, future::ready, stream::BoxStream};
use litellm_framer::{
    frames,
    sse::{SseCodec, SseEvent, decode_all},
};
use serde::{Serialize, de::DeserializeOwned};

use crate::{Error, ErrorDetail, base_llm::base_model_iterator::ByteStream};

pub type EventStream<E> = BoxStream<'static, Result<E, Error>>;

/// Which frames are control frames rather than events of the format.
pub type IsControl = fn(&SseEvent) -> bool;

/// OpenAI's end-of-stream marker on Chat Completions and Responses streams. It closes the
/// stream and is not an event; Anthropic Messages streams never carry it.
pub const DONE_FRAME: &[u8] = b"data: [DONE]\n\n";

pub fn is_done(event: &SseEvent) -> bool {
    event.data.trim() == "[DONE]"
}

pub fn event_stream<E: DeserializeOwned + Send + 'static>(
    bytes: ByteStream,
    is_control: IsControl,
) -> EventStream<E> {
    frames(bytes, SseCodec::default())
        .map_err(framing)
        .try_filter(move |event| ready(!is_control(event)))
        .and_then(|event| ready(parse(&event)))
        .boxed()
}

/// Decodes a whole stream at once. An event the stream ends inside is dropped.
pub fn decode<E: DeserializeOwned>(wire: &[u8], is_control: IsControl) -> Result<Vec<E>, Error> {
    decode_all(wire)
        .map_err(framing)?
        .iter()
        .filter(|event| !is_control(event))
        .map(parse)
        .collect()
}

/// A bare `data:` frame, which is what LiteLLM's proxy sends to clients on OpenAI formats.
pub fn encode_data<E: Serialize>(event: &E) -> Result<Bytes, Error> {
    let data = serde_json::to_string(event).map_err(invalid)?;
    Ok(Bytes::from(format!("data: {data}\n\n")))
}

fn parse<E: DeserializeOwned>(event: &SseEvent) -> Result<E, Error> {
    serde_json::from_str(&event.data).map_err(invalid)
}

pub fn invalid(error: serde_json::Error) -> Error {
    Error::InvalidResponse(ErrorDetail::invalid("stream event", error))
}

fn framing(error: litellm_framer::SseError) -> Error {
    Error::InvalidResponse(ErrorDetail::failed("stream framing", error))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::exact("[DONE]", true)]
    #[case::padded(" [DONE] ", true)]
    #[case::json("{\"type\":\"x\"}", false)]
    #[case::mentions("[DONE] was sent", false)]
    #[case::empty("", false)]
    fn done_is_an_exact_marker(#[case] data: &str, #[case] done: bool) {
        let event = SseEvent {
            data: data.into(),
            ..SseEvent::default()
        };
        assert_eq!(is_done(&event), done);
    }

    #[rstest]
    fn the_done_frame_we_send_decodes_as_a_control_frame() {
        let events: Vec<serde_json::Value> = decode(DONE_FRAME, is_done).unwrap();
        assert!(events.is_empty());
    }
}
