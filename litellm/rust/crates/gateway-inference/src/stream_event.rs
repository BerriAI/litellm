//! What one event of a stream says, read once per format. A streaming route names the reader of
//! its format, and the shared stream code asks that reader instead of reading the format itself:
//! `stream_outcome` for the tokens and the failure it tells the router, `stream_hold` for whether
//! the stream has begun to answer.

use litellm_llms_types::formats::{
    messages::streaming::{MessagesStreamEvent, MessagesStreamUsage},
    responses::streaming::ResponsesEventType,
};
use serde::Deserialize;

/// The status of a Responses error event, which names none.
const RESPONSES_ERROR_STATUS: u16 = 500;

/// The tokens an event's `usage` counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize)]
pub struct Counts {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    total_tokens: Option<u64>,
}

impl Counts {
    /// These counts over `earlier` ones, a count this event does not carry kept from them.
    pub(crate) fn over(self, earlier: Self) -> Self {
        Self {
            input_tokens: self.input_tokens.or(earlier.input_tokens),
            output_tokens: self.output_tokens.or(earlier.output_tokens),
            total_tokens: self.total_tokens.or(earlier.total_tokens),
        }
    }

    /// `total_tokens`, else the input plus the output, as [`crate::routing::total_tokens`]
    /// counts a complete response.
    pub(crate) fn total(self) -> u64 {
        self.total_tokens
            .unwrap_or_else(|| self.input_tokens.unwrap_or(0) + self.output_tokens.unwrap_or(0))
    }
}

impl From<MessagesStreamUsage> for Counts {
    fn from(usage: MessagesStreamUsage) -> Self {
        Self {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: None,
        }
    }
}

/// What an event says about how far its stream has come.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Nothing a client would miss if another attempt answered instead.
    Lifecycle,
    /// Generated output: the stream is the answer from here on.
    Content,
    /// The provider's error. `status` and `answer` are what the provider answers a request with
    /// when it fails the same way before any stream.
    Failed { status: u16, answer: String },
}

/// What one event of a stream says.
#[derive(Clone, Debug, PartialEq)]
pub struct Reading {
    pub kind: Kind,
    pub usage: Option<Counts>,
}

impl Reading {
    fn lifecycle(usage: Option<impl Into<Counts>>) -> Self {
        Self {
            kind: Kind::Lifecycle,
            usage: usage.map(Into::into),
        }
    }

    fn of(kind: Kind) -> Self {
        Self { kind, usage: None }
    }
}

/// Reads the data of one event of a format's stream.
pub type Read = fn(&str) -> Reading;

/// An Anthropic Messages event. `message_start` counts the input and `message_delta` the output,
/// only a `content_block_delta` carries generated output (`is_anthropic_content_delta_chunk`),
/// and an `error` event is answered as its error type is before a stream, with the event's own
/// data as the body (`parse_anthropic_error_event`).
pub(crate) fn messages_event(data: &str) -> Reading {
    match serde_json::from_str(data) {
        Ok(MessagesStreamEvent::MessageStart { message }) => Reading::lifecycle(message.usage),
        Ok(
            MessagesStreamEvent::MessageDelta { usage, .. }
            | MessagesStreamEvent::MessageStop { usage, .. },
        ) => Reading::lifecycle(usage),
        Ok(MessagesStreamEvent::ContentBlockDelta { .. }) => Reading::of(Kind::Content),
        Ok(MessagesStreamEvent::Error { error }) => Reading::of(Kind::Failed {
            status: error.error_type.status(),
            answer: data.to_owned(),
        }),
        Ok(
            MessagesStreamEvent::ContentBlockStart { .. }
            | MessagesStreamEvent::ContentBlockStop { .. }
            | MessagesStreamEvent::Ping
            | MessagesStreamEvent::Other(_),
        )
        | Err(_) => Reading::of(Kind::Lifecycle),
    }
}

/// The part of a Responses event that carries its counts.
#[derive(Default, Deserialize)]
struct Counted {
    usage: Option<Counts>,
}

#[derive(Deserialize)]
struct ResponsesEvent {
    #[serde(rename = "type")]
    kind: ResponsesEventType,
    response: Option<Counted>,
}

/// An OpenAI Responses event: the closing event carries the response and its usage. Responses
/// streams are not held, so no event of one reads as content yet.
pub(crate) fn responses_event(data: &str) -> Reading {
    let Ok(event) = serde_json::from_str::<ResponsesEvent>(data) else {
        return Reading::of(Kind::Lifecycle);
    };
    match event.kind {
        ResponsesEventType::ResponseCompleted | ResponsesEventType::ResponseIncomplete => {
            Reading::lifecycle(event.response.unwrap_or_default().usage)
        }
        ResponsesEventType::ResponseFailed | ResponsesEventType::Error => {
            Reading::of(Kind::Failed {
                status: RESPONSES_ERROR_STATUS,
                answer: data.to_owned(),
            })
        }
        _ => Reading::of(Kind::Lifecycle),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// A provider's stream goes out as it was sent, so the reader takes events Anthropic itself
    /// would never send: a strict reading would hold back content or lose the usage they carry.
    #[rstest]
    #[case::a_start_counts_its_input(
        r#"{"type":"message_start","message":{"usage":{"input_tokens":7}}}"#,
        Kind::Lifecycle,
        Some(7)
    )]
    #[case::a_closing_delta_counts_its_output(
        r#"{"type":"message_delta","delta":{},"usage":{"output_tokens":5}}"#,
        Kind::Lifecycle,
        Some(5)
    )]
    #[case::a_ping_says_nothing(r#"{"type":"ping"}"#, Kind::Lifecycle, None)]
    #[case::a_block_opening_is_not_content_yet(
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        Kind::Lifecycle,
        None
    )]
    #[case::a_text_delta_is_content(
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#,
        Kind::Content,
        None
    )]
    #[case::a_delta_of_a_type_not_named_yet_is_content(
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"future_delta","payload":1}}"#,
        Kind::Content,
        None
    )]
    #[case::a_known_delta_with_a_field_of_another_shape_is_content(
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":["hi"]}}"#,
        Kind::Content,
        None
    )]
    #[case::an_error_without_a_type_is_no_failure(r#"{"type":"error"}"#, Kind::Lifecycle, None)]
    #[case::data_that_is_no_event_says_nothing("[DONE]", Kind::Lifecycle, None)]
    fn a_messages_event_that_is_no_error_says_whether_content_began_and_what_it_counts(
        #[case] data: &str,
        #[case] kind: Kind,
        #[case] total_tokens: Option<u64>,
    ) {
        let reading = messages_event(data);

        assert_eq!(
            (reading.kind, reading.usage.map(Counts::total)),
            (kind, total_tokens)
        );
    }

    #[rstest]
    #[case::overloaded(
        r#"{"type":"error","error":{"type":"overloaded_error","message":"busy"}}"#,
        529
    )]
    #[case::without_a_message(r#"{"type":"error","error":{"type":"overloaded_error"}}"#, 529)]
    #[case::refused(
        r#"{"type":"error","error":{"type":"invalid_request_error","message":"no"}}"#,
        400
    )]
    #[case::of_a_type_not_named_yet(
        r#"{"type":"error","error":{"type":"future_error","message":"busy"}}"#,
        500
    )]
    fn a_messages_error_event_is_answered_as_its_type_is_before_a_stream_with_its_own_data(
        #[case] data: &str,
        #[case] status: u16,
    ) {
        assert_eq!(
            messages_event(data),
            Reading::of(Kind::Failed {
                status,
                answer: data.to_owned()
            })
        );
    }
}
