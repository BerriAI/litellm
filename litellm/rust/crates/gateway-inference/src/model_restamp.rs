//! The `model` an answer reports: the name the client asked for, in place of the provider's name
//! for the deployment that served it, as the proxy's `_override_openai_response_model` restamps
//! the answer of every route. A complete answer a fallback served keeps the provider's name, and
//! a stream's event is restamped either way.
//!
//! A route restamps its own answer while it is still typed, before it renders it.

use futures_util::{StreamExt, future, stream::BoxStream};
use litellm_framer::sse::SseBlock;
use litellm_host::call::CallOutput;
use litellm_inference_messages::MessagesBody;
use litellm_llms_types::formats::messages::MessagesResponse;
use litellm_router_core::RouterResponse;
use serde_json::Value;

/// A complete answer that names the model that served it.
pub(crate) trait NamesModel {
    /// The answer naming `model`. One that names no model stays as it is, as
    /// `_override_openai_response_model` leaves it.
    fn naming(self, model: &str) -> Self;
}

impl NamesModel for MessagesBody {
    fn naming(self, model: &str) -> Self {
        match self {
            Self::Message(message) => Self::Message(Box::new(MessagesResponse {
                model: model.to_owned(),
                ..*message
            })),
            Self::Unchecked(fields) => Self::Unchecked(
                fields
                    .into_iter()
                    .map(|(name, value)| match name.as_str() {
                        "model" => (name, Value::from(model)),
                        _ => (name, value),
                    })
                    .collect(),
            ),
        }
    }
}

/// What one event of a stream says about the model the stream reports.
pub(crate) enum Stamp {
    /// Not the event that names the model, and one may still come: a keepalive.
    Later,
    /// The stream names no model from here on, or already names the right one.
    Never,
    /// The event's data, naming the model it was asked to.
    Now(String),
}

/// Reads the data of one event of a route's stream, given the model the stream has to report.
pub(crate) type StampEvent = fn(&str, &str) -> Stamp;

/// The model the answer of one routed call reports.
pub(crate) struct ReportedModel<'a> {
    requested: &'a str,
    fell_back: bool,
}

impl<'a> ReportedModel<'a> {
    /// For the name the client asked for and the call that served it.
    pub(crate) fn of<R>(requested: &'a str, routed: &RouterResponse<R>) -> Self {
        Self {
            requested,
            fell_back: routed.attempted_fallbacks > 0,
        }
    }

    /// `answer` naming the model. A complete answer names it unless a fallback served it. A
    /// stream names it in the event `stamp` picks, whether or not a fallback served it.
    pub(crate) fn answer<A: NamesModel, H, E: Send + 'static>(
        &self,
        answer: CallOutput<A, H, SseBlock, E>,
        stamp: StampEvent,
    ) -> CallOutput<A, H, SseBlock, E> {
        match answer {
            CallOutput::Complete(answer) if self.fell_back => CallOutput::Complete(answer),
            CallOutput::Complete(answer) => CallOutput::Complete(answer.naming(self.requested)),
            CallOutput::Stream { head, chunks } => CallOutput::Stream {
                head,
                chunks: stamped(chunks, self.requested.to_owned(), stamp),
            },
        }
    }
}

/// `blocks` with the event `stamp` picks naming `requested`. Every block after the event that
/// settles it passes through as it is.
fn stamped<E: Send + 'static>(
    blocks: BoxStream<'static, Result<SseBlock, E>>,
    requested: String,
    stamp: StampEvent,
) -> BoxStream<'static, Result<SseBlock, E>> {
    blocks
        .scan(false, move |settled, block| {
            let sent = match block {
                Ok(block) if !*settled => {
                    let (block, now_settled) = read(block, &requested, stamp);
                    *settled = now_settled;
                    Ok(block)
                }
                block => block,
            };
            future::ready(Some(sent))
        })
        .boxed()
}

/// `block` as `stamp` leaves it, and whether it settled which model the stream reports. Only the
/// data of an event `stamp` renames is written again.
fn read(block: SseBlock, requested: &str, stamp: StampEvent) -> (SseBlock, bool) {
    let Some(event) = &block.event else {
        return (block, false);
    };
    match stamp(&event.data, requested) {
        Stamp::Later => (block, false),
        Stamp::Never => (block, true),
        Stamp::Now(data) => (block.with_data(&data), true),
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use futures_util::stream;
    use litellm_framer::sse;
    use litellm_router_core::{Served, call::StreamOrigin};
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    const COMMENT: &str = ": keepalive\n\n";
    const PING: &str = "event: ping\ndata: {\"type\":\"ping\"}\n\n";
    const START: &str = "event: start\ndata: {\"type\":\"start\",\"model\":\"provider-model\"}\n\n";
    const RESTAMPED: &str = "event: start\ndata: {\"type\":\"start\",\"model\":\"public\"}\n\n";
    const DELTA: &str = "event: delta\ndata: {\"type\":\"delta\",\"text\":\"provider-model\"}\n\n";
    const START_AMONG_MORE: &str = ": note\r\nevent: start\r\nid: 7\r\ndata: {\"type\":\"start\",\"model\":\"provider-model\"}\r\n\r\n";
    const RESTAMPED_AMONG_MORE: &str = ": note\r\nevent: start\r\nid: 7\r\ndata: {\"type\":\"start\",\"model\":\"public\"}\r\n\r\n";

    /// A route whose `start` event names the model and whose `ping` is a keepalive.
    fn stamp(data: &str, model: &str) -> Stamp {
        match serde_json::from_str::<Value>(data).unwrap()["type"].as_str() {
            Some("ping") => Stamp::Later,
            Some("start") => Stamp::Now(data.replace("provider-model", model)),
            _ => Stamp::Never,
        }
    }

    fn reported(attempted_fallbacks: u32) -> RouterResponse<()> {
        RouterResponse {
            output: Served::new((), 0),
            deployment_id: "d1".to_owned(),
            name: "public".to_owned(),
            model_group: "public".to_owned(),
            attempted_retries: 0,
            max_retries: None,
            attempted_fallbacks,
            origin: StreamOrigin::default(),
        }
    }

    #[rstest]
    #[case::served_by_its_own_group(
        0,
        json!({"id": "1", "model": "provider-model", "usage": {"input_tokens": 7}}),
        json!({"id": "1", "model": "public", "usage": {"input_tokens": 7}})
    )]
    #[case::served_by_a_fallback(
        1,
        json!({"id": "1", "model": "provider-model"}),
        json!({"id": "1", "model": "provider-model"})
    )]
    #[case::an_answer_that_names_no_model(0, json!({"id": "1"}), json!({"id": "1"}))]
    fn a_complete_answer_names_the_requested_model_unless_a_fallback_served_it(
        #[case] fallbacks: u32,
        #[case] sent: Value,
        #[case] expected: Value,
    ) {
        let answer: CallOutput<MessagesBody, (), SseBlock, Broke> =
            CallOutput::Complete(MessagesBody::Unchecked(sent.as_object().cloned().unwrap()));

        let answered = ReportedModel::of("public", &reported(fallbacks)).answer(answer, stamp);

        let CallOutput::Complete(answered) = answered else {
            panic!("a complete answer came back as a stream");
        };
        assert_eq!(serde_json::to_value(answered).unwrap(), expected);
    }

    #[derive(Debug, PartialEq)]
    struct Broke;

    #[rstest]
    #[case::the_event_that_names_it(vec![Ok(START), Ok(DELTA)], vec![RESTAMPED, DELTA])]
    #[case::after_keepalives(vec![Ok(COMMENT), Ok(PING), Ok(START)], vec![COMMENT, PING, RESTAMPED])]
    #[case::only_the_first_event_that_is_not_a_keepalive(
        vec![Ok(DELTA), Ok(START)],
        vec![DELTA, START]
    )]
    #[case::only_once(vec![Ok(START), Ok(START)], vec![RESTAMPED, START])]
    #[case::every_other_byte_of_its_block_stays(vec![Ok(START_AMONG_MORE)], vec![RESTAMPED_AMONG_MORE])]
    #[case::a_break_passes_through(vec![Err(Broke), Ok(START)], vec!["<broke>", RESTAMPED])]
    #[tokio::test]
    async fn a_stream_names_the_requested_model_in_the_event_its_route_picks(
        #[case] chunks: Vec<Result<&'static str, Broke>>,
        #[case] expected: Vec<&str>,
    ) {
        let chunks = stream::iter(chunks).map(|chunk| chunk.map(Bytes::from));

        let sent: Vec<String> = stamped(sse::blocks(chunks).boxed(), "public".to_owned(), stamp)
            .map(|block| match block {
                Ok(block) => String::from_utf8(block.bytes.to_vec()).unwrap(),
                Err(Broke) => "<broke>".to_owned(),
            })
            .collect()
            .await;

        assert_eq!(sent, expected);
    }
}
