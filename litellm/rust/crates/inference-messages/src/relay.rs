use bytes::Bytes;
use futures_util::{StreamExt, TryStreamExt, stream::BoxStream};
use litellm_host::protocol::UnobservedTail;
use litellm_python_compat::{Value, json};

use super::Error;

pub(crate) const INCOMPLETE_STREAM_ERROR_MESSAGE: &str = "Provider stream ended before emitting a message_stop event; the response is incomplete and any partial content (e.g. tool_use input JSON) may be truncated.";

type Frames = BoxStream<'static, Result<Bytes, Error>>;

/// `_incomplete_stream_error_sse_event`.
fn incomplete_stream_error_event() -> Bytes {
    Bytes::from(format!(
        "event: error\ndata: {{\"type\": \"error\", \"error\": {{\"type\": \"api_error\", \"message\": \"{INCOMPLETE_STREAM_ERROR_MESSAGE}\"}}}}\n\n"
    ))
}

/// `bytes.splitlines()`.
fn lines(chunk: &[u8]) -> impl Iterator<Item = &[u8]> {
    let mut rest = Some(chunk);
    std::iter::from_fn(move || {
        let current = rest.filter(|current| !current.is_empty())?;
        match current
            .iter()
            .position(|byte| matches!(byte, b'\n' | b'\r'))
        {
            Some(at) => {
                let skip = if current[at..].starts_with(b"\r\n") {
                    2
                } else {
                    1
                };
                rest = Some(&current[at + skip..]);
                Some(&current[..at])
            }
            None => {
                rest = None;
                Some(current)
            }
        }
    })
}

fn field<'a>(fields: &'a [(Value, Value)], name: &str) -> Option<&'a Value> {
    fields
        .iter()
        .find(|(key, _)| matches!(key, Value::Str(key) if key == name))
        .map(|(_, value)| value)
}

/// `_decoded_sse_data_line`.
fn decoded_data_line(line: &[u8]) -> Option<Value> {
    let data = line.strip_prefix(b"data:")?.trim_ascii();
    json::loads(std::str::from_utf8(data).ok()?).ok()
}

/// `is_message_stop_chunk` for a chunk of SSE bytes.
fn is_message_stop_chunk(chunk: &[u8]) -> bool {
    lines(chunk).any(|line| line == b"event: message_stop")
}

/// `is_provider_error_chunk` for a chunk of SSE bytes.
fn is_provider_error_chunk(chunk: &[u8]) -> bool {
    lines(chunk)
        .filter_map(decoded_data_line)
        .find_map(|candidate| match candidate {
            Value::Dict(fields)
                if matches!(field(&fields, "type"), Some(Value::Str(kind)) if kind == "error") =>
            {
                Some(fields)
            }
            _ => None,
        })
        .is_some_and(|fields| matches!(field(&fields, "error"), Some(Value::Dict(_))))
}

/// `_is_terminal_stream_chunk`.
fn is_terminal_stream_chunk(chunk: &[u8]) -> bool {
    is_message_stop_chunk(chunk) || is_provider_error_chunk(chunk)
}

struct Relay {
    frames: Option<Frames>,
    terminal: bool,
    tail: UnobservedTail,
}

/// A decoded provider stream as `BaseAnthropicMessagesStreamingIterator.async_sse_wrapper`
/// relays it: one that ends without a `message_stop` or provider error event closes with an
/// `api_error` event. Python adds that event after billing, so it goes out on `tail`, which the
/// stream hooks never observe.
pub(crate) fn relayed(frames: Frames, tail: UnobservedTail) -> Frames {
    let relay = Relay {
        frames: Some(frames),
        terminal: false,
        tail,
    };
    futures_util::stream::try_unfold(relay, |mut relay| async move {
        let Some(frames) = relay.frames.as_mut() else {
            return Ok(None);
        };
        if let Some(frame) = frames.try_next().await? {
            relay.terminal |= is_terminal_stream_chunk(&frame);
            return Ok(Some((frame, relay)));
        }
        relay.frames = None;
        if relay.terminal {
            return Ok(None);
        }
        relay.tail.begin();
        Ok(Some((incomplete_stream_error_event(), relay)))
    })
    .boxed()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const START: &str = "event: message_start\ndata: {\"type\": \"message_start\"}\n\n";
    const STOP: &str = "event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n";
    const PROVIDER_ERROR: &str = "event: error\ndata: {\"type\": \"error\", \"error\": {\"type\": \"overloaded_error\"}}\n\n";
    const UNSHAPED_ERROR: &str =
        "event: error\ndata: {\"type\": \"error\", \"error\": \"overloaded\"}\n\n";

    fn frames(items: Vec<Result<&'static str, Error>>) -> Frames {
        futures_util::stream::iter(items.into_iter().map(|item| item.map(Bytes::from))).boxed()
    }

    async fn relayed_frames(
        items: Vec<Result<&'static str, Error>>,
    ) -> (Vec<Result<Bytes, Error>>, UnobservedTail) {
        let tail = UnobservedTail::default();
        let relayed = relayed(frames(items), tail.clone()).collect().await;
        (relayed, tail)
    }

    #[rstest]
    #[case::message_stop(vec![START, STOP])]
    #[case::provider_error(vec![START, PROVIDER_ERROR])]
    #[case::stop_then_more(vec![START, STOP, START])]
    #[tokio::test]
    async fn a_terminated_stream_is_relayed_as_it_is(#[case] sent: Vec<&'static str>) {
        let (relayed, tail) = relayed_frames(sent.iter().copied().map(Ok).collect()).await;
        let relayed: Vec<Bytes> = relayed.into_iter().map(Result::unwrap).collect();
        assert_eq!(
            relayed,
            sent.into_iter().map(Bytes::from).collect::<Vec<_>>()
        );
        assert!(!tail.begun());
    }

    #[rstest]
    #[case::empty(vec![])]
    #[case::cut_short(vec![START])]
    #[case::error_without_an_error_object(vec![START, UNSHAPED_ERROR])]
    #[tokio::test]
    async fn an_unterminated_stream_ends_with_the_incomplete_stream_error_on_the_tail(
        #[case] sent: Vec<&'static str>,
    ) {
        let (relayed, tail) = relayed_frames(sent.iter().copied().map(Ok).collect()).await;
        let mut relayed: Vec<Bytes> = relayed.into_iter().map(Result::unwrap).collect();
        assert_eq!(
            relayed.pop().unwrap(),
            Bytes::from(format!(
                "event: error\ndata: {{\"type\": \"error\", \"error\": {{\"type\": \"api_error\", \"message\": \"{INCOMPLETE_STREAM_ERROR_MESSAGE}\"}}}}\n\n"
            ))
        );
        assert_eq!(
            relayed,
            sent.into_iter().map(Bytes::from).collect::<Vec<_>>()
        );
        assert!(tail.begun());
    }

    #[tokio::test]
    async fn a_failed_stream_ends_with_its_failure() {
        let failure = Error::from(litellm_http::Error::upstream(500, "cut".to_owned()));
        let (relayed, tail) = relayed_frames(vec![Ok(START), Err(failure)]).await;
        assert_eq!(relayed.len(), 2);
        assert_eq!(relayed[0].as_ref().unwrap(), START.as_bytes());
        assert!(relayed[1].is_err());
        assert!(!tail.begun());
    }

    #[rstest]
    #[case::crlf("event: message_stop\r\ndata: {}\r\n\r\n", true)]
    #[case::cr("event: ping\revent: message_stop\r", true)]
    #[case::unspaced_event("event:message_stop\n\n", false)]
    #[case::error_data_padded("data:   {\"type\": \"error\", \"error\": {}}  \n\n", true)]
    #[case::first_error_payload_decides(
        "data: {\"type\": \"error\", \"error\": 1}\ndata: {\"type\": \"error\", \"error\": {}}\n\n",
        false
    )]
    #[case::undecodable_data("data: {\n\n", false)]
    fn terminal_chunks_are_python_s(#[case] chunk: &str, #[case] terminal: bool) {
        assert_eq!(is_terminal_stream_chunk(chunk.as_bytes()), terminal);
    }
}
