use std::convert::Infallible;

use bytes::Bytes;
use litellm_host::{
    call::{HostedCompletion, HostedMachine, hosted_call},
    protocol::Protocol,
};

use super::{Error, MessagesBody, MessagesCall};

pub type MessagesOutput = HostedCompletion<MessagesBody>;

/// The upstream response as the caller sees it at stream hand-off, before any chunk.
pub struct MessagesStreamHead {
    pub headers: Vec<(String, String)>,
}

pub struct Messages;

impl Protocol for Messages {
    type Response = MessagesBody;
    type Error = Error;
    type Request = MessagesCall;
    type HostCall = Infallible;
    type Chunk = Bytes;
    type StreamHead = MessagesStreamHead;
}

pub type MessagesMachine = HostedMachine<Messages>;

impl super::MessagesRoute {
    pub fn machine(
        self,
        request: super::MessagesCall,
        options: impl Into<litellm_inference::CallOptions>,
    ) -> MessagesMachine {
        let litellm_inference::CallOptions {
            cache: cache_options,
            observers,
        } = options.into();
        hosted_call(
            request,
            observers,
            move |call, _, interceptors, observers| async move {
                let context = litellm_inference::context::CallContext::new(
                    &interceptors,
                    litellm_inference::CallOptions {
                        cache: cache_options,
                        observers,
                    },
                );
                self.run(call, context).await
            },
        )
    }
}

impl litellm_inference::caching::Cachable for Messages {
    const SURFACE: &'static str = "messages";
}

impl litellm_inference::caching::StreamCachable for Messages {
    const TERMINAL_EVENT: &'static str = "message_stop";

    fn replay(data: bytes::Bytes) -> Option<litellm_host::call::OutputOf<Self>> {
        Some(litellm_host::call::CallOutput::Stream {
            head: MessagesStreamHead {
                headers: Vec::new(),
            },
            chunks: Box::pin(futures_util::stream::iter([Ok(data)])),
        })
    }

    fn bytes(chunk: &Self::Chunk) -> &[u8] {
        chunk.as_ref()
    }

    fn storable_stream(text: &str) -> bool {
        let lines: Vec<&str> = sse_lines(text).collect();
        lines.contains(&"event: message_stop")
            && !carries_provider_error(&lines)
            && lines.iter().any(|line| line_carries_output(line))
    }
}

fn sse_lines(text: &str) -> impl Iterator<Item = &str> {
    let mut rest = Some(text);
    std::iter::from_fn(move || {
        let current = rest?;
        match current.find(['\r', '\n']) {
            Some(at) => {
                let skip = if current[at..].starts_with("\r\n") {
                    2
                } else {
                    1
                };
                rest = Some(&current[at + skip..]);
                Some(&current[..at])
            }
            None => {
                rest = None;
                (!current.is_empty()).then_some(current)
            }
        }
    })
}

fn sse_data(line: &str, stripped: bool) -> Option<serde_json::Map<String, serde_json::Value>> {
    let data = line.strip_prefix("data:")?;
    let data = if stripped {
        data.trim_matches(|c: char| matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c'))
    } else {
        data
    };
    match serde_json::from_str(data) {
        Ok(serde_json::Value::Object(map)) => Some(map),
        _ => None,
    }
}

fn carries_provider_error(lines: &[&str]) -> bool {
    lines
        .iter()
        .filter_map(|line| sse_data(line, true))
        .find(|event| event.get("type").and_then(serde_json::Value::as_str) == Some("error"))
        .is_some_and(|event| event.get("error").is_some_and(serde_json::Value::is_object))
}

fn line_carries_output(line: &str) -> bool {
    let Some(event) = sse_data(line, false) else {
        return false;
    };
    let block = match event.get("type").and_then(serde_json::Value::as_str) {
        Some("content_block_start") => event.get("content_block"),
        Some("content_block_delta") => event.get("delta"),
        _ => None,
    };
    block
        .and_then(serde_json::Value::as_object)
        .is_some_and(|fields| {
            fields.iter().any(|(name, value)| {
                name != "type"
                    && !matches!(value, serde_json::Value::Null)
                    && value.as_str() != Some("")
                    && value.as_array().is_none_or(|items| !items.is_empty())
            })
        })
}

#[cfg(test)]
mod tests {
    use litellm_inference::caching::StreamCachable;
    use rstest::rstest;

    use super::Messages;

    const TEXT: &str = "event: message_start\ndata: {\"type\":\"message_start\"}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Paris\"}}\n\n";
    const EMPTY: &str = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"\"}}\n\n";
    const STOP: &str = "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";

    #[rstest]
    #[case::complete(format!("{TEXT}{STOP}"), true)]
    #[case::crlf_lines(format!("{TEXT}{STOP}").replace('\n', "\r\n"), true)]
    #[case::done_sentinel(format!("{TEXT}{STOP}data: [DONE]\n\n"), true)]
    #[case::data_only_frames(format!("{TEXT}{STOP}").replace("event: message_stop\n", ""), false)]
    #[case::event_without_space(format!("{TEXT}{STOP}").replace("event: message_stop", "event:message_stop"), false)]
    #[case::provider_error(format!("{TEXT}{STOP}event: error\ndata: {{\"type\":\"error\",\"error\":{{\"type\":\"overloaded_error\"}}}}\n\n"), false)]
    #[case::error_without_body(format!("{TEXT}{STOP}data: {{\"type\":\"error\",\"error\":\"overloaded\"}}\n\n"), true)]
    #[case::no_output(format!("{EMPTY}{STOP}"), false)]
    fn a_stream_is_stored_by_python_s_message_stop_error_and_output_tests(
        #[case] text: String,
        #[case] stored: bool,
    ) {
        assert_eq!(Messages::storable_stream(&text), stored);
    }
}
