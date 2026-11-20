//! The gateway's estimate of a Messages prompt's tokens. `enable_pre_call_checks` holds it to
//! each deployment's context window, and this module is the only place that decides how a prompt
//! is counted.
//!
//! It is an estimate on purpose. The proxy's own count (`_count_pre_call_check_tokens` over
//! `litellm.token_counter`) is not the provider's either, since it counts a Claude prompt on
//! `cl100k_base`. So this aims at the same size, not the same number, and no request goes to
//! Python over how its prompt is counted. A prompt that sits within the difference of a window
//! can be kept here and dropped by the proxy, or the other way around.
//!
//! Counted as the proxy counts it, through `litellm-token-counter`: the `system` block as a
//! leading message, the text, thinking and tool reference blocks, and the tools.
//!
//! Counted differently, each one a decision made here:
//! - A text over 1024 characters is counted whole. The proxy sums 1024 character chunks and
//!   reads about 0.5% higher
//! - A block the counter does not read (a tool call, a tool result, a search result) is counted
//!   as the text of its JSON. The proxy counts only some of its fields and reads lower
//! - Media carried as bytes or by reference is [`MEDIA_TOKENS`] each, since no tokenizer reads
//!   it. The proxy sizes an image by its dimensions and a PDF by its pages
//! - Tools the counter does not read are counted as the text of their JSON
//! - The encoding is always [`ENCODING`]. The proxy counts a group that serves only OpenAI
//!   models on their own encoding
//!
//! A body with no list of messages, or one the counter cannot read at all, has no estimate. The
//! pre-call checks then hold it to no window, as the proxy does when its own count fails.
//!
//! A route hands its prompt over as a [`Prompt`], and a new format is a new arm of
//! [`Countable::of`] with its own rules here.

use litellm_gateway_inference::routing::Prompt;
use litellm_token_counter::{CountableRequest, TokenCounter};
use serde::Deserialize;
use serde_json::{Map, Value, json};

/// The encoding `litellm.token_counter` counts with for a group that does not serve only OpenAI
/// models.
const ENCODING: &str = "cl100k_base";

/// What one media source counts as: the proxy's `DEFAULT_IMAGE_TOKEN_COUNT`, its own count of an
/// image it cannot size.
const MEDIA_TOKENS: usize = 250;

/// The kinds of `source` that carry media as bytes or by reference. A `text` or `content` source
/// carries text, which is counted as text.
const MEDIA_SOURCES: [&str; 3] = ["base64", "url", "file"];

/// How one content block is counted.
enum Counted<'a> {
    /// The counter reads it as the proxy does.
    Read(&'a Value),
    /// The counter does not read it: the text of its JSON, and how many media sources were taken
    /// out of it.
    Estimated { text: String, media: usize },
}

fn is_read(block: &Value) -> bool {
    let holds_text = |field: &str| block.get(field).is_some_and(Value::is_string);
    match block.get("type").and_then(Value::as_str) {
        Some("text") => holds_text("text"),
        Some("thinking") => holds_text("thinking"),
        Some("tool_reference") => block
            .get("tool_name")
            .is_none_or(|name| name.is_string() || name.is_null()),
        _ => block.is_string(),
    }
}

fn is_media(source: &Value) -> bool {
    source
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| MEDIA_SOURCES.contains(&kind))
}

/// `value` without the media sources inside it, and how many it held.
fn without_media(value: &Value) -> (Value, usize) {
    match value {
        Value::Object(fields) => {
            let (kept, media): (Map<String, Value>, Vec<usize>) = fields
                .iter()
                .map(|(key, value)| {
                    let (value, media) = if key == "source" && is_media(value) {
                        (Value::Null, 1)
                    } else {
                        without_media(value)
                    };
                    ((key.clone(), value), media)
                })
                .unzip();
            (Value::Object(kept), media.iter().sum())
        }
        Value::Array(items) => {
            let (kept, media): (Vec<Value>, Vec<usize>) = items.iter().map(without_media).unzip();
            (Value::Array(kept), media.iter().sum())
        }
        other => (other.clone(), 0),
    }
}

fn counted(block: &Value) -> Counted<'_> {
    if is_read(block) {
        return Counted::Read(block);
    }
    let (kept, media) = without_media(block);
    Counted::Estimated {
        text: kept.to_string(),
        media,
    }
}

/// A message's content as the counter reads it, and how many media sources were taken out of it.
fn content(sent: Option<&Value>) -> (Value, usize) {
    match sent {
        Some(Value::Array(blocks)) => {
            let (read, media): (Vec<Value>, Vec<usize>) = blocks
                .iter()
                .map(|block| match counted(block) {
                    Counted::Read(block) => (block.clone(), 0),
                    Counted::Estimated { text, media } => {
                        (json!({"type": "text", "text": text}), media)
                    }
                })
                .unzip();
            (Value::Array(read), media.iter().sum())
        }
        Some(text @ Value::String(_)) => (text.clone(), 0),
        _ => (Value::Null, 0),
    }
}

/// `anthropic_system_to_openai_message`: a `system` that is text or a list of blocks, and not
/// empty, counts as a leading system message.
fn system_message(body: &Map<String, Value>) -> Option<Value> {
    let system = body.get("system").filter(|system| match system {
        Value::String(text) => !text.is_empty(),
        Value::Array(blocks) => !blocks.is_empty(),
        _ => false,
    })?;
    Some(json!({"role": "system", "content": system}))
}

/// What of a prompt is counted, in the shape the counter reads.
pub(super) struct Countable {
    messages: Vec<Value>,
    tools: Option<Value>,
    media: usize,
}

impl Countable {
    /// `None` for a body with no list of messages.
    pub(super) fn of(prompt: &Prompt<'_>) -> Option<Self> {
        let Prompt::Messages(body) = prompt;
        let sent = body.get("messages")?.as_array()?;
        let (messages, media): (Vec<Value>, Vec<usize>) = system_message(body)
            .iter()
            .chain(sent)
            .map(|message| {
                let (content, media) = content(message.get("content"));
                let role = message.get("role").and_then(Value::as_str);
                (json!({"role": role, "content": content}), media)
            })
            .unzip();
        let tools = body
            .get("tools")
            .filter(|tools| tools.as_array().is_some_and(|tools| !tools.is_empty()));
        Some(Self {
            messages,
            tools: tools.cloned(),
            media: media.iter().sum(),
        })
    }

    /// The estimate. It tokenizes the whole prompt, so it blocks for as long as that takes.
    pub(super) fn tokens(&self) -> Option<i64> {
        let counter = TokenCounter::from_tiktoken(ENCODING).ok()?;
        let read = |tools: Option<&Value>| {
            let fields = json!({"messages": self.messages, "tools": tools});
            let request = CountableRequest::deserialize(fields).ok()?;
            Some(counter.count_request(&request).ok()?.input_tokens)
        };
        let text = match &self.tools {
            None => read(None)?,
            Some(tools) => read(Some(tools))
                .or_else(|| Some(read(None)? + counter.count_text(&tools.to_string()).ok()?))?,
        };
        i64::try_from(text + self.media * MEDIA_TOKENS).ok()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn estimate(body: Value) -> Option<i64> {
        Countable::of(&Prompt::Messages(body.as_object().unwrap()))?.tokens()
    }

    fn asking(content: Value) -> Value {
        json!({"messages": [{"role": "user", "content": content}]})
    }

    fn image(data: &str) -> Value {
        json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
    }

    #[rstest]
    #[case::text(asking(json!("word ".repeat(100))), 108)]
    #[case::text_in_blocks(asking(json!([{"type": "text", "text": "hello there"}])), 9)]
    #[case::thinking(
        json!({"messages": [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hmm let me think", "signature": "abc"},
            {"type": "text", "text": "ok"},
        ]}]}),
        13
    )]
    #[case::a_system_block(
        json!({"system": "Answer briefly.", "messages": [{"role": "user", "content": "word ".repeat(100)}]}),
        115
    )]
    #[case::tools(
        json!({
            "messages": [{"role": "user", "content": "word ".repeat(100)}],
            "tools": [{"name": "noop", "input_schema": {"type": "object"}}],
        }),
        131
    )]
    fn what_the_counter_reads_is_counted_as_the_proxy_counts_it(
        #[case] body: Value,
        #[case] proxy: i64,
    ) {
        assert_eq!(estimate(body), Some(proxy));
    }

    #[rstest]
    fn a_block_the_counter_does_not_read_is_counted_as_its_json() {
        let call = json!({"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}});
        let as_text = asking(json!([{"type": "text", "text": call.to_string()}]));

        assert_eq!(estimate(asking(json!([call]))), estimate(as_text));
    }

    #[rstest]
    #[case::alone(json!([image("AA==")]))]
    #[case::whatever_its_size(json!([image(&"A".repeat(100_000))]))]
    #[case::inside_a_tool_result(json!([{"type": "tool_result", "tool_use_id": "t", "content": [image("AA==")]}]))]
    fn media_counts_flat_and_its_bytes_are_never_tokenized(#[case] content: Value) {
        let (without, _) = without_media(&content);

        assert_eq!(
            estimate(asking(content)),
            estimate(asking(without)).map(|text| text + 250)
        );
    }

    #[rstest]
    fn a_document_that_carries_text_is_counted_as_text() {
        let document = |data: &str| {
            asking(
                json!([{"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": data}}]),
            )
        };

        assert_eq!(
            estimate(document(&"word ".repeat(100)))
                .zip(estimate(document("")))
                .map(|(long, empty)| long - empty),
            Some(101)
        );
    }

    #[rstest]
    fn tools_the_counter_does_not_read_are_counted_as_their_json() {
        let tools = json!([{"name": "pick", "input_schema": {"type": "object", "properties": {
            "on": {"type": "boolean", "enum": [true, false]},
        }}}]);
        let messages = json!([{"role": "user", "content": "hi"}]);
        let counter = TokenCounter::from_tiktoken(ENCODING).unwrap();
        let as_text = i64::try_from(counter.count_text(&tools.to_string()).unwrap()).unwrap();

        assert_eq!(
            estimate(json!({"messages": messages, "tools": tools})),
            estimate(json!({"messages": messages})).map(|alone| alone + as_text)
        );
    }

    #[rstest]
    #[case::no_messages(json!({"model": "claude"}))]
    #[case::messages_that_are_no_list(json!({"messages": "hi"}))]
    fn a_body_with_no_list_of_messages_has_no_estimate(#[case] body: Value) {
        assert_eq!(estimate(body), None);
    }
}
