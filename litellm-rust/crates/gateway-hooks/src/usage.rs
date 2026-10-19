//! Token usage, read from a response without holding it: a complete body is read once, and a
//! stream is scanned line by line, keeping only the few lines that carry usage.

use serde::Deserialize;
use serde_json::Value;

/// A provider's service tier, which can change the price.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ServiceTier {
    #[default]
    Standard,
    Priority,
    Flex,
    Batch,
}

impl ServiceTier {
    fn parse(name: &str) -> Self {
        match name {
            "priority" => Self::Priority,
            "flex" => Self::Flex,
            "batch" => Self::Batch,
            _ => Self::Standard,
        }
    }
}

/// Token counts of one response. Input is split the way prices are: tokens read from the
/// prompt cache, tokens written to it, and the rest.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Usage {
    /// Input tokens neither read from nor written to the prompt cache.
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    /// Every token written to the prompt cache, at either lifetime.
    pub cache_write_tokens: u64,
    /// The part of `cache_write_tokens` written for an hour, billed at its own rate.
    pub cache_write_1h_tokens: u64,
    pub web_search_requests: u64,
    pub service_tier: ServiceTier,
}

impl Usage {
    /// Prompt tokens as the proxy counts them: cache reads and writes included.
    pub fn prompt_tokens(&self) -> u64 {
        self.input_tokens
            .saturating_add(self.cache_read_tokens)
            .saturating_add(self.cache_write_tokens)
    }

    pub fn total_tokens(&self) -> u64 {
        self.prompt_tokens().saturating_add(self.output_tokens)
    }

    /// Usage from an Anthropic Messages `usage` object.
    pub fn from_anthropic(value: &Value) -> Option<Self> {
        let usage = AnthropicUsage::deserialize(value).ok()?;
        let mut merged = Self::default();
        usage.merge_into(&mut merged);
        Some(merged)
    }

    /// Usage from an OpenAI Chat Completions `usage` object, whose prompt count includes cache
    /// reads.
    pub fn from_openai(value: &Value) -> Option<Self> {
        OpenAiUsage::deserialize(value).ok().map(OpenAiUsage::usage)
    }

    /// Usage from counts in the proxy's convention: `prompt_tokens` includes `cached_tokens`
    /// and `cache_creation_tokens`.
    pub fn from_prompt_counts(
        prompt_tokens: u64,
        completion_tokens: u64,
        cached_tokens: u64,
        cache_creation_tokens: u64,
    ) -> Self {
        Self {
            input_tokens: prompt_tokens
                .saturating_sub(cached_tokens)
                .saturating_sub(cache_creation_tokens),
            output_tokens: completion_tokens,
            cache_read_tokens: cached_tokens,
            cache_write_tokens: cache_creation_tokens,
            ..Self::default()
        }
    }
}

#[derive(Deserialize, Default)]
struct AnthropicUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    cache_creation: Option<CacheCreation>,
    server_tool_use: Option<ServerToolUse>,
    service_tier: Option<String>,
}

#[derive(Deserialize, Default)]
struct CacheCreation {
    ephemeral_1h_input_tokens: Option<u64>,
}

#[derive(Deserialize, Default)]
struct ServerToolUse {
    web_search_requests: Option<u64>,
}

impl AnthropicUsage {
    /// Applies the counts this object carries over `usage`; a stream's later events repeat or
    /// raise the counts of earlier ones, so a present count replaces the earlier one.
    fn merge_into(self, usage: &mut Usage) {
        if let Some(tokens) = self.input_tokens {
            usage.input_tokens = tokens;
        }
        if let Some(tokens) = self.output_tokens {
            usage.output_tokens = tokens;
        }
        if let Some(tokens) = self.cache_read_input_tokens {
            usage.cache_read_tokens = tokens;
        }
        if let Some(tokens) = self.cache_creation_input_tokens {
            usage.cache_write_tokens = tokens;
        }
        if let Some(tokens) = self
            .cache_creation
            .and_then(|cache| cache.ephemeral_1h_input_tokens)
        {
            usage.cache_write_1h_tokens = tokens.min(usage.cache_write_tokens);
        }
        if let Some(requests) = self
            .server_tool_use
            .and_then(|tools| tools.web_search_requests)
        {
            usage.web_search_requests = requests;
        }
        if let Some(tier) = self.service_tier {
            usage.service_tier = ServiceTier::parse(&tier);
        }
    }
}

/// OpenAI's usage, with the fields LiteLLM adds when it translates another provider's:
/// cache writes by lifetime and server tool counts.
#[derive(Deserialize)]
struct OpenAiUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    prompt_tokens_details: Option<OpenAiPromptDetails>,
    server_tool_use: Option<ServerToolUse>,
}

#[derive(Deserialize, Default)]
struct OpenAiPromptDetails {
    cached_tokens: Option<u64>,
    cache_creation_tokens: Option<u64>,
    cache_creation_token_details: Option<CacheCreation>,
}

impl OpenAiUsage {
    fn usage(self) -> Usage {
        let details = self.prompt_tokens_details.unwrap_or_default();
        let mut usage = Usage::from_prompt_counts(
            self.prompt_tokens,
            self.completion_tokens,
            details.cached_tokens.unwrap_or(0),
            details.cache_creation_tokens.unwrap_or(0),
        );
        if let Some(tokens) = details
            .cache_creation_token_details
            .and_then(|cache| cache.ephemeral_1h_input_tokens)
        {
            usage.cache_write_1h_tokens = tokens.min(usage.cache_write_tokens);
        }
        if let Some(requests) = self
            .server_tool_use
            .and_then(|tools| tools.web_search_requests)
        {
            usage.web_search_requests = requests;
        }
        usage
    }
}

/// The wire format of a stream being scanned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamFormat {
    /// Anthropic Messages events: usage in `message_start` and `message_delta`.
    Anthropic,
    /// OpenAI Chat Completions chunks: usage in the last chunk when the caller asked for it.
    OpenAi,
}

/// The first bytes of a line, enough to tell an Anthropic usage event from content.
const DECIDE_BYTES: usize = 96;
/// The longest line kept. Usage lines are a few hundred bytes; anything longer is content.
const MAX_LINE_BYTES: usize = 64 * 1024;

#[derive(Debug)]
enum Line {
    Reading,
    Skipping,
}

/// What the stream's `event:` line named for the event in progress. Anthropic streams name every
/// event, which decides a data line whatever order its JSON keys come in.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Event {
    Unnamed,
    Usage,
    Other,
}

/// Reads usage out of a server-sent-event stream as its chunks pass through. Only the line in
/// progress is buffered, and only while it may carry usage.
#[derive(Debug)]
pub struct StreamUsage {
    format: StreamFormat,
    line: Vec<u8>,
    state: Line,
    event: Event,
    usage: Option<Usage>,
    model: Option<String>,
}

impl StreamUsage {
    pub fn new(format: StreamFormat) -> Self {
        Self {
            format,
            line: Vec::new(),
            state: Line::Reading,
            event: Event::Unnamed,
            usage: None,
            model: None,
        }
    }

    pub fn feed(&mut self, mut chunk: &[u8]) {
        while !chunk.is_empty() {
            let end = memchr::memchr(b'\n', chunk);
            let (part, rest) = match end {
                Some(index) => (&chunk[..index], &chunk[index + 1..]),
                None => (chunk, &[][..]),
            };
            if let Line::Reading = self.state {
                self.take(part);
            }
            if end.is_some() {
                match self.state {
                    Line::Reading => self.read_line(),
                    // A skipped line is content, never the blank line that ends an event.
                    Line::Skipping => {}
                }
                self.line.clear();
                self.state = Line::Reading;
            }
            chunk = rest;
        }
    }

    /// Appends part of the line in progress, deciding as early as possible whether to keep it,
    /// so a long content line is never copied.
    fn take(&mut self, part: &[u8]) {
        let undecided = match self.format {
            StreamFormat::Anthropic => DECIDE_BYTES.saturating_sub(self.line.len()),
            StreamFormat::OpenAi => 0,
        };
        let (head, tail) = part.split_at(undecided.min(part.len()));
        for piece in [head, tail] {
            if piece.is_empty() {
                continue;
            }
            if self.line.len() + piece.len() > MAX_LINE_BYTES {
                self.skip();
                return;
            }
            self.line.extend_from_slice(piece);
            if !self.may_carry_usage() {
                self.skip();
                return;
            }
        }
    }

    fn skip(&mut self) {
        self.line.clear();
        self.state = Line::Skipping;
    }

    /// The usage seen so far; `None` when no event carried any.
    pub fn usage(&self) -> Option<Usage> {
        self.usage
    }

    /// The model named by the chunk that carried an OpenAI stream's usage.
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    fn may_carry_usage(&self) -> bool {
        let line = &self.line;
        if line.len() > MAX_LINE_BYTES {
            return false;
        }
        match self.format {
            StreamFormat::Anthropic if line.len() >= DECIDE_BYTES => match self.event {
                Event::Usage => true,
                Event::Other => false,
                Event::Unnamed => {
                    let head = &line[..DECIDE_BYTES];
                    contains(head, b"\"message_start\"") || contains(head, b"\"message_delta\"")
                }
            },
            StreamFormat::Anthropic | StreamFormat::OpenAi => true,
        }
    }

    fn read_line(&mut self) {
        let line = self.line.strip_suffix(b"\r").unwrap_or(&self.line);
        if line.is_empty() {
            self.event = Event::Unnamed;
            return;
        }
        if let Some(name) = line.strip_prefix(b"event:") {
            let name = name.strip_prefix(b" ").unwrap_or(name);
            self.event = match name {
                b"message_start" | b"message_delta" => Event::Usage,
                _ => Event::Other,
            };
            return;
        }
        let Some(data) = line.strip_prefix(b"data:") else {
            return;
        };
        let data = data.strip_prefix(b" ").unwrap_or(data);
        match self.format {
            StreamFormat::Anthropic => {
                let named = self.event == Event::Usage;
                if !named
                    && !(contains(data, b"\"message_start\"")
                        || contains(data, b"\"message_delta\""))
                {
                    return;
                }
                let Ok(event) = serde_json::from_slice::<AnthropicEvent>(data) else {
                    return;
                };
                let usage = self.usage.get_or_insert_with(Usage::default);
                if let Some(start) = event.message.and_then(|message| message.usage) {
                    start.merge_into(usage);
                }
                if let Some(delta) = event.usage {
                    delta.merge_into(usage);
                }
            }
            StreamFormat::OpenAi => {
                if !contains(data, b"\"usage\":{") && !contains(data, b"\"usage\": {") {
                    return;
                }
                if let Ok(OpenAiChunk {
                    usage: Some(usage),
                    model,
                }) = serde_json::from_slice(data)
                {
                    self.usage = Some(usage.usage());
                    self.model = model.filter(|model| !model.is_empty());
                }
            }
        }
    }
}

#[derive(Deserialize)]
struct AnthropicEvent {
    message: Option<AnthropicMessage>,
    usage: Option<AnthropicUsage>,
}

#[derive(Deserialize)]
struct AnthropicMessage {
    usage: Option<AnthropicUsage>,
}

#[derive(Deserialize)]
struct OpenAiChunk {
    usage: Option<OpenAiUsage>,
    model: Option<String>,
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    memchr::memmem::find(haystack, needle).is_some()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    const ANTHROPIC_STREAM: &str = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-x\",\"usage\":{\"input_tokens\":10,\"cache_creation_input_tokens\":6,\"cache_read_input_tokens\":20,\"cache_creation\":{\"ephemeral_5m_input_tokens\":2,\"ephemeral_1h_input_tokens\":4},\"output_tokens\":1,\"service_tier\":\"standard\"}}}\n",
        "\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"a long piece of text that is longer than the decision window, and then some more\"}}\n",
        "\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":42,\"server_tool_use\":{\"web_search_requests\":2}}}\n",
        "\n",
    );

    fn anthropic_usage() -> Usage {
        Usage {
            input_tokens: 10,
            output_tokens: 42,
            cache_read_tokens: 20,
            cache_write_tokens: 6,
            cache_write_1h_tokens: 4,
            web_search_requests: 2,
            service_tier: ServiceTier::Standard,
        }
    }

    #[rstest]
    #[case::whole(ANTHROPIC_STREAM.len())]
    #[case::bytes(1)]
    #[case::small(7)]
    #[case::mid(50)]
    fn anthropic_stream_usage_survives_any_chunking(#[case] size: usize) {
        let mut meter = StreamUsage::new(StreamFormat::Anthropic);
        for chunk in ANTHROPIC_STREAM.as_bytes().chunks(size) {
            meter.feed(chunk);
        }
        assert_eq!(meter.usage(), Some(anthropic_usage()));
    }

    #[rstest]
    fn content_lines_are_not_buffered() {
        let mut meter = StreamUsage::new(StreamFormat::Anthropic);
        let text = "x".repeat(MAX_LINE_BYTES * 2);
        meter.feed(b"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"");
        meter.feed(text.as_bytes());
        assert!(meter.line.capacity() < 4 * DECIDE_BYTES);
        meter.feed(b"\"}}\n\n");
        assert_eq!(meter.usage(), None);
    }

    /// Key order is not part of JSON: an upstream may put `type` after the message object, past
    /// the decision window, and the event line still marks the usage events.
    #[rstest]
    #[case::whole(usize::MAX)]
    #[case::bytes(1)]
    #[case::small(7)]
    fn anthropic_usage_events_are_found_by_event_name(#[case] size: usize) {
        let stream = concat!(
            "event: message_start\n",
            "data: {\"message\":{\"content\":[],\"id\":\"msg_0123456789abcdef0123456789abcdef\",\"model\":\"claude-x\",\"role\":\"assistant\",\"stop_reason\":null,\"type\":\"message\",\"usage\":{\"input_tokens\":1234,\"output_tokens\":1}},\"type\":\"message_start\"}\n",
            "\n",
            "event: content_block_delta\n",
            "data: {\"delta\":{\"text\":\"text that mentions \\\"message_delta\\\" and is longer than the decision window\",\"type\":\"text_delta\"},\"index\":0,\"type\":\"content_block_delta\",\"usage\":{\"output_tokens\":999}}\n",
            "\n",
            "event: message_delta\n",
            "data: {\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"type\":\"message_delta\",\"usage\":{\"output_tokens\":42}}\n",
            "\n",
        );
        let mut meter = StreamUsage::new(StreamFormat::Anthropic);
        for chunk in stream.as_bytes().chunks(size.min(stream.len())) {
            meter.feed(chunk);
        }
        let usage = meter.usage().expect("usage");
        assert_eq!((usage.input_tokens, usage.output_tokens), (1234, 42));
    }

    #[rstest]
    fn crlf_lines_are_read() {
        let mut meter = StreamUsage::new(StreamFormat::Anthropic);
        meter.feed(b"data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":3}}\r\n\r\n");
        assert_eq!(meter.usage().map(|usage| usage.output_tokens), Some(3));
    }

    #[rstest]
    fn openai_usage_comes_from_the_last_chunk() {
        let stream = concat!(
            "data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}],\"usage\":null}\n\n",
            "data: {\"id\":\"c\",\"model\":\"gpt-4o-2024-11-20\",\"choices\":[],\"usage\":{\"prompt_tokens\":30,\"completion_tokens\":5,\"total_tokens\":35,\"prompt_tokens_details\":{\"cached_tokens\":10}}}\n\n",
            "data: [DONE]\n\n",
        );
        let mut meter = StreamUsage::new(StreamFormat::OpenAi);
        for chunk in stream.as_bytes().chunks(9) {
            meter.feed(chunk);
        }
        assert_eq!(
            meter.usage(),
            Some(Usage {
                input_tokens: 20,
                output_tokens: 5,
                cache_read_tokens: 10,
                ..Usage::default()
            })
        );
        assert_eq!(meter.model(), Some("gpt-4o-2024-11-20"));
    }

    #[rstest]
    fn anthropic_body_usage() {
        let usage = Usage::from_anthropic(&json!({
            "input_tokens": 10,
            "output_tokens": 42,
            "cache_creation_input_tokens": 6,
            "cache_read_input_tokens": 20,
            "cache_creation": {"ephemeral_5m_input_tokens": 2, "ephemeral_1h_input_tokens": 4},
            "server_tool_use": {"web_search_requests": 2},
        }));
        assert_eq!(usage, Some(anthropic_usage()));
        assert_eq!(anthropic_usage().prompt_tokens(), 36);
        assert_eq!(anthropic_usage().total_tokens(), 78);
    }

    #[rstest]
    fn translated_openai_usage_keeps_cache_lifetimes_and_server_tools() {
        let usage = Usage::from_openai(&json!({
            "prompt_tokens": 36,
            "completion_tokens": 42,
            "prompt_tokens_details": {"cached_tokens": 20, "cache_creation_tokens": 6,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 2, "ephemeral_1h_input_tokens": 4}},
            "server_tool_use": {"web_search_requests": 2},
        }));
        assert_eq!(usage, Some(anthropic_usage()));
    }

    #[rstest]
    fn openai_body_usage() {
        let usage = Usage::from_openai(&json!({
            "prompt_tokens": 100,
            "completion_tokens": 7,
            "total_tokens": 107,
            "prompt_tokens_details": {"cached_tokens": 60},
        }));
        assert_eq!(
            usage,
            Some(Usage {
                input_tokens: 40,
                output_tokens: 7,
                cache_read_tokens: 60,
                ..Usage::default()
            })
        );
    }
}
