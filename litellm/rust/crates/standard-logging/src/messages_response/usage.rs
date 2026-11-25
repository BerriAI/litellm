//! An Anthropic answer's `usage` as Python's litellm `Usage` counts it
//! (`AnthropicConfig.calculate_usage`), and the two shapes its `model_dump()` takes: the `Usage` a
//! complete answer's transform builds, and the one `stream_chunk_builder` rebuilds from a
//! stream's chunks (`ChunkProcessor.calculate_usage`).

use serde::Serialize;
use serde_json::{Map, Value};

use super::{Error, Null};

/// A `usage` block read as `calculate_usage` reads it: a count it cannot read is `0`, and a
/// field of another type is absent.
pub(super) struct ReportedUsage {
    input_tokens: u64,
    output_tokens: u64,
    cache_creation_input_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    cache_creation: Option<CacheCreationSplit>,
    server_tool_use: Option<ServerToolRequests>,
    thinking_tokens: Option<u64>,
    inference_geo: Option<String>,
    speed: Option<String>,
    service_tier: Option<String>,
    /// An empty `iterations` list, which Python carries onto its `Usage` as it is.
    iterations: Option<Vec<Value>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CacheCreationSplit {
    ephemeral_5m_input_tokens: Option<u64>,
    ephemeral_1h_input_tokens: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ServerToolRequests {
    web_search_requests: Option<u64>,
    tool_search_requests: Option<u64>,
}

/// `int(value) if isinstance(value, (int, float)) else 0` over `usage.get(key, 0) or 0`.
fn count(usage: &Map<String, Value>, key: &str) -> u64 {
    match usage.get(key) {
        Some(Value::Number(number)) => number
            .as_u64()
            .or_else(|| number.as_f64().map(|float| float as u64))
            .unwrap_or(0),
        _ => 0,
    }
}

fn whole(value: Option<&Value>) -> Option<u64> {
    value.and_then(Value::as_u64)
}

fn text(value: Option<&Value>) -> Option<String> {
    value.and_then(Value::as_str).map(str::to_owned)
}

impl ReportedUsage {
    /// Reads `usage`, refusing per-iteration usage, which the Rust builders do not port.
    pub(super) fn read(usage: &Map<String, Value>) -> Result<Self, Error> {
        let iterations = match usage.get("iterations") {
            Some(Value::Array(iterations)) if !iterations.is_empty() => {
                return Err(Error::UnportedIterations);
            }
            Some(Value::Array(_)) => Some(Vec::new()),
            _ => None,
        };
        let split = |details: &Value| CacheCreationSplit {
            ephemeral_5m_input_tokens: whole(details.get("ephemeral_5m_input_tokens")),
            ephemeral_1h_input_tokens: whole(details.get("ephemeral_1h_input_tokens")),
        };
        let requests = |tools: &Value| ServerToolRequests {
            web_search_requests: whole(tools.get("web_search_requests")),
            tool_search_requests: whole(tools.get("tool_search_requests")),
        };
        Ok(Self {
            input_tokens: count(usage, "input_tokens"),
            output_tokens: count(usage, "output_tokens"),
            cache_creation_input_tokens: whole(usage.get("cache_creation_input_tokens")),
            cache_read_input_tokens: whole(usage.get("cache_read_input_tokens")),
            cache_creation: usage
                .get("cache_creation")
                .filter(|details| details.is_object())
                .map(split),
            server_tool_use: usage
                .get("server_tool_use")
                .filter(|tools| tools.is_object())
                .map(requests),
            thinking_tokens: usage
                .get("output_tokens_details")
                .and_then(|details| whole(details.get("thinking_tokens"))),
            inference_geo: text(usage.get("inference_geo")),
            speed: text(usage.get("speed")),
            service_tier: text(usage.get("service_tier")),
            iterations,
        })
    }
}

/// What the answer's tokens tell about its reasoning.
#[derive(Clone, Copy)]
pub(super) struct Reasoning<'a> {
    /// The answer's thinking text joined, `None` when it has none to join.
    pub(super) content: Option<&'a str>,
    /// Whether the answer the usage belongs to carries a thinking block. A stream's chunks never
    /// say so, since its usage is counted with no answer in hand.
    pub(super) has_thinking_block: bool,
}

/// The `Usage` `calculate_usage` returns, before either shape writes it.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Counted {
    prompt_tokens: u64,
    completion_tokens: u64,
    cache_creation_input_tokens: u64,
    cache_read_input_tokens: u64,
    cache_creation: Option<CacheCreationSplit>,
    reasoning_tokens: Option<u64>,
    text_tokens: Option<u64>,
    server_tool_use: Option<ServerToolRequests>,
    inference_geo: Option<String>,
    speed: Option<String>,
    service_tier: Option<String>,
    iterations: Option<Vec<Value>>,
}

/// `AnthropicConfig.calculate_usage` for usage without iterations.
pub(super) fn calculate(
    reported: ReportedUsage,
    reasoning: Reasoning<'_>,
    requested_speed: Option<&str>,
    count_tokens: &dyn Fn(&str) -> u64,
) -> Counted {
    let cache_creation_input_tokens = reported.cache_creation_input_tokens.unwrap_or(0);
    let cache_read_input_tokens = reported.cache_read_input_tokens.unwrap_or(0);
    let completion_tokens = reported.output_tokens;
    let (reasoning_tokens, text_tokens) = match (reported.thinking_tokens, reasoning) {
        (Some(thinking_tokens), _) => {
            let reasoning_tokens = thinking_tokens.min(completion_tokens);
            (
                Some(reasoning_tokens),
                Some(completion_tokens - reasoning_tokens),
            )
        }
        (
            None,
            Reasoning {
                content: Some(content),
                ..
            },
        ) if !content.is_empty() => {
            let estimated = count_tokens(content).min(completion_tokens);
            (Some(estimated), Some(completion_tokens - estimated))
        }
        (
            None,
            Reasoning {
                has_thinking_block: true,
                ..
            },
        ) => (None, None),
        (None, _) => (Some(0), Some(completion_tokens)),
    };
    Counted {
        prompt_tokens: reported.input_tokens
            + cache_creation_input_tokens
            + cache_read_input_tokens,
        completion_tokens,
        cache_creation_input_tokens,
        cache_read_input_tokens,
        cache_creation: reported.cache_creation,
        reasoning_tokens,
        text_tokens,
        server_tool_use: reported.server_tool_use.filter(|requests| {
            requests.web_search_requests.is_some() || requests.tool_search_requests.is_some()
        }),
        inference_geo: reported.inference_geo,
        speed: reported
            .speed
            .or_else(|| requested_speed.map(str::to_owned)),
        service_tier: reported.service_tier,
        iterations: reported.iterations,
    }
}

impl Counted {
    fn prompt_tokens_details(&self) -> PromptTokensDetails {
        PromptTokensDetails {
            audio_tokens: Null,
            cached_tokens: self.cache_read_input_tokens,
            text_tokens: self.prompt_tokens
                - self.cache_read_input_tokens
                - self.cache_creation_input_tokens,
            image_tokens: Null,
            video_tokens: Null,
            initial_prompt_tokens: Null,
            cache_write_tokens: self.cache_creation_input_tokens,
            cache_creation_tokens: self.cache_creation_input_tokens,
            cache_creation_token_details: self.cache_creation.map(CacheCreationTokenDetails::from),
        }
    }

    /// Whether the counts report anything on the prompt side, so a later chunk's cache counts
    /// replace an earlier one's (`_reports_prompt_side_usage`).
    fn reports_prompt_side(&self) -> bool {
        self.prompt_tokens > 0
            || self.cache_creation_input_tokens > 0
            || self.cache_read_input_tokens > 0
    }
}

/// `PromptTokensDetailsWrapper.model_dump()` as the Anthropic usage fills it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PromptTokensDetails {
    audio_tokens: Null,
    cached_tokens: u64,
    text_tokens: u64,
    image_tokens: Null,
    video_tokens: Null,
    initial_prompt_tokens: Null,
    cache_write_tokens: u64,
    cache_creation_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_creation_token_details: Option<CacheCreationTokenDetails>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CacheCreationTokenDetails {
    ephemeral_5m_input_tokens: Option<u64>,
    ephemeral_30m_input_tokens: Null,
    ephemeral_1h_input_tokens: Option<u64>,
    ephemeral_12h_input_tokens: Null,
}

impl From<CacheCreationSplit> for CacheCreationTokenDetails {
    fn from(split: CacheCreationSplit) -> Self {
        Self {
            ephemeral_5m_input_tokens: split.ephemeral_5m_input_tokens,
            ephemeral_30m_input_tokens: Null,
            ephemeral_1h_input_tokens: split.ephemeral_1h_input_tokens,
            ephemeral_12h_input_tokens: Null,
        }
    }
}

/// The `Usage` a complete answer's transform builds, as its `model_dump()` writes it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CompleteUsage {
    completion_tokens: u64,
    prompt_tokens: u64,
    total_tokens: u64,
    completion_tokens_details: CompleteCompletionTokensDetails,
    prompt_tokens_details: PromptTokensDetails,
    #[serde(skip_serializing_if = "Option::is_none")]
    server_tool_use: Option<CompleteServerToolUse>,
    cache_creation_input_tokens: u64,
    cache_read_input_tokens: u64,
    iterations: Option<Vec<Value>>,
    inference_geo: Option<String>,
    speed: Option<String>,
    service_tier: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
struct CompleteCompletionTokensDetails {
    reasoning_tokens: Option<u64>,
    text_tokens: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
struct CompleteServerToolUse {
    web_search_requests: Option<u64>,
    tool_search_requests: Option<u64>,
}

impl From<Counted> for CompleteUsage {
    fn from(counted: Counted) -> Self {
        Self {
            completion_tokens: counted.completion_tokens,
            prompt_tokens: counted.prompt_tokens,
            total_tokens: counted.prompt_tokens + counted.completion_tokens,
            completion_tokens_details: CompleteCompletionTokensDetails {
                reasoning_tokens: counted.reasoning_tokens,
                text_tokens: counted.text_tokens,
            },
            prompt_tokens_details: counted.prompt_tokens_details(),
            server_tool_use: counted
                .server_tool_use
                .map(|requests| CompleteServerToolUse {
                    web_search_requests: requests.web_search_requests,
                    tool_search_requests: requests.tool_search_requests,
                }),
            cache_creation_input_tokens: counted.cache_creation_input_tokens,
            cache_read_input_tokens: counted.cache_read_input_tokens,
            iterations: counted.iterations,
            inference_geo: counted.inference_geo,
            speed: counted.speed,
            service_tier: counted.service_tier,
        }
    }
}

/// The `Usage` `stream_chunk_builder` rebuilds from a stream's usage chunks, as its
/// `model_dump()` writes it. It keeps no `service_tier` or `iterations`, and names
/// `inference_geo` and `speed` only when a chunk did.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StreamedUsage {
    completion_tokens: u64,
    prompt_tokens: u64,
    total_tokens: u64,
    completion_tokens_details: StreamedCompletionTokensDetails,
    prompt_tokens_details: PromptTokensDetails,
    #[serde(skip_serializing_if = "Option::is_none")]
    server_tool_use: Option<StreamedServerToolUse>,
    cache_creation_input_tokens: u64,
    cache_read_input_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    inference_geo: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
struct StreamedCompletionTokensDetails {
    accepted_prediction_tokens: Null,
    audio_tokens: Null,
    reasoning_tokens: Option<u64>,
    rejected_prediction_tokens: Null,
    text_tokens: Option<u64>,
    image_tokens: Null,
    video_tokens: Null,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
struct StreamedServerToolUse {
    web_search_requests: Option<u64>,
    tool_search_requests: Option<u64>,
    browser_open_requests: Null,
}

impl StreamedUsage {
    /// `ChunkProcessor._calculate_usage_per_chunk` and `calculate_usage` over the `Usage` each
    /// usage-bearing chunk counted, in stream order: a token count is the last one above zero,
    /// a cache count the last one from a chunk that reports the prompt side, and the details the
    /// last chunk's, keeping a cache split only an earlier chunk named. `None` for a stream that
    /// counted nothing.
    pub(super) fn of(chunks: Vec<Counted>) -> Option<Self> {
        let last_positive = |tokens: fn(&Counted) -> u64| {
            chunks
                .iter()
                .map(tokens)
                .rfind(|&count| count > 0)
                .or_else(|| chunks.first().map(tokens))
        };
        let prompt_tokens = last_positive(|chunk| chunk.prompt_tokens)?;
        let completion_tokens = last_positive(|chunk| chunk.completion_tokens)?;
        let prompt_side = |tokens: fn(&Counted) -> u64| {
            chunks
                .iter()
                .rfind(|chunk| chunk.reports_prompt_side())
                .or(chunks.first())
                .map_or(0, tokens)
        };
        let cache_creation_input_tokens = prompt_side(|chunk| chunk.cache_creation_input_tokens);
        let cache_read_input_tokens = prompt_side(|chunk| chunk.cache_read_input_tokens);
        let cache_creation = chunks.iter().rev().find_map(|chunk| chunk.cache_creation);
        let server_tool_use = chunks.iter().rev().find_map(|chunk| chunk.server_tool_use);
        let inference_geo = chunks
            .iter()
            .rev()
            .find_map(|chunk| chunk.inference_geo.clone());
        let speed = chunks.iter().rev().find_map(|chunk| chunk.speed.clone());
        let last = chunks.last()?;
        let mut prompt_tokens_details = last.prompt_tokens_details();
        if prompt_tokens_details.cache_creation_token_details.is_none() {
            prompt_tokens_details.cache_creation_token_details =
                cache_creation.map(CacheCreationTokenDetails::from);
        }
        Some(Self {
            completion_tokens,
            prompt_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            completion_tokens_details: StreamedCompletionTokensDetails {
                accepted_prediction_tokens: Null,
                audio_tokens: Null,
                reasoning_tokens: last.reasoning_tokens,
                rejected_prediction_tokens: Null,
                text_tokens: last.text_tokens,
                image_tokens: Null,
                video_tokens: Null,
            },
            prompt_tokens_details,
            server_tool_use: server_tool_use.map(|requests| StreamedServerToolUse {
                web_search_requests: requests.web_search_requests,
                tool_search_requests: requests.tool_search_requests,
                browser_open_requests: Null,
            }),
            cache_creation_input_tokens,
            cache_read_input_tokens,
            inference_geo,
            speed,
        })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn usage(value: Value) -> Map<String, Value> {
        value.as_object().cloned().unwrap_or_default()
    }

    fn counted(reported: Value, reasoning: Reasoning<'_>, counts: u64) -> Value {
        let reported = ReportedUsage::read(&usage(reported)).expect("readable");
        json!(CompleteUsage::from(calculate(
            reported,
            reasoning,
            None,
            &|_| counts
        )))
    }

    const NO_REASONING: Reasoning<'static> = Reasoning {
        content: None,
        has_thinking_block: false,
    };

    #[rstest]
    #[case::reported_thinking_is_capped_at_the_output(
        json!({"input_tokens": 1, "output_tokens": 4, "output_tokens_details": {"thinking_tokens": 9}}),
        NO_REASONING,
        json!({"reasoning_tokens": 4, "text_tokens": 0})
    )]
    #[case::unreported_thinking_is_estimated_from_its_text(
        json!({"input_tokens": 1, "output_tokens": 10}),
        Reasoning { content: Some("thought"), has_thinking_block: true },
        json!({"reasoning_tokens": 3, "text_tokens": 7})
    )]
    #[case::a_thinking_block_without_text_counts_neither(
        json!({"input_tokens": 1, "output_tokens": 10}),
        Reasoning { content: Some(""), has_thinking_block: true },
        json!({"reasoning_tokens": null, "text_tokens": null})
    )]
    #[case::no_thinking_is_all_text(
        json!({"input_tokens": 1, "output_tokens": 10}),
        NO_REASONING,
        json!({"reasoning_tokens": 0, "text_tokens": 10})
    )]
    fn completion_details_split_the_output_like_python(
        #[case] reported: Value,
        #[case] reasoning: Reasoning<'_>,
        #[case] expected: Value,
    ) {
        assert_eq!(
            counted(reported, reasoning, 3)["completion_tokens_details"],
            expected
        );
    }

    #[rstest]
    #[case::a_float(json!({"input_tokens": 3.7, "output_tokens": 1}), 3)]
    #[case::a_string(json!({"input_tokens": "3", "output_tokens": 1}), 0)]
    #[case::null(json!({"input_tokens": null, "output_tokens": 1}), 0)]
    fn an_unreadable_count_reads_as_python_reads_it(#[case] reported: Value, #[case] prompt: u64) {
        assert_eq!(
            counted(reported, NO_REASONING, 0)["prompt_tokens"],
            json!(prompt)
        );
    }

    #[rstest]
    fn per_iteration_usage_is_refused() {
        let reported = usage(json!({"input_tokens": 1, "output_tokens": 1, "iterations": [{}]}));
        assert!(matches!(
            ReportedUsage::read(&reported),
            Err(Error::UnportedIterations)
        ));
    }

    #[rstest]
    fn a_stream_without_usage_counts_nothing() {
        assert_eq!(StreamedUsage::of(Vec::new()), None);
    }
}
