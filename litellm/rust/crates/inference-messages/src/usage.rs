//! Anthropic Messages usage read into the quantities the pricing core bills. A port of main's
//! `AnthropicConfig.calculate_usage` (`llms/anthropic/chat/transformation.py`) followed by `billed_usage` and
//! `priced_usage` (`litellm_core_utils/llm_cost_calc`), the two steps every `/v1/messages` call priced on main went
//! through, so one reader serves message bodies and streams alike

use litellm_cost::call::{InferenceGeo, Speed};
use litellm_cost::query::{CacheReads, CacheWrites, Modalities, TokenUsage, WebSearchUse};
use litellm_llms_types::formats::messages::{
    CacheCreationUsage, MessagesUsage, Speed as WireSpeed, UsageIteration,
};
use serde_json::{Map, Value};

/// A call's billed quantities and the service tier the provider says it served
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Metered {
    pub(crate) usage: TokenUsage,
    pub(crate) served_tier: Option<String>,
}

/// What a caller of [`metered`] knows besides the usage object
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct UsageContext {
    /// The request asked for `speed: fast`, which main falls back to when the usage names no speed
    pub(crate) requested_fast: bool,
    /// The answer carries a thinking block
    pub(crate) thinking: bool,
}

/// What main prices outside the Rust token path, so a call whose answer uses it is left `Deferred`. Each variant
/// names both the answer that leaves a call unpriced ([`metered`]) and the request that can get such an answer
/// ([`Unpriced::asked_by`]), so a caller that must know before the call reads the same rule
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr, strum::VariantArray)]
pub enum Unpriced {
    /// Thinking text without reported counts, whose tokens main estimates with a tokenizer
    #[strum(serialize = "thinking")]
    Thinking,
    /// Server web searches, which main bills from the response's citations
    #[strum(serialize = "web_search")]
    WebSearch,
}

impl Unpriced {
    /// Whether `request` can get an answer this leaves unpriced: thinking it does not turn off, or a web search
    /// server tool among its tools
    pub fn asked_by(self, request: &Map<String, Value>) -> bool {
        match self {
            Self::Thinking => request
                .get("thinking")
                .filter(|thinking| !thinking.is_null())
                .is_some_and(|thinking| {
                    thinking.get("type").and_then(Value::as_str) != Some("disabled")
                }),
            Self::WebSearch => request
                .get("tools")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|tool| tool.get("type")?.as_str())
                .any(|kind| kind.starts_with("web_search_")),
        }
    }
}

/// The first of [`Unpriced`] that `request` can get an answer priced by, before it is sent
pub fn unpriced(request: &Map<String, Value>) -> Option<Unpriced> {
    <Unpriced as strum::VariantArray>::VARIANTS
        .iter()
        .copied()
        .find(|unpriced| unpriced.asked_by(request))
}

/// The billed quantities of `usage`, or what in it main prices outside the Rust token path
pub(crate) fn metered(usage: &MessagesUsage, context: UsageContext) -> Result<Metered, Unpriced> {
    let iterations = usage.iterations.as_deref().filter(|its| !its.is_empty());
    let counts = match iterations {
        Some(iterations) => Counts::summed(iterations),
        None => Counts::of(usage),
    };
    let reported_thinking = iterations
        .and_then(iteration_thinking_tokens)
        .or_else(|| thinking_tokens(usage));
    if context.thinking && reported_thinking.is_none() {
        return Err(Unpriced::Thinking);
    }
    let web_search = match usage
        .server_tool_use
        .as_ref()
        .and_then(|used| used.web_search_requests)
    {
        Some(0) => Some(WebSearchUse {
            requests: 0,
            context: None,
        }),
        Some(_) => return Err(Unpriced::WebSearch),
        None => None,
    };
    let reasoning = reported_thinking.map_or(0, |thinking| thinking.min(counts.output));
    let writes = cache_writes(
        counts.cache_creation,
        cache_creation_details(usage, iterations),
    );
    let speed = match usage.speed {
        Some(served) => served == WireSpeed::Fast,
        None => context.requested_fast,
    };
    Ok(Metered {
        usage: TokenUsage {
            prompt_tokens: counts.input + counts.cache_creation + counts.cache_read,
            input: Modalities {
                text: counts.input,
                ..Modalities::default()
            },
            cache_read: CacheReads {
                text: counts.cache_read,
                ..CacheReads::default()
            },
            cache_write: writes,
            output: Modalities {
                text: counts.output - reasoning,
                ..Modalities::default()
            },
            reasoning,
            web_search,
            speed: speed.then_some(Speed::Fast),
            inference_geo: usage.inference_geo.as_deref().map(InferenceGeo::new),
            ..TokenUsage::default()
        },
        served_tier: usage.service_tier.clone(),
    })
}

/// The token counts `calculate_usage` reads: the usage's own, or the sums over its iterations when it has any
struct Counts {
    input: u64,
    output: u64,
    cache_creation: u64,
    cache_read: u64,
}

impl Counts {
    fn of(usage: &MessagesUsage) -> Self {
        Self {
            input: usage.input_tokens.unwrap_or(0),
            output: usage.output_tokens.unwrap_or(0),
            cache_creation: usage.cache_creation_input_tokens.unwrap_or(0),
            cache_read: usage.cache_read_input_tokens.unwrap_or(0),
        }
    }

    fn summed(iterations: &[UsageIteration]) -> Self {
        let sum = |count: fn(&UsageIteration) -> Option<u64>| {
            iterations.iter().map(|it| count(it).unwrap_or(0)).sum()
        };
        Self {
            input: sum(|it| it.input_tokens),
            output: sum(|it| it.output_tokens),
            cache_creation: sum(|it| it.cache_creation_input_tokens),
            cache_read: sum(|it| it.cache_read_input_tokens),
        }
    }
}

/// `_resolve_cache_creation_token_details`: the 5m and 1h write counts summed over the iterations that report them,
/// with writes no iteration attributes counted as 5m, or the usage's own `cache_creation`
fn cache_creation_details(
    usage: &MessagesUsage,
    iterations: Option<&[UsageIteration]>,
) -> Option<(u64, u64)> {
    let aggregated = iterations.and_then(|iterations| {
        let detailed: Vec<&CacheCreationUsage> = iterations
            .iter()
            .filter_map(|it| it.cache_creation.as_ref())
            .collect();
        if detailed.is_empty() {
            return None;
        }
        let five_minutes: u64 = detailed
            .iter()
            .map(|it| it.ephemeral_5m_input_tokens.unwrap_or(0))
            .sum();
        let one_hour: u64 = detailed
            .iter()
            .map(|it| it.ephemeral_1h_input_tokens.unwrap_or(0))
            .sum();
        let total: u64 = iterations
            .iter()
            .map(|it| it.cache_creation_input_tokens.unwrap_or(0))
            .sum();
        let undetailed = total.saturating_sub(five_minutes + one_hour);
        Some((five_minutes + undetailed, one_hour))
    });
    aggregated.or_else(|| {
        usage.cache_creation.as_ref().map(|written| {
            (
                written.ephemeral_5m_input_tokens.unwrap_or(0),
                written.ephemeral_1h_input_tokens.unwrap_or(0),
            )
        })
    })
}

/// `billed_usage`: 1h writes bill at the 1h rate, and the 5m writes and every write no lifetime covers at the base rate
fn cache_writes(total: u64, details: Option<(u64, u64)>) -> CacheWrites {
    let (five_minutes, one_hour) = details.unwrap_or((0, 0));
    let unattributed = total.saturating_sub(five_minutes + one_hour);
    CacheWrites {
        default: unattributed + five_minutes,
        one_hour,
        twelve_hours: 0,
    }
}

fn thinking_tokens(usage: &MessagesUsage) -> Option<u64> {
    usage.output_tokens_details.as_ref()?.thinking_tokens
}

/// `_sum_iteration_thinking_tokens`: the sum when every iteration reports its thinking tokens
fn iteration_thinking_tokens(iterations: &[UsageIteration]) -> Option<u64> {
    iterations
        .iter()
        .map(|it| {
            it.extra
                .get("output_tokens_details")?
                .get("thinking_tokens")?
                .as_u64()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use litellm_llms_types::formats::messages::MessagesUsage;
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn usage(value: serde_json::Value) -> MessagesUsage {
        serde_json::from_value(value).unwrap()
    }

    #[rstest]
    #[case::reported_thinking_counts_bill_as_reasoning(true, Some(30))]
    #[case::thinking_text_alone_is_left_to_main(true, None)]
    #[case::no_thinking_bills_every_output_token_as_text(false, None)]
    fn thinking_prices_only_from_reported_counts(
        #[case] thinking: bool,
        #[case] reported: Option<u64>,
    ) {
        let read = usage(json!({
            "input_tokens": 10,
            "output_tokens": 100,
            "output_tokens_details": reported.map(|tokens| json!({"thinking_tokens": tokens})),
        }));
        let metered = metered(
            &read,
            UsageContext {
                thinking,
                ..UsageContext::default()
            },
        );
        match (thinking, reported) {
            (true, None) => assert_eq!(metered, Err(Unpriced::Thinking)),
            _ => {
                let billed = metered.unwrap().usage;
                assert_eq!(billed.reasoning, reported.unwrap_or(0));
                assert_eq!(billed.output.text + billed.reasoning, 100);
            }
        }
    }

    #[rstest]
    fn reported_thinking_beyond_the_output_is_capped_at_the_output() {
        let read =
            usage(json!({"output_tokens": 5, "output_tokens_details": {"thinking_tokens": 9}}));
        let billed = metered(&read, UsageContext::default()).unwrap().usage;
        assert_eq!((billed.reasoning, billed.output.text), (5, 0));
    }

    #[rstest]
    #[case::served_fast(json!("fast"), false, true)]
    #[case::served_standard_overrides_a_fast_request(json!("standard"), true, false)]
    #[case::requested_fast_when_the_usage_is_silent(json!(null), true, true)]
    fn the_served_speed_wins_over_the_requested_one(
        #[case] served: serde_json::Value,
        #[case] requested_fast: bool,
        #[case] fast: bool,
    ) {
        let read = usage(json!({"input_tokens": 1, "output_tokens": 1, "speed": served}));
        let billed = metered(
            &read,
            UsageContext {
                requested_fast,
                ..UsageContext::default()
            },
        )
        .unwrap()
        .usage;
        assert_eq!(billed.speed.is_some(), fast);
    }

    #[rstest]
    #[case::no_search(json!(null), Ok(None))]
    #[case::zero_searches_are_reported(json!({"web_search_requests": 0}), Ok(Some(0)))]
    #[case::searches_are_left_to_main(json!({"web_search_requests": 3}), Err(Unpriced::WebSearch))]
    fn web_search_counts_reach_the_core_only_when_nothing_was_searched(
        #[case] server_tool_use: serde_json::Value,
        #[case] expected: Result<Option<u64>, Unpriced>,
    ) {
        let read = usage(
            json!({"input_tokens": 1, "output_tokens": 1, "server_tool_use": server_tool_use}),
        );
        let metered = metered(&read, UsageContext::default());
        assert_eq!(
            metered.map(|metered| metered.usage.web_search.map(|used| used.requests)),
            expected
        );
    }

    #[rstest]
    fn cache_writes_bill_undetailed_writes_at_the_base_rate() {
        let read = usage(json!({
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 600,
            "cache_creation": {"ephemeral_5m_input_tokens": 300, "ephemeral_1h_input_tokens": 200},
        }));
        let writes = metered(&read, UsageContext::default())
            .unwrap()
            .usage
            .cache_write;
        assert_eq!(writes.default + writes.one_hour, 600);
        assert_eq!(writes.one_hour, 200);
    }

    #[rstest]
    #[case::thinking_turned_on(json!({"thinking": {"type": "enabled", "budget_tokens": 1024}}), Some(Unpriced::Thinking))]
    #[case::adaptive_thinking(json!({"thinking": {"type": "adaptive"}}), Some(Unpriced::Thinking))]
    #[case::thinking_turned_off(json!({"thinking": {"type": "disabled"}}), None)]
    #[case::thinking_sent_as_null(json!({"thinking": null}), None)]
    #[case::the_web_search_server_tool(
        json!({"tools": [{"type": "web_search_20250305", "name": "web_search"}]}),
        Some(Unpriced::WebSearch)
    )]
    #[case::a_later_web_search_version(
        json!({"tools": [{"type": "web_search_20260209", "name": "web_search"}]}),
        Some(Unpriced::WebSearch)
    )]
    #[case::a_client_tool_named_web_search(
        json!({"tools": [{"name": "web_search", "input_schema": {"type": "object"}}]}),
        None
    )]
    #[case::a_server_tool_priced_by_its_tokens(
        json!({"tools": [{"type": "web_fetch_20250910", "name": "web_fetch"}]}),
        None
    )]
    #[case::no_tools(json!({"messages": []}), None)]
    fn a_request_names_what_can_leave_its_answer_unpriced(
        #[case] request: serde_json::Value,
        #[case] expected: Option<Unpriced>,
    ) {
        assert_eq!(unpriced(request.as_object().unwrap()), expected);
    }

    #[rstest]
    #[case::thinking(
        json!({"thinking": {"type": "enabled", "budget_tokens": 1024}}),
        json!({"input_tokens": 1, "output_tokens": 50}),
        true
    )]
    #[case::web_search(
        json!({"tools": [{"type": "web_search_20250305", "name": "web_search"}]}),
        json!({"input_tokens": 1, "output_tokens": 1, "server_tool_use": {"web_search_requests": 1}}),
        false
    )]
    fn an_answer_left_unpriced_was_named_by_its_request(
        #[case] request: serde_json::Value,
        #[case] answered: serde_json::Value,
        #[case] thinking: bool,
    ) {
        let left = metered(
            &usage(answered),
            UsageContext {
                thinking,
                ..UsageContext::default()
            },
        )
        .err();

        assert_eq!(
            (left.is_some(), unpriced(request.as_object().unwrap())),
            (true, left)
        );
    }
}
