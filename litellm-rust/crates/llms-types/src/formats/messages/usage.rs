use serde_json::{Map, Value};

use crate::recognized::Recognized;

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct ServerToolUsage {
    pub web_search_requests: Option<u64>,
    pub web_fetch_requests: Option<u64>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct UsageIteration {
    #[serde(rename = "type")]
    pub iteration_type: Recognized<UsageIterationType>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cache_creation_input_tokens: Option<u64>,
    pub cache_read_input_tokens: Option<u64>,
    pub cache_creation: Option<CacheCreationUsage>,
    pub model: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(rename_all = "snake_case")]
pub enum UsageIterationType {
    Compaction,
    Message,
    AdvisorMessage,
    FallbackMessage,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct MessagesUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cache_creation_input_tokens: Option<u64>,
    pub cache_read_input_tokens: Option<u64>,
    pub server_tool_use: Option<ServerToolUsage>,
    pub cache_creation: Option<CacheCreationUsage>,
    pub output_tokens_details: Option<MessagesOutputTokensDetails>,
    pub service_tier: Option<String>,
    pub inference_geo: Option<String>,
    pub speed: Option<super::Speed>,
    pub iterations: Option<Vec<UsageIteration>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct CacheCreationUsage {
    pub ephemeral_1h_input_tokens: Option<u64>,
    pub ephemeral_5m_input_tokens: Option<u64>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Default)]
pub struct MessagesOutputTokensDetails {
    pub thinking_tokens: Option<u64>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[cfg(test)]
mod tests {

    use crate::formats::messages::MessagesUsage;

    use crate::formats::messages::UsageIterationType;

    use crate::{recognized::Recognized, test_support::*};

    use rstest::rstest;

    use serde_json::json;

    #[rstest]
    fn usage_contracts_round_trip() {
        let usage = round_trip::<MessagesUsage>(json!({
            "input_tokens":10,
            "output_tokens":4,
            "server_tool_use":{"web_search_requests":2,"web_fetch_requests":1},
            "cache_creation":{"ephemeral_1h_input_tokens":3,"ephemeral_5m_input_tokens":1},
            "output_tokens_details":{"thinking_tokens":2},
            "iterations":[
                {"type":"compaction","input_tokens":7,"output_tokens":1},
                {"type":"message","input_tokens":3,"output_tokens":3}
            ],
            "service_tier":"priority",
            "inference_geo":"global",
            "speed":"fast"
        }));
        assert_eq!(usage.inference_geo.as_deref(), Some("global"));
        assert_eq!(usage.input_tokens, Some(10));
        assert_eq!(usage.output_tokens, Some(4));
        assert!(usage.extra.is_empty());
        let server = usage.server_tool_use.as_ref().unwrap();
        assert_eq!(server.web_search_requests, Some(2));
        assert_eq!(server.web_fetch_requests, Some(1));
        let cache = usage.cache_creation.as_ref().unwrap();
        assert_eq!(cache.ephemeral_1h_input_tokens, Some(3));
        assert_eq!(cache.ephemeral_5m_input_tokens, Some(1));
        assert_eq!(
            usage
                .output_tokens_details
                .as_ref()
                .unwrap()
                .thinking_tokens,
            Some(2)
        );
        let [compaction, message] = usage.iterations.as_ref().unwrap().as_slice() else {
            panic!("expected usage iterations");
        };
        assert_eq!(
            compaction.iteration_type,
            Recognized::Known(UsageIterationType::Compaction)
        );
        assert_eq!(
            message.iteration_type,
            Recognized::Known(UsageIterationType::Message)
        );
        assert_eq!(
            compaction.input_tokens.unwrap() + message.input_tokens.unwrap(),
            usage.input_tokens.unwrap()
        );
        assert_eq!(
            compaction.output_tokens.unwrap() + message.output_tokens.unwrap(),
            usage.output_tokens.unwrap()
        );
    }

    #[rstest]
    fn usage_iterations_expose_advisor_and_fallback_models() {
        let usage = round_trip::<MessagesUsage>(json!({
            "iterations":[
                {"type":"advisor_message","model":"claude-opus-5-5","input_tokens":5,"output_tokens":2,
                 "cache_creation_input_tokens":4,"cache_read_input_tokens":1,
                 "cache_creation":{"ephemeral_1h_input_tokens":0,"ephemeral_5m_input_tokens":4}},
                {"type":"fallback_message","model":"claude-sonnet-5-5","input_tokens":6,"output_tokens":3,
                 "cache_creation_input_tokens":0,"cache_read_input_tokens":2},
                {"type":"future_iteration","input_tokens":1}
            ]
        }));
        let [advisor, fallback, future] = usage.iterations.as_deref().unwrap() else {
            panic!("expected three iterations");
        };
        assert_eq!(
            advisor.iteration_type,
            Recognized::Known(UsageIterationType::AdvisorMessage)
        );
        assert_eq!(advisor.model.as_deref(), Some("claude-opus-5-5"));
        assert_eq!(
            (
                advisor.cache_creation_input_tokens,
                advisor.cache_read_input_tokens
            ),
            (Some(4), Some(1))
        );
        assert_eq!(
            advisor
                .cache_creation
                .as_ref()
                .unwrap()
                .ephemeral_5m_input_tokens,
            advisor.cache_creation_input_tokens
        );
        assert!(advisor.extra.is_empty());
        assert_eq!(
            fallback.iteration_type,
            Recognized::Known(UsageIterationType::FallbackMessage)
        );
        assert_eq!(fallback.model.as_deref(), Some("claude-sonnet-5-5"));
        assert_eq!(fallback.cache_read_input_tokens, Some(2));
        assert!(fallback.extra.is_empty());
        assert_eq!(
            future.iteration_type,
            Recognized::Unrecognized(json!("future_iteration"))
        );
        assert_eq!(future.input_tokens, Some(1));
    }
}
