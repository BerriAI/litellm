use serde_json::{Map, Value};

use crate::recognized::Recognized;
use crate::serde_compat::Nullable;
use crate::serde_compat::deserialize_present;

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct ServerToolUsage {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub web_search_requests: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub web_fetch_requests: Option<Recognized<u64>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
pub struct UsageIteration {
    #[serde(rename = "type")]
    pub iteration_type: UsageIterationType,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub input_tokens: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub output_tokens: Option<Recognized<u64>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(rename_all = "snake_case")]
pub enum UsageIterationType {
    Compaction,
    Message,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct MessagesUsage {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub input_tokens: Option<Nullable<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub output_tokens: Option<Nullable<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub cache_creation_input_tokens: Option<Nullable<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub cache_read_input_tokens: Option<Nullable<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub server_tool_use: Option<Recognized<ServerToolUsage>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub cache_creation: Option<Recognized<CacheCreationUsage>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub output_tokens_details: Option<Recognized<MessagesOutputTokensDetails>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub service_tier: Option<Recognized<String>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub speed: Option<Recognized<super::Speed>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub iterations: Option<Recognized<Vec<Recognized<UsageIteration>>>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct CacheCreationUsage {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub ephemeral_1h_input_tokens: Option<Recognized<u64>>,
    #[serde(default, deserialize_with = "deserialize_present")]
    pub ephemeral_5m_input_tokens: Option<Recognized<u64>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[serde_with::skip_serializing_none]
#[macro_rules_attribute::apply(wire_type)]
#[derive(Default)]
pub struct MessagesOutputTokensDetails {
    #[serde(default, deserialize_with = "deserialize_present")]
    pub thinking_tokens: Option<Recognized<u64>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::omitted(None, json!({}))]
    #[case::null(Some(Nullable::Null), json!({"input_tokens":null,"output_tokens":null,"cache_creation_input_tokens":null,"cache_read_input_tokens":null}))]
    #[case::zero(Some(Nullable::Value(0)), json!({"input_tokens":0,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}))]
    fn counts_preserve_missing_null_and_zero(
        #[case] count: Option<Nullable<u64>>,
        #[case] wire: Value,
    ) {
        let usage = MessagesUsage {
            input_tokens: count,
            output_tokens: count,
            cache_creation_input_tokens: count,
            cache_read_input_tokens: count,
            ..Default::default()
        };
        assert_eq!(serde_json::to_value(&usage).unwrap(), wire);
        assert_eq!(
            serde_json::from_value::<MessagesUsage>(wire).unwrap(),
            usage
        );
    }

    #[rstest]
    fn extended_usage_keeps_cache_durations_iterations_and_opaque_counters() {
        let usage = MessagesUsage {
            input_tokens: Some(Nullable::Value(3)),
            output_tokens: Some(Nullable::Value(2)),
            cache_creation: Some(Recognized::Known(CacheCreationUsage {
                ephemeral_1h_input_tokens: Some(Recognized::Known(5)),
                ephemeral_5m_input_tokens: Some(Recognized::Known(7)),
                extra: Map::new(),
            })),
            server_tool_use: Some(Recognized::Known(ServerToolUsage {
                web_search_requests: Some(Recognized::Known(1)),
                web_fetch_requests: Some(Recognized::Unrecognized(Value::Null)),
                extra: Map::from_iter([("future_tool".into(), json!({"count":null}))]),
            })),
            output_tokens_details: Some(Recognized::Known(MessagesOutputTokensDetails {
                thinking_tokens: Some(Recognized::Known(2)),
                extra: Map::new(),
            })),
            iterations: Some(Recognized::Known(vec![
                Recognized::Known(UsageIteration {
                    iteration_type: UsageIterationType::Compaction,
                    input_tokens: Some(Recognized::Known(3)),
                    output_tokens: Some(Recognized::Known(1)),
                    extra: Map::new(),
                }),
                Recognized::Unrecognized(json!({"type":"future","input_tokens":"many"})),
            ])),
            ..Default::default()
        };
        let wire = json!({"input_tokens":3,"output_tokens":2,"cache_creation":{"ephemeral_1h_input_tokens":5,"ephemeral_5m_input_tokens":7},"server_tool_use":{"web_search_requests":1,"web_fetch_requests":null,"future_tool":{"count":null}},"output_tokens_details":{"thinking_tokens":2},"iterations":[{"type":"compaction","input_tokens":3,"output_tokens":1},{"type":"future","input_tokens":"many"}]});
        assert_eq!(serde_json::to_value(&usage).unwrap(), wire);
        assert_eq!(
            serde_json::from_value::<MessagesUsage>(wire).unwrap(),
            usage
        );
    }
}
