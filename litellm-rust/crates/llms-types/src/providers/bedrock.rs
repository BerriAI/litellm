use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::recognized::{Recognized, deserialize_present};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BedrockInvocationMetrics {
    #[serde(
        default,
        deserialize_with = "deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub input_token_count: Option<Recognized<u64>>,
    #[serde(
        default,
        deserialize_with = "deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub output_token_count: Option<Recognized<u64>>,
    #[serde(
        default,
        deserialize_with = "deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_read_input_token_count: Option<Recognized<u64>>,
    #[serde(
        default,
        deserialize_with = "deserialize_present",
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_write_input_token_count: Option<Recognized<u64>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::BedrockInvocationMetrics;
    use crate::recognized::Recognized;
    use rstest::rstest;
    use serde_json::{Value, json};

    #[rstest]
    fn invocation_metrics_expose_token_counts_and_preserve_extensions() {
        let wire = json!({
            "inputTokenCount": 1,
            "outputTokenCount": 2,
            "cacheReadInputTokenCount": 3,
            "cacheWriteInputTokenCount": 4,
            "futureMetric": {"value": 5}
        });
        let metrics: BedrockInvocationMetrics = serde_json::from_value(wire.clone()).unwrap();

        assert_eq!(metrics.input_token_count, Some(Recognized::Known(1)));
        assert_eq!(metrics.output_token_count, Some(Recognized::Known(2)));
        assert_eq!(
            metrics.cache_read_input_token_count,
            Some(Recognized::Known(3))
        );
        assert_eq!(
            metrics.cache_write_input_token_count,
            Some(Recognized::Known(4))
        );
        assert_eq!(
            metrics.extra,
            serde_json::Map::from_iter([("futureMetric".into(), json!({"value": 5}))])
        );
        assert_eq!(serde_json::to_value(metrics).unwrap(), wire);
    }

    #[rstest]
    #[case::zero(json!(0), Some(0))]
    #[case::maximum(json!(u64::MAX), Some(u64::MAX))]
    #[case::null(Value::Null, None)]
    #[case::negative(json!(-1), None)]
    #[case::fractional(json!(1.5), None)]
    #[case::string(json!("3"), None)]
    #[case::object(json!({"count": 3}), None)]
    fn token_counts_preserve_unrecognized_values(#[case] value: Value, #[case] known: Option<u64>) {
        let wire = json!({
            "inputTokenCount": value,
            "outputTokenCount": value,
            "cacheReadInputTokenCount": value,
            "cacheWriteInputTokenCount": value
        });
        let metrics: BedrockInvocationMetrics = serde_json::from_value(wire.clone()).unwrap();
        let expected = Some(match known {
            Some(count) => Recognized::Known(count),
            None => Recognized::Unrecognized(value),
        });

        assert_eq!(metrics.input_token_count, expected);
        assert_eq!(metrics.output_token_count, expected);
        assert_eq!(metrics.cache_read_input_token_count, expected);
        assert_eq!(metrics.cache_write_input_token_count, expected);
        assert_eq!(serde_json::to_value(metrics).unwrap(), wire);
    }

    #[rstest]
    fn missing_counts_remain_absent() {
        let metrics: BedrockInvocationMetrics = serde_json::from_value(json!({})).unwrap();

        assert_eq!(metrics, BedrockInvocationMetrics::default());
        assert_eq!(serde_json::to_value(metrics).unwrap(), json!({}));
    }
}
