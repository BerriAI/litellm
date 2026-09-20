use serde_json::Value;

#[derive(Clone, Debug, Default)]
pub struct TokenPrices {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_creation: f64,
    pub cache_creation_1hr: f64,
}

impl TokenPrices {
    pub fn from_model_info(info: &Value, usage: &Value) -> Option<Self> {
        let input = info.get("input_cost_per_token")?.as_f64()?;
        let output = info.get("output_cost_per_token")?.as_f64()?;
        let total_input = [
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ]
        .iter()
        .map(|key| count(usage, key))
        .sum::<f64>();
        let threshold = info
            .as_object()?
            .keys()
            .filter_map(|key| {
                let threshold = key
                    .strip_prefix("input_cost_per_token_above_")?
                    .strip_suffix("_tokens")?;
                let number = threshold
                    .strip_suffix('k')
                    .unwrap_or(threshold)
                    .parse::<f64>()
                    .ok()?
                    * if threshold.ends_with('k') {
                        1000.0
                    } else {
                        1.0
                    };
                (total_input > number).then_some((number, threshold))
            })
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, suffix)| suffix);
        let tier = usage
            .get("service_tier")
            .and_then(Value::as_str)
            .filter(|tier| matches!(*tier, "priority" | "flex"));
        let rate = |key: &str| -> Option<f64> {
            let base = tier
                .and_then(|tier| info.get(format!("{key}_{tier}")).and_then(Value::as_f64))
                .or_else(|| info.get(key).and_then(Value::as_f64));
            threshold
                .and_then(|threshold| {
                    let key = format!("{key}_above_{threshold}_tokens");
                    tier.and_then(|tier| info.get(format!("{key}_{tier}")).and_then(Value::as_f64))
                        .or_else(|| info.get(key).and_then(Value::as_f64))
                })
                .or(base)
        };
        let input = rate("input_cost_per_token").unwrap_or(input);
        let creation = rate("cache_creation_input_token_cost").unwrap_or(input);
        Some(Self {
            input,
            output: rate("output_cost_per_token").unwrap_or(output),
            cache_read: rate("cache_read_input_token_cost").unwrap_or(input),
            cache_creation: creation,
            cache_creation_1hr: rate("cache_creation_input_token_cost_above_1hr")
                .unwrap_or(creation),
        })
    }

    pub fn messages_cost(&self, usage: &Value) -> f64 {
        let creation = match usage
            .get("cache_creation")
            .filter(|value| value.is_object())
        {
            Some(details) => {
                count(details, "ephemeral_5m_input_tokens") * self.cache_creation
                    + count(details, "ephemeral_1h_input_tokens") * self.cache_creation_1hr
            }
            None => count(usage, "cache_creation_input_tokens") * self.cache_creation,
        };
        count(usage, "input_tokens") * self.input
            + count(usage, "output_tokens") * self.output
            + count(usage, "cache_read_input_tokens") * self.cache_read
            + creation
    }
}

fn count(value: &Value, key: &str) -> f64 {
    value.get(key).and_then(Value::as_f64).unwrap_or_default()
}

pub fn messages_cost(info: &Value, usage: &Value) -> Option<f64> {
    let prices = TokenPrices::from_model_info(info, usage)?;
    let provider = info.get("provider_specific_entry").unwrap_or(&Value::Null);
    let speed = if usage.get("speed").and_then(Value::as_str) == Some("fast") {
        provider.get("fast").and_then(Value::as_f64).unwrap_or(1.0)
    } else {
        1.0
    };
    let geo = usage
        .get("inference_geo")
        .and_then(Value::as_str)
        .and_then(|geo| provider.get(geo))
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    let searches = usage
        .pointer("/server_tool_use/web_search_requests")
        .and_then(Value::as_f64)
        .unwrap_or_default();
    let search_rate = info
        .pointer("/search_context_cost_per_query/search_context_size_medium")
        .and_then(Value::as_f64)
        .unwrap_or_default();
    Some(prices.messages_cost(usage) * speed * geo + searches * search_rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn accounts_for_each_input_category_once_and_preserves_zero_prices() {
        let info = json!({"input_cost_per_token": 2.0, "output_cost_per_token": 3.0,
            "cache_read_input_token_cost": 0.0, "cache_creation_input_token_cost": 4.0,
            "cache_creation_input_token_cost_above_1hr": 5.0});
        let usage = json!({"input_tokens": 10, "output_tokens": 3,
            "cache_read_input_tokens": 100, "cache_creation_input_tokens": 7,
            "cache_creation": {"ephemeral_5m_input_tokens": 2, "ephemeral_1h_input_tokens": 5}});
        assert_eq!(messages_cost(&info, &usage), Some(62.0));
    }

    #[test]
    fn threshold_uses_total_input_and_is_strict() {
        let info = json!({"input_cost_per_token": 1.0, "output_cost_per_token": 2.0,
            "input_cost_per_token_above_100_tokens": 4.0});
        assert_eq!(
            messages_cost(&info, &json!({"input_tokens": 100})),
            Some(100.0)
        );
        assert_eq!(
            messages_cost(
                &info,
                &json!({"input_tokens": 100, "cache_read_input_tokens": 1})
            ),
            Some(404.0)
        );
        assert_eq!(
            messages_cost(&json!({}), &json!({"input_tokens": 100})),
            None
        );
    }
}
