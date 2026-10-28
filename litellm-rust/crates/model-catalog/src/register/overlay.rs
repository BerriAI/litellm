use serde_json::{Map, Value};

use crate::wire::Fields;

const TOKEN_COST_FIELDS: [&str; 2] = ["input_cost_per_token", "output_cost_per_token"];

/// main's merge of a registration onto the row it inherits from. A `None` provider on the base is dropped, the
/// registration goes on through `update_dictionary`, and a cache-read price without an explicit one clears the
/// inherited explicit rate
pub fn overlay(base: &Fields, registration: &Fields) -> Fields {
    let base: Fields = base
        .iter()
        .filter(|(name, value)| !(*name == "litellm_provider" && value.is_null()))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    let mut merged = update_dictionary(&base, registration);
    let sets = |name: &str| registration.get(name).is_some_and(|value| !value.is_null());
    if sets("cache_read_input_token_cost") && !sets("explicit_cache_read_input_token_cost") {
        merged.insert(
            "explicit_cache_read_input_token_cost".to_owned(),
            Value::Null,
        );
    }
    merged
}

/// main's `update_dictionary`. A `None` value skips its field, a numeric string becomes a number, and a nested object
/// is merged one level into the existing one
pub fn update_dictionary(base: &Fields, update: &Fields) -> Fields {
    let mut merged = base.clone();
    for (name, value) in update.iter().filter(|(_, value)| !value.is_null()) {
        let value = match (value, merged.get(name)) {
            (Value::String(text), _) => stringified_number(text),
            (Value::Object(nested), Some(Value::Object(existing))) => {
                let mut combined: Map<String, Value> = existing.clone();
                combined.extend(
                    nested
                        .iter()
                        .map(|(key, value)| (key.clone(), value.clone())),
                );
                Value::Object(combined)
            }
            (value, _) => value.clone(),
        };
        merged.insert(name.clone(), value);
    }
    merged
}

/// main's `_convert_stringified_numbers`. A string with `e` or `.` reads as a float, any other as an int, and one that
/// does not parse stays a string
fn stringified_number(text: &str) -> Value {
    let parsed = if text.to_lowercase().contains('e') || text.contains('.') {
        python_float(text)
            .and_then(|number| serde_json::Number::from_f64(number).map(Value::Number))
    } else {
        text.trim().parse::<i64>().ok().map(Value::from)
    };
    parsed.unwrap_or_else(|| Value::String(text.to_owned()))
}

fn python_float(text: &str) -> Option<f64> {
    text.trim()
        .parse::<f64>()
        .ok()
        .filter(|number| number.is_finite())
}

/// main's `_without_overridden_price_variants`. A flat token price is that direction's price above every threshold,
/// for every tier and batch, in every `tiered_pricing` tier and off-peak window, so each inherited rate that would stand
/// in for it is dropped, the reasoning rate included for output, and each tier carries the flat price instead
pub fn without_overridden_price_variants(entry: &Fields, registration: &Fields) -> Fields {
    let flat_prices: Fields = TOKEN_COST_FIELDS
        .iter()
        .filter_map(|field| {
            let value = registration.get(*field).filter(|value| !value.is_null())?;
            Some(((*field).to_owned(), value.clone()))
        })
        .collect();
    let overridden: Vec<&str> = flat_prices
        .keys()
        .flat_map(|field| match field.as_str() {
            "input_cost_per_token" => ["input_cost_per_token"].as_slice(),
            _ => ["output_cost_per_token", "output_cost_per_reasoning_token"].as_slice(),
        })
        .copied()
        .collect();
    if overridden.is_empty() {
        return entry.clone();
    }
    entry
        .iter()
        .filter(|(name, _)| !is_overridden_rate(name, &overridden))
        .map(|(name, value)| {
            (
                name.clone(),
                rate_table_with_flat_prices(name, value, &overridden, &flat_prices),
            )
        })
        .collect()
}

fn is_overridden_rate(name: &str, overridden: &[&str]) -> bool {
    let is_variant = overridden
        .iter()
        .any(|field| name.starts_with(&format!("{field}_")));
    overridden.contains(&name) || (is_variant && !name.contains("_cache_"))
}

fn without_overridden_rates(rates: &Map<String, Value>, overridden: &[&str]) -> Map<String, Value> {
    rates
        .iter()
        .filter(|(name, _)| !is_overridden_rate(name, overridden))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

fn rate_table_with_flat_prices(
    name: &str,
    table: &Value,
    overridden: &[&str],
    flat_prices: &Fields,
) -> Value {
    match (name, table) {
        ("tiered_pricing", Value::Array(tiers)) if tiers.iter().all(Value::is_object) => {
            Value::Array(
                tiers
                    .iter()
                    .filter_map(Value::as_object)
                    .map(|tier| {
                        let mut tier = without_overridden_rates(tier, overridden);
                        tier.extend(
                            flat_prices
                                .iter()
                                .map(|(name, value)| (name.clone(), value.clone())),
                        );
                        Value::Object(tier)
                    })
                    .collect(),
            )
        }
        ("off_peak_pricing", Value::Object(rates)) => {
            Value::Object(without_overridden_rates(rates, overridden))
        }
        _ => table.clone(),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn fields(value: Value) -> Fields {
        let Value::Object(fields) = value else {
            unreachable!()
        };
        fields
    }

    #[rstest]
    #[case::none_skips(json!({"a": 1}), json!({"a": null}), json!({"a": 1}))]
    #[case::float_string(json!({}), json!({"a": "3e-07"}), json!({"a": 3e-07}))]
    #[case::int_string(json!({}), json!({"a": "8192"}), json!({"a": 8192}))]
    #[case::word_stays(json!({}), json!({"a": "chat"}), json!({"a": "chat"}))]
    #[case::nested_merges(json!({"a": {"x": 1, "y": 2}}), json!({"a": {"y": 3}}), json!({"a": {"x": 1, "y": 3}}))]
    fn update_dictionary_follows_main(
        #[case] base: Value,
        #[case] update: Value,
        #[case] expected: Value,
    ) {
        assert_eq!(
            Value::Object(update_dictionary(&fields(base), &fields(update))),
            expected
        );
    }

    #[rstest]
    fn overlay_drops_a_none_provider_and_clears_an_inherited_explicit_cache_rate() {
        let merged = overlay(
            &fields(json!({"litellm_provider": null, "explicit_cache_read_input_token_cost": 1.0})),
            &fields(json!({"cache_read_input_token_cost": 0.5})),
        );
        assert_eq!(
            Value::Object(merged),
            json!({"explicit_cache_read_input_token_cost": null, "cache_read_input_token_cost": 0.5})
        );
    }

    #[rstest]
    fn a_flat_price_replaces_its_variants_in_every_tier() {
        let entry = fields(json!({
            "input_cost_per_token": 1.0,
            "input_cost_per_token_batches": 0.5,
            "input_cost_per_token_cache_hit": 0.1,
            "output_cost_per_token": 2.0,
            "tiered_pricing": [{"range": [0, 10], "input_cost_per_token": 1.0, "output_cost_per_token": 2.0}]
        }));
        let stripped = without_overridden_price_variants(
            &entry,
            &fields(json!({"input_cost_per_token": 9.0})),
        );
        assert_eq!(
            Value::Object(stripped),
            json!({
                "input_cost_per_token_cache_hit": 0.1,
                "output_cost_per_token": 2.0,
                "tiered_pricing": [{"range": [0, 10], "output_cost_per_token": 2.0, "input_cost_per_token": 9.0}]
            })
        );
    }
}
