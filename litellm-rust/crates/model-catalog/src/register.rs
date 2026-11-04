use indexmap::IndexMap;
use litellm_python_compat::{BigInt, number};
use serde_json::Value;

/// One row's fields in the order Python's dict holds them
pub type Fields<V> = IndexMap<String, V>;

const INPUT_COST_PER_TOKEN: &str = "input_cost_per_token";
const OUTPUT_COST_PER_TOKEN: &str = "output_cost_per_token";
const TOKEN_COST_FIELDS: [&str; 2] = [INPUT_COST_PER_TOKEN, OUTPUT_COST_PER_TOKEN];
const CACHE_READ_INPUT_TOKEN_COST: &str = "cache_read_input_token_cost";
const EXPLICIT_CACHE_READ_INPUT_TOKEN_COST: &str = "explicit_cache_read_input_token_cost";
const LITELLM_PROVIDER: &str = "litellm_provider";

/// How the merge rule sees one field value
#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    None,
    Str(String),
    Map,
    Other,
}

/// A number parsed from a stringified registration value
#[derive(Clone, Debug, PartialEq)]
pub enum Number {
    Float(f64),
    Int(BigInt),
}

/// A field value of the host that owns the rows, Python objects in python-bridge, JSON in a JSON host
pub trait FieldValue: Clone {
    fn shape(&self) -> Shape;
    fn none() -> Self;
    /// `None` when the host cannot hold `number`, which keeps the string
    fn number(number: Number) -> Option<Self>;
    /// For a dict `self`, `{**base, **self}` when `base` is a dict, else `dict(self)`
    fn merged_over(&self, base: Option<&Self>) -> Self;
    /// The value as pydantic validates `tuple[Mapping[str, object], ...]`, `None` where it raises
    fn rate_tiers(&self) -> Option<Vec<Fields<Self>>>;
    /// The value as pydantic validates `Mapping[str, object]`, `None` where it raises
    fn rate_table(&self) -> Option<Fields<Self>>;
    fn tier_list(tiers: Vec<Fields<Self>>) -> Self;
    fn table(fields: Fields<Self>) -> Self;
}

/// Main's `update_dictionary`. A `None` value is skipped, a string becomes the number it spells, a dict is merged
/// over a dict already there, and anything else replaces the field. New fields land after the existing ones
pub fn update_dictionary<V: FieldValue>(existing: Fields<V>, new: &Fields<V>) -> Fields<V> {
    new.iter().fold(existing, |mut merged, (name, value)| {
        let updated = match value.shape() {
            Shape::None => return merged,
            Shape::Str(text) => stringified_number(&text)
                .and_then(V::number)
                .unwrap_or_else(|| value.clone()),
            Shape::Map => value.merged_over(merged.get(name)),
            Shape::Other => value.clone(),
        };
        merged.insert(name.clone(), updated);
        merged
    })
}

/// Main's `_convert_stringified_numbers`, `float` when the text has an `e` or a `.`, else `int`
pub fn stringified_number(text: &str) -> Option<Number> {
    if text.contains(['e', 'E', '.']) {
        return number::float(text).map(Number::Float);
    }
    number::int(text).map(Number::Int)
}

/// The row a `register_model` call writes, its `base` row with `registration` merged over it. A `litellm_provider`
/// of `None` in `base` is dropped so the registration's provider or none is kept
pub fn registration_row<V: FieldValue>(base: &Fields<V>, registration: &Fields<V>) -> Fields<V> {
    let existing: Fields<V> = base
        .iter()
        .filter(|(name, value)| *name != LITELLM_PROVIDER || value.shape() != Shape::None)
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    let merged = update_dictionary(existing, registration);
    base_fields_replaced_by(registration)
        .iter()
        .fold(merged, |mut row, name| {
            row.insert((*name).to_owned(), V::none());
            row
        })
}

/// Fields of the base row a registration clears, so a flat cache read price is not shadowed by the base row's
/// explicit one
pub fn base_fields_replaced_by<V: FieldValue>(registration: &Fields<V>) -> &'static [&'static str] {
    if is_set(registration, CACHE_READ_INPUT_TOKEN_COST)
        && !is_set(registration, EXPLICIT_CACHE_READ_INPUT_TOKEN_COST)
    {
        return &[EXPLICIT_CACHE_READ_INPUT_TOKEN_COST];
    }
    &[]
}

/// Main's `_without_synthesized_costs`. `get_model_info` fills a missing per-token price with `0`, so a token price
/// field survives only when the raw row has it or the registration names it
pub fn without_synthesized_costs<V: FieldValue>(
    model_info: &Fields<V>,
    raw_fields: impl Fn(&str) -> bool,
    registration: &Fields<V>,
) -> Fields<V> {
    model_info
        .iter()
        .filter(|(name, _)| {
            !TOKEN_COST_FIELDS.contains(&name.as_str())
                || raw_fields(name)
                || registration.contains_key(*name)
        })
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

/// Main's `_without_overridden_price_variants`. A flat per-token price in `registration` is that direction's price
/// everywhere, so every inherited rate that would stand in for it is dropped and each `tiered_pricing` tier carries
/// the flat price
pub fn without_overridden_price_variants<V: FieldValue>(
    entry: &Fields<V>,
    registration: &Fields<V>,
) -> Fields<V> {
    let flat_prices: Fields<V> = TOKEN_COST_FIELDS
        .iter()
        .filter_map(|name| {
            registration
                .get(*name)
                .filter(|value| value.shape() != Shape::None)
                .map(|value| ((*name).to_owned(), value.clone()))
        })
        .collect();
    let overridden: Vec<&str> = flat_prices
        .keys()
        .flat_map(|name| rate_fields_overridden_by(name))
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

/// Main's `model_info_with_pricing_overrides` before its pydantic validation, `overrides` already narrowed to
/// server derived pricing keys
pub fn with_pricing_overrides<V: FieldValue>(
    model_info: &Fields<V>,
    overrides: &Fields<V>,
) -> Fields<V> {
    update_dictionary(
        without_overridden_price_variants(model_info, overrides),
        overrides,
    )
}

fn is_set<V: FieldValue>(fields: &Fields<V>, name: &str) -> bool {
    fields
        .get(name)
        .is_some_and(|value| value.shape() != Shape::None)
}

fn rate_fields_overridden_by(name: &str) -> &'static [&'static str] {
    match name {
        INPUT_COST_PER_TOKEN => &[INPUT_COST_PER_TOKEN],
        OUTPUT_COST_PER_TOKEN => &[OUTPUT_COST_PER_TOKEN, "output_cost_per_reasoning_token"],
        _ => &[],
    }
}

fn is_overridden_rate(name: &str, overridden: &[&str]) -> bool {
    let is_variant = overridden.iter().any(|rate| {
        name.strip_prefix(rate)
            .is_some_and(|rest| rest.starts_with('_'))
    });
    overridden.contains(&name) || (is_variant && !name.contains("_cache_"))
}

fn without_overridden_rates<V: FieldValue>(rates: &Fields<V>, overridden: &[&str]) -> Fields<V> {
    rates
        .iter()
        .filter(|(name, _)| !is_overridden_rate(name, overridden))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

fn rate_table_with_flat_prices<V: FieldValue>(
    name: &str,
    table: &V,
    overridden: &[&str],
    flat_prices: &Fields<V>,
) -> V {
    let rewritten = match name {
        "tiered_pricing" => table.rate_tiers().map(|tiers| {
            V::tier_list(
                tiers
                    .iter()
                    .map(|tier| {
                        let mut priced = without_overridden_rates(tier, overridden);
                        priced.extend(flat_prices.clone());
                        priced
                    })
                    .collect(),
            )
        }),
        "off_peak_pricing" => table
            .rate_table()
            .map(|rates| V::table(without_overridden_rates(&rates, overridden))),
        _ => None,
    };
    rewritten.unwrap_or_else(|| table.clone())
}

impl FieldValue for Value {
    fn shape(&self) -> Shape {
        match self {
            Value::Null => Shape::None,
            Value::String(text) => Shape::Str(text.clone()),
            Value::Object(_) => Shape::Map,
            _ => Shape::Other,
        }
    }

    fn none() -> Self {
        Value::Null
    }

    fn number(number: Number) -> Option<Self> {
        match number {
            Number::Float(value) => serde_json::Number::from_f64(value).map(Value::Number),
            Number::Int(value) => i64::try_from(&value)
                .map(Value::from)
                .or_else(|_| u64::try_from(&value).map(Value::from))
                .ok(),
        }
    }

    fn merged_over(&self, base: Option<&Self>) -> Self {
        match (base, self) {
            (Some(Value::Object(base)), Value::Object(over)) => {
                let mut merged = base.clone();
                merged.extend(over.clone());
                Value::Object(merged)
            }
            _ => self.clone(),
        }
    }

    fn rate_tiers(&self) -> Option<Vec<Fields<Self>>> {
        let Value::Array(tiers) = self else {
            return None;
        };
        tiers.iter().map(Value::rate_table).collect()
    }

    fn rate_table(&self) -> Option<Fields<Self>> {
        let Value::Object(rates) = self else {
            return None;
        };
        Some(
            rates
                .iter()
                .map(|(name, rate)| (name.clone(), rate.clone()))
                .collect(),
        )
    }

    fn tier_list(tiers: Vec<Fields<Self>>) -> Self {
        Value::Array(tiers.into_iter().map(Value::table).collect())
    }

    fn table(fields: Fields<Self>) -> Self {
        Value::Object(fields.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn fields(value: Value) -> Fields<Value> {
        value.rate_table().expect("a JSON object")
    }

    fn written(row: Fields<Value>) -> String {
        Value::table(row).to_string()
    }

    /// Each expected row is what main's `update_dictionary` returns for the same JSON, keys in dict order
    #[rstest]
    #[case::none_is_skipped(json!({"a": 1}), json!({"a": null, "b": null}), r#"{"a":1}"#)]
    #[case::scientific_string(json!({}), json!({"input_cost_per_token": "3e-07"}), r#"{"input_cost_per_token":3e-7}"#)]
    #[case::integer_string(json!({}), json!({"max_tokens": "8192"}), r#"{"max_tokens":8192}"#)]
    #[case::padded_integer_string(json!({}), json!({"max_tokens": " 8_192 "}), r#"{"max_tokens":8192}"#)]
    #[case::point_makes_a_float(json!({}), json!({"max_tokens": "8192."}), r#"{"max_tokens":8192.0}"#)]
    #[case::text_stays(json!({}), json!({"mode": "chat"}), r#"{"mode":"chat"}"#)]
    #[case::e_without_a_number_stays(json!({}), json!({"litellm_provider": "openai"}), r#"{"litellm_provider":"openai"}"#)]
    #[case::infinity_without_e_or_point_stays(json!({}), json!({"x": "inf"}), r#"{"x":"inf"}"#)]
    #[case::dict_merges_over_dict(json!({"m": {"a": 1, "b": 2}}), json!({"m": {"b": 3, "c": 4}}), r#"{"m":{"a":1,"b":3,"c":4}}"#)]
    #[case::dict_replaces_a_scalar(json!({"m": 1}), json!({"m": {"b": 3}}), r#"{"m":{"b":3}}"#)]
    #[case::nested_strings_are_kept(json!({}), json!({"m": {"b": "3"}}), r#"{"m":{"b":"3"}}"#)]
    #[case::other_values_replace(json!({"a": 1, "b": true}), json!({"b": [1, "2"], "c": 0.5}), r#"{"a":1,"b":[1,"2"],"c":0.5}"#)]
    #[case::existing_order_then_new(json!({"z": 1, "a": 2}), json!({"m": 3, "z": 4}), r#"{"z":4,"a":2,"m":3}"#)]
    fn update_dictionary_matches_main(
        #[case] existing: Value,
        #[case] new: Value,
        #[case] expected: &str,
    ) {
        assert_eq!(
            written(update_dictionary(fields(existing), &fields(new))),
            expected
        );
    }

    #[rstest]
    #[case::provider_none_is_dropped(
        json!({"litellm_provider": null, "max_tokens": 10}),
        json!({"input_cost_per_token": 1}),
        r#"{"max_tokens":10,"input_cost_per_token":1}"#
    )]
    #[case::provider_none_gives_way_to_the_registration(
        json!({"litellm_provider": null}),
        json!({"litellm_provider": "openai"}),
        r#"{"litellm_provider":"openai"}"#
    )]
    #[case::cache_read_clears_explicit(
        json!({"explicit_cache_read_input_token_cost": 1, "a": 1}),
        json!({"cache_read_input_token_cost": 2}),
        r#"{"explicit_cache_read_input_token_cost":null,"a":1,"cache_read_input_token_cost":2}"#
    )]
    #[case::explicit_cache_read_is_kept_when_registered(
        json!({}),
        json!({"cache_read_input_token_cost": 2, "explicit_cache_read_input_token_cost": 3}),
        r#"{"cache_read_input_token_cost":2,"explicit_cache_read_input_token_cost":3}"#
    )]
    #[case::cache_read_none_clears_nothing(
        json!({"explicit_cache_read_input_token_cost": 1}),
        json!({"cache_read_input_token_cost": null}),
        r#"{"explicit_cache_read_input_token_cost":1}"#
    )]
    fn registration_row_matches_main(
        #[case] base: Value,
        #[case] registration: Value,
        #[case] expected: &str,
    ) {
        assert_eq!(
            written(registration_row(&fields(base), &fields(registration))),
            expected
        );
    }

    #[rstest]
    #[case::no_flat_price(
        json!({"input_cost_per_token_above_200k_tokens": 1}),
        json!({"cache_read_input_token_cost": 1}),
        r#"{"input_cost_per_token_above_200k_tokens":1}"#
    )]
    #[case::input_variants_dropped_cache_kept(
        json!({"input_cost_per_token": 1, "input_cost_per_token_above_200k_tokens": 2, "input_cost_per_token_cache_hit": 3, "output_cost_per_token_priority": 4}),
        json!({"input_cost_per_token": 9}),
        r#"{"input_cost_per_token_cache_hit":3,"output_cost_per_token_priority":4}"#
    )]
    #[case::output_drops_reasoning(
        json!({"output_cost_per_reasoning_token": 1, "output_cost_per_reasoning_token_batches": 2, "output_cost_per_token_batches": 3}),
        json!({"output_cost_per_token": 9}),
        r#"{}"#
    )]
    #[case::none_price_overrides_nothing(
        json!({"input_cost_per_token_priority": 1}),
        json!({"input_cost_per_token": null}),
        r#"{"input_cost_per_token_priority":1}"#
    )]
    #[case::name_without_underscore_is_not_a_variant(
        json!({"input_cost_per_tokens": 1}),
        json!({"input_cost_per_token": 9}),
        r#"{"input_cost_per_tokens":1}"#
    )]
    #[case::tiers_carry_the_flat_price(
        json!({"tiered_pricing": [{"input_cost_per_token": 1, "output_cost_per_token": 2, "range": [0, 10]}]}),
        json!({"input_cost_per_token": "9"}),
        r#"{"tiered_pricing":[{"output_cost_per_token":2,"range":[0,10],"input_cost_per_token":"9"}]}"#
    )]
    #[case::invalid_tiers_are_kept(
        json!({"tiered_pricing": [1, {"input_cost_per_token": 1}]}),
        json!({"input_cost_per_token": 9}),
        r#"{"tiered_pricing":[1,{"input_cost_per_token":1}]}"#
    )]
    #[case::off_peak_drops_overridden(
        json!({"off_peak_pricing": {"input_cost_per_token": 1, "hours": [1, 2]}}),
        json!({"input_cost_per_token": 9}),
        r#"{"off_peak_pricing":{"hours":[1,2]}}"#
    )]
    #[case::invalid_off_peak_is_kept(
        json!({"off_peak_pricing": [1]}),
        json!({"input_cost_per_token": 9}),
        r#"{"off_peak_pricing":[1]}"#
    )]
    fn without_overridden_price_variants_matches_main(
        #[case] entry: Value,
        #[case] registration: Value,
        #[case] expected: &str,
    ) {
        assert_eq!(
            written(without_overridden_price_variants(
                &fields(entry),
                &fields(registration)
            )),
            expected
        );
    }

    #[rstest]
    #[case::synthesized_dropped(json!({}), json!({}), r#"{"key":"m","mode":"chat"}"#)]
    #[case::raw_price_kept(json!({"input_cost_per_token": 1}), json!({}), r#"{"key":"m","input_cost_per_token":0,"mode":"chat"}"#)]
    #[case::registered_price_kept(json!({}), json!({"output_cost_per_token": null}), r#"{"key":"m","output_cost_per_token":0,"mode":"chat"}"#)]
    fn without_synthesized_costs_matches_main(
        #[case] raw: Value,
        #[case] registration: Value,
        #[case] expected: &str,
    ) {
        let model_info = fields(
            json!({"key": "m", "input_cost_per_token": 0, "output_cost_per_token": 0, "mode": "chat"}),
        );
        let raw = fields(raw);
        assert_eq!(
            written(without_synthesized_costs(
                &model_info,
                |name| raw.contains_key(name),
                &fields(registration)
            )),
            expected
        );
    }

    #[rstest]
    fn pricing_overrides_replace_variants_then_merge() {
        let model_info = fields(
            json!({"input_cost_per_token": 1, "input_cost_per_token_above_200k_tokens": 2, "mode": "chat"}),
        );
        let overrides = fields(json!({"input_cost_per_token": "5e-07"}));
        assert_eq!(
            written(with_pricing_overrides(&model_info, &overrides)),
            r#"{"mode":"chat","input_cost_per_token":5e-7}"#
        );
    }

    #[rstest]
    fn a_number_json_cannot_hold_keeps_the_string() {
        let row = update_dictionary(
            Fields::new(),
            &fields(json!({"a": "1e400", "b": "99999999999999999999999"})),
        );
        assert_eq!(
            written(row),
            r#"{"a":"1e400","b":"99999999999999999999999"}"#
        );
    }
}
