use std::collections::BTreeMap;

use litellm_cost::money::{Multiplier, UnitPrice};
use schemars::{JsonSchema, Schema};
use serde_json::{Map, Value, json};

use crate::fallback::FallbackGeneralizations;
use crate::wire::{Attributes, OffPeakWindow, SearchContextCostPerQuery, UtcHours};

#[derive(JsonSchema)]
#[schemars(rename = "ModelEntry")]
struct ModelEntry {
    #[serde(flatten)]
    _attributes: Attributes,
    #[serde(flatten)]
    _blocks: PriceBlocks,
}

#[derive(JsonSchema)]
#[allow(dead_code)]
struct PriceBlocks {
    /// USD cost per billable guardrail unit, keyed by the provider's usage counter name (e.g. Bedrock's contentPolicyUnits).
    guardrail_cost_per_unit: Option<BTreeMap<String, UnitPrice>>,
    /// Rates that replace the same-named base fields while the request falls inside the stated UTC windows.
    off_peak_pricing: Option<OffPeakPricing>,
    /// Multiplier applied to all token costs when served from a non-global Vertex AI endpoint (e.g. 1.10 = +10%).
    regional_endpoint_uplift_multiplier: Option<Multiplier>,
    /// Multiplier applied to all token costs for EU data residency (e.g. 1.10 = +10%).
    regional_processing_uplift_multiplier_eu: Option<Multiplier>,
    /// Multiplier applied to all token costs for US data residency (e.g. 1.10 = +10%).
    regional_processing_uplift_multiplier_us: Option<Multiplier>,
    /// USD cost per web search query, keyed by search context size.
    search_context_cost_per_query: Option<SearchContextCostPerQuery>,
    /// Context-length or result-count tiered rates; each tier's costs apply within its range.
    tiered_pricing: Option<Vec<TieredRate>>,
}

/// Rates that replace the same-named base fields inside the stated UTC windows.
#[derive(JsonSchema)]
#[allow(dead_code)]
struct OffPeakPricing {
    hours_utc: Option<UtcHours>,
    windows: Option<Vec<OffPeakWindow>>,
    weekday_timezone: Option<String>,
}

/// One tier of a context-length or result-count tiered rate.
#[derive(JsonSchema)]
#[allow(dead_code)]
struct TieredRate {
    range: Option<[u64; 2]>,
    max_results_range: Option<[u64; 2]>,
}

#[derive(JsonSchema)]
#[allow(dead_code)]
struct RegistryDocument {
    sample_spec: Option<Map<String, Value>>,
    fallback_generalizations: Option<FallbackGeneralizations>,
}

pub(crate) const FLAT_KEY_PATTERN: &str = concat!(
    r"^(",
    r"input_cost_per_((audio_|image_|video_)?token|character|image|pixel|query|request|second|text_message|audio_per_second|video_per_second(_above_\d+s_interval)?)",
    r"|output_cost_per_((audio_|image_|video_|reasoning_)?token|character|image(_\d+(\.\d+)?K?)?|pixel|second(_\d+[pk])?|video_per_second)",
    r"|cache_read_input_(audio_|image_|video_)?token_cost",
    r"|explicit_cache_read_input_token_cost",
    r"|cache_creation_input_(audio_|image_|video_)?token_cost(_above_1hr|_above_12hr)?",
    r"|annotation_cost_per_page|citation_cost_per_token|code_interpreter_cost_per_session",
    r"|google_maps_grounding_cost_per_query|ocr_cost_per_credit|ocr_cost_per_page|cost_per_second",
    r"|browser_open_cost_per_request|cost_per_additional_result",
    r")(_above_\d+k?_tokens)?(_(priority|flex|ultrafast|balanced|batches|decisions))?$",
);

/// JSON Schema for one model entry, including registry validation constraints.
pub fn model_entry_json_schema() -> Schema {
    let mut schema = serde_json::to_value(schemars::schema_for!(ModelEntry))
        .expect("derived model schema serializes");
    remove_nullable_optional_fields(&mut schema);
    decorate_model_entry(&mut schema);
    Schema::from(
        schema
            .as_object()
            .expect("derived schema is an object")
            .clone(),
    )
}

/// JSON Schema for the complete model prices registry document.
pub fn registry_json_schema() -> Schema {
    let mut entry = model_entry_json_schema().as_value().clone();
    let mut definitions = take_definitions(&mut entry);
    entry.as_object_mut().unwrap().remove("$schema");
    definitions.insert("modelEntry".into(), entry);

    let mut registry = serde_json::to_value(schemars::schema_for!(RegistryDocument))
        .expect("derived registry schema serializes");
    remove_nullable_optional_fields(&mut registry);
    definitions.extend(take_definitions(&mut registry));
    let root = registry.as_object_mut().unwrap();
    root.insert(
        "title".into(),
        "LiteLLM model prices and context window registry".into(),
    );
    root.insert(
        "additionalProperties".into(),
        json!({"$ref": "#/$defs/modelEntry"}),
    );
    root.insert("$defs".into(), Value::Object(definitions));
    Schema::from(root.clone())
}

fn take_definitions(schema: &mut Value) -> Map<String, Value> {
    schema
        .as_object_mut()
        .unwrap()
        .remove("$defs")
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default()
}

fn remove_nullable_optional_fields(value: &mut Value) {
    match value {
        Value::Array(values) => values.iter_mut().for_each(remove_nullable_optional_fields),
        Value::Object(map) => {
            map.values_mut().for_each(remove_nullable_optional_fields);
            if let Some(Value::Array(types)) = map.get_mut("type") {
                types.retain(|value| value != "null");
                if types.len() == 1 {
                    let only = types[0].clone();
                    map.insert("type".into(), only);
                }
            }
            if let Some(Value::Array(branches)) = map.get_mut("anyOf") {
                branches.retain(|branch| branch.get("type") != Some(&Value::String("null".into())));
                if branches.len() == 1 {
                    let only = branches[0]
                        .as_object()
                        .expect("schema branch is an object")
                        .clone();
                    map.remove("anyOf");
                    map.extend(only);
                }
            }
        }
        _ => {}
    }
}

fn decorate_model_entry(schema: &mut Value) {
    let rates = json!({FLAT_KEY_PATTERN: {"type": "number", "minimum": 0}});
    let object = schema.as_object_mut().unwrap();
    object.insert("required".into(), json!(["litellm_provider"]));
    object.insert("additionalProperties".into(), Value::Bool(true));
    object.insert(
        "patternProperties".into(),
        json!({
            FLAT_KEY_PATTERN: {"type": "number", "minimum": 0},
            "^supports_": {"type": "boolean"},
        }),
    );
    let definitions = object.get_mut("$defs").unwrap().as_object_mut().unwrap();
    for definition in ["OffPeakPricing", "TieredRate"] {
        definitions[definition]["patternProperties"] = rates.clone();
        definitions[definition]["additionalProperties"] = Value::Bool(false);
    }
    definitions["OffPeakPricing"]["anyOf"] = json!([
        {"required": ["hours_utc"]},
        {"required": ["windows"]}
    ]);
    definitions["OffPeakPricing"]["properties"]["windows"]["minItems"] = json!(1);
    definitions["OffPeakWindow"]["properties"]["weekdays"]["minItems"] = json!(1);
    definitions["UtcHours"]["anyOf"][1]["minItems"] = json!(1);
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::FLAT_KEY_PATTERN;
    use crate::wire::FlatKey;

    #[rstest]
    #[case::input_token("input_cost_per_token")]
    #[case::cache_write_one_hour_band(
        "cache_creation_input_token_cost_above_1hr_above_200k_tokens"
    )]
    #[case::image_label("output_cost_per_image_0.5K")]
    #[case::resolution_tier("output_cost_per_second_4k_flex")]
    #[case::feature_batch("annotation_cost_per_page_batches")]
    #[case::video_interval("input_cost_per_video_per_second_above_8s_interval")]
    #[case::explicit_cache_read("explicit_cache_read_input_token_cost")]
    #[case::wall_clock("cost_per_second")]
    #[case::unused_meter("cost_per_token")]
    #[case::other_currency("input_dbu_cost_per_token")]
    #[case::typo("input_cost_per_tokne")]
    #[case::read_has_no_ttl("cache_read_input_token_cost_above_1hr")]
    #[case::interval_on_a_token_meter("input_cost_per_token_above_8s_interval")]
    fn flat_key_pattern_agrees_with_the_parser(#[case] key: &str) {
        let schema = serde_json::json!({
            "type": "object",
            "patternProperties": {FLAT_KEY_PATTERN: {}},
            "additionalProperties": false,
        });
        let validator = jsonschema::validator_for(&schema).unwrap();

        assert_eq!(
            validator.is_valid(&serde_json::json!({key: 1})),
            FlatKey::parse(key).is_some()
        );
    }
}
