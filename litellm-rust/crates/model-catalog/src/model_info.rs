use indexmap::IndexMap;
use serde::Serialize;
use serde_json::Value;

use crate::wire::Fields;

macro_rules! model_info_fields {
    ($($name:ident),* $(,)?) => {
        /// The fields main's `ModelInfoBase(...)` copies in `get_model_info`, one field each, read from the raw row
        /// without lowering it, so building one never fails and a bad value comes back as written
        #[derive(Clone, Debug, Default, PartialEq, Serialize)]
        pub struct ModelInfoFields {
            $(pub $name: Value,)*
            /// Every key matching `ABOVE_THRESHOLD_COST_KEY_PATTERN` the fields above do not name, copied as written
            #[serde(flatten)]
            pub above_threshold_costs: IndexMap<String, Value>,
        }

        impl ModelInfoFields {
            /// The copied field names in main's order
            pub const FIELDS: &[&str] = &[$(stringify!($name)),*];

            fn copied(key: &str, fields: &Fields, provider: Option<&str>) -> Self {
                Self {
                    $($name: copied_field(stringify!($name), key, fields, provider),)*
                    above_threshold_costs: IndexMap::new(),
                }
            }
        }
    };
}

model_info_fields! {
    key,
    max_tokens,
    max_input_tokens,
    max_output_tokens,
    input_cost_per_token,
    input_cost_per_token_flex,
    input_cost_per_token_priority,
    input_cost_per_token_balanced,
    input_cost_per_token_ultrafast,
    cache_creation_input_token_cost,
    cache_creation_input_token_cost_above_200k_tokens,
    cache_creation_input_token_cost_above_272k_tokens,
    cache_creation_input_token_cost_above_272k_tokens_priority,
    cache_creation_input_token_cost_above_272k_tokens_flex,
    cache_creation_input_token_cost_above_272k_tokens_ultrafast,
    cache_creation_input_token_cost_flex,
    cache_creation_input_token_cost_priority,
    cache_creation_input_token_cost_ultrafast,
    cache_read_input_token_cost,
    cache_read_input_audio_token_cost,
    cache_read_input_audio_token_cost_flex,
    cache_read_input_audio_token_cost_priority,
    cache_read_input_audio_token_cost_batches,
    explicit_cache_read_input_token_cost,
    prompt_cache_min_tokens,
    cache_read_input_token_cost_above_200k_tokens,
    cache_read_input_token_cost_above_200k_tokens_priority,
    cache_read_input_token_cost_above_272k_tokens,
    cache_read_input_token_cost_above_272k_tokens_priority,
    cache_read_input_token_cost_above_272k_tokens_flex,
    cache_read_input_token_cost_above_272k_tokens_ultrafast,
    cache_read_input_token_cost_above_512k_tokens,
    cache_read_input_token_cost_flex,
    cache_read_input_token_cost_priority,
    cache_read_input_token_cost_balanced,
    cache_read_input_token_cost_ultrafast,
    cache_read_input_token_cost_batches,
    cache_read_input_token_cost_above_200k_tokens_batches,
    cache_read_input_token_cost_above_272k_tokens_batches,
    cache_creation_input_token_cost_batches,
    cache_creation_input_token_cost_above_200k_tokens_batches,
    cache_creation_input_token_cost_above_272k_tokens_batches,
    cache_creation_input_token_cost_above_1hr,
    cache_creation_input_token_cost_above_12hr,
    cache_creation_input_token_cost_above_12hr_flex,
    cache_creation_input_token_cost_above_12hr_priority,
    cache_creation_input_token_cost_above_12hr_above_272k_tokens,
    cache_creation_input_token_cost_above_12hr_above_272k_tokens_flex,
    cache_creation_input_token_cost_above_12hr_above_272k_tokens_priority,
    cache_creation_input_token_cost_above_12hr_batches,
    cache_creation_input_token_cost_above_12hr_ultrafast,
    cache_creation_input_token_cost_above_12hr_above_272k_tokens_batches,
    cache_creation_input_token_cost_above_12hr_above_272k_tokens_ultrafast,
    off_peak_pricing,
    input_cost_per_character,
    input_cost_per_token_above_128k_tokens,
    input_cost_per_token_above_200k_tokens,
    input_cost_per_token_above_200k_tokens_priority,
    input_cost_per_token_above_272k_tokens,
    input_cost_per_token_above_272k_tokens_priority,
    input_cost_per_token_above_272k_tokens_flex,
    input_cost_per_token_above_272k_tokens_ultrafast,
    input_cost_per_token_above_512k_tokens,
    input_cost_per_query,
    cost_per_second,
    input_cost_per_second,
    input_cost_per_text_message,
    input_cost_per_audio_token,
    input_cost_per_audio_token_flex,
    input_cost_per_audio_token_priority,
    input_cost_per_image_token,
    input_cost_per_video_token,
    input_cost_per_audio_token_batches,
    input_cost_per_image_token_batches,
    input_cost_per_image,
    input_cost_per_audio_per_second,
    input_cost_per_video_per_second,
    input_cost_per_token_batches,
    input_cost_per_video_token_batches,
    input_cost_per_token_above_200k_tokens_batches,
    input_cost_per_token_above_272k_tokens_batches,
    output_cost_per_token_batches,
    output_cost_per_token_above_200k_tokens_batches,
    output_cost_per_token_above_272k_tokens_batches,
    output_cost_per_token,
    output_cost_per_token_flex,
    output_cost_per_token_priority,
    output_cost_per_token_balanced,
    output_cost_per_token_ultrafast,
    regional_processing_uplift_multiplier_eu,
    regional_processing_uplift_multiplier_us,
    regional_endpoint_uplift_multiplier,
    output_cost_per_audio_token,
    output_cost_per_character,
    output_cost_per_reasoning_token,
    output_cost_per_reasoning_token_flex,
    output_cost_per_reasoning_token_priority,
    output_cost_per_token_above_128k_tokens,
    output_cost_per_character_above_128k_tokens,
    output_cost_per_token_above_200k_tokens,
    output_cost_per_token_above_200k_tokens_priority,
    output_cost_per_token_above_272k_tokens,
    output_cost_per_token_above_272k_tokens_priority,
    output_cost_per_token_above_272k_tokens_flex,
    output_cost_per_token_above_272k_tokens_ultrafast,
    output_cost_per_token_above_512k_tokens,
    output_cost_per_second,
    output_cost_per_second_1080p,
    output_cost_per_second_480p,
    output_cost_per_second_720p,
    output_cost_per_second_768p,
    output_cost_per_second_2k,
    output_cost_per_second_4k,
    output_cost_per_video_per_second,
    output_cost_per_image,
    output_cost_per_pixel,
    output_cost_per_image_token,
    output_cost_per_image_token_batches,
    output_cost_per_video_token,
    output_vector_size,
    citation_cost_per_token,
    tiered_pricing,
    litellm_provider,
    mode,
    supported_endpoints,
    supports_system_messages,
    supports_response_schema,
    supports_vision,
    supports_function_calling,
    supports_parallel_function_calling,
    supports_tool_choice,
    supports_assistant_prefill,
    supports_prompt_caching,
    supports_prompt_cache_breakpoint,
    supports_thinking_cache_preservation,
    supports_audio_input,
    supports_video_input,
    supports_box_questions,
    supports_audio_output,
    supports_pdf_input,
    supports_embedding_image_input,
    supports_native_streaming,
    supports_native_structured_output,
    supports_bedrock_native_structured_output,
    supports_web_search,
    supports_url_context,
    supports_reasoning,
    supports_adaptive_thinking,
    supports_legacy_thinking,
    thinking_always_on,
    supports_tool_search,
    supports_mid_conversation_system,
    supports_anthropic_thinking_payload,
    supports_anthropic_compaction,
    supports_none_reasoning_effort,
    supports_minimal_reasoning_effort,
    supports_low_reasoning_effort,
    supports_xhigh_reasoning_effort,
    supports_max_reasoning_effort,
    reasoning_effort_levels,
    default_reasoning_effort,
    bedrock_output_config_effort_ceiling,
    bedrock_converse_supports_strict_tools,
    supports_regex_lookaround,
    supports_computer_use,
    search_context_cost_per_query,
    browser_open_cost_per_request,
    cost_per_additional_result,
    web_search_billing_unit,
    search_free_requests,
    google_maps_grounding_cost_per_query,
    tpm,
    rpm,
    ocr_cost_per_page,
    ocr_cost_per_page_batches,
    ocr_cost_per_credit,
    annotation_cost_per_page,
    annotation_cost_per_page_batches,
    provider_specific_entry,
    uses_embed_content,
    supports_image_size,
    supported_audio_formats,
    token_count_framing,
    token_count_model,
    token_count_tokenizer,
    vertex_ai_audio_api,
}

/// main's `get_model_info_helper` copy of one matched row. `provider` is the requested provider, which a row with no
/// `litellm_provider` field reports as its own
pub fn model_info(key: &str, fields: &Fields, provider: Option<&str>) -> ModelInfoFields {
    let copied = ModelInfoFields::copied(key, fields, provider);
    ModelInfoFields {
        above_threshold_costs: fields
            .iter()
            .filter(|(name, _)| {
                !ModelInfoFields::FIELDS.contains(&name.as_str())
                    && is_above_threshold_cost_key(name)
            })
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect(),
        ..copied
    }
}

/// One field as main's copy writes it. `max_tokens` is derived, a missing token price defaults to `0`
pub fn copied_field(name: &str, key: &str, fields: &Fields, provider: Option<&str>) -> Value {
    let field = |name: &str| fields.get(name).cloned().unwrap_or(Value::Null);
    match name {
        "key" => Value::String(key.to_owned()),
        "max_tokens" => {
            let output = field("max_output_tokens");
            if truthy(&output) {
                output
            } else {
                field("max_input_tokens")
            }
        }
        "input_cost_per_token" | "output_cost_per_token" => match field(name) {
            Value::Null => Value::from(0),
            value => value,
        },
        "litellm_provider" => fields.get(name).cloned().unwrap_or_else(|| {
            provider.map_or(Value::Null, |provider| Value::String(provider.to_owned()))
        }),
        name => field(name),
    }
}

/// `ABOVE_THRESHOLD_COST_KEY_PATTERN`, `_above_\d+k?_tokens(?:_batches)?$`
pub fn is_above_threshold_cost_key(name: &str) -> bool {
    let name = name.strip_suffix("_batches").unwrap_or(name);
    let Some(rest) = name.strip_suffix("_tokens") else {
        return false;
    };
    let rest = rest.strip_suffix('k').unwrap_or(rest);
    let Some((_, digits)) = rest.rsplit_once("_above_") else {
        return false;
    };
    !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
}

/// Python truthiness of a JSON value, for main's `a or b`
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|number| number != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn fields(row: Value) -> Fields {
        let Value::Object(fields) = row else {
            unreachable!()
        };
        fields
    }

    #[rstest]
    #[case::output_first(json!({"max_output_tokens": 10, "max_input_tokens": 20}), json!(10))]
    #[case::zero_output_falls_to_input(json!({"max_output_tokens": 0, "max_input_tokens": 20}), json!(20))]
    #[case::neither(json!({}), Value::Null)]
    fn max_tokens_is_output_or_input(#[case] row: Value, #[case] expected: Value) {
        assert_eq!(model_info("m", &fields(row), None).max_tokens, expected);
    }

    #[rstest]
    fn a_missing_token_price_is_zero_and_a_bad_one_comes_back() {
        let info = model_info("m", &fields(json!({"input_cost_per_token": "abc"})), None);
        assert_eq!(info.input_cost_per_token, json!("abc"));
        assert_eq!(info.output_cost_per_token, json!(0));
        assert_eq!(info.key, json!("m"));
    }

    #[rstest]
    fn above_threshold_keys_the_copy_does_not_name_are_kept() {
        let info = model_info(
            "m",
            &fields(json!({"input_cost_per_token_above_999k_tokens": 1.0, "unrelated_field": 2.0})),
            None,
        );
        assert_eq!(
            info.above_threshold_costs
                .get("input_cost_per_token_above_999k_tokens"),
            Some(&json!(1.0))
        );
        assert!(!info.above_threshold_costs.contains_key("unrelated_field"));
    }

    #[rstest]
    #[case::absent_takes_the_request(json!({}), json!("azure"))]
    #[case::null_stays_null(json!({"litellm_provider": null}), Value::Null)]
    #[case::written_wins(json!({"litellm_provider": "openai"}), json!("openai"))]
    fn a_row_without_a_provider_reports_the_requested_one(
        #[case] row: Value,
        #[case] expected: Value,
    ) {
        assert_eq!(
            model_info("m", &fields(row), Some("azure")).litellm_provider,
            expected
        );
    }

    #[rstest]
    #[case::k("input_cost_per_token_above_200k_tokens", true)]
    #[case::batches("input_cost_per_token_above_200k_tokens_batches", true)]
    #[case::plain_count("input_cost_per_token_above_128_tokens", true)]
    #[case::priority_suffix("input_cost_per_token_above_200k_tokens_priority", false)]
    #[case::one_hour("cache_creation_input_token_cost_above_1hr", false)]
    fn the_threshold_key_pattern_matches_mains(#[case] name: &str, #[case] expected: bool) {
        assert_eq!(is_above_threshold_cost_key(name), expected);
    }
}
