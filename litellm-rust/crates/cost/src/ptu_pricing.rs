use jiff::Timestamp;
use jiff::civil::{Date, DateTime};
use serde_json::{Map, Value, json};

use crate::wire::{is_truthy, py_float, py_int, py_str};

pub const PTU_COST_ATTRIBUTION_ENV_VAR: &str = "LITELLM_ENABLE_PTU_COST_ATTRIBUTION";
pub const AZURE_SPILLOVER_HEADER: &str = "x-ms-is-spilled-over";
pub const AZURE_SPILLOVER_FROM_HEADER: &str = "x-ms-spillover-from-deployment";

pub const MAX_PTU_COUNT: i64 = 1_000_000;
pub const MAX_COST_PER_PTU_PER_HOUR: f64 = 1_000_000.0;

pub const PTU_MODEL_INFO_FIELDS: &[&str] = &[
    "ptu_count",
    "cost_per_ptu_per_hour",
    "ptu_effective_from",
    "ptu_effective_to",
];

pub const PTU_ZEROED_PRICING_FIELDS: &[&str] = &[
    "input_cost_per_token",
    "output_cost_per_token",
    "input_cost_per_character",
    "output_cost_per_character",
    "cache_read_input_token_cost",
    "cache_creation_input_token_cost",
    "cache_creation_input_token_cost_above_1hr",
    "cache_creation_input_token_cost_above_200k_tokens",
    "cache_read_input_token_cost_above_200k_tokens",
    "google_maps_grounding_cost_per_query",
];

pub const SEARCH_CONTEXT_SIZES: &[&str] = &[
    "search_context_size_low",
    "search_context_size_medium",
    "search_context_size_high",
];

// The CustomPricingLiteLLMParams fields whose name contains "cost", from
// litellm/types/utils.py as of 2026-09-22.
const CUSTOM_PRICING_FIELDS: &[&str] = &[
    "annotation_cost_per_page",
    "annotation_cost_per_page_batches",
    "cache_creation_input_audio_token_cost",
    "cache_creation_input_token_cost",
    "cache_creation_input_token_cost_above_1hr",
    "cache_creation_input_token_cost_above_200k_tokens",
    "cache_creation_input_token_cost_above_272k_tokens",
    "cache_creation_input_token_cost_above_272k_tokens_batches",
    "cache_creation_input_token_cost_above_272k_tokens_flex",
    "cache_creation_input_token_cost_above_272k_tokens_priority",
    "cache_creation_input_token_cost_batches",
    "cache_creation_input_token_cost_flex",
    "cache_creation_input_token_cost_priority",
    "cache_creation_input_token_cost_ultrafast",
    "cache_read_input_audio_token_cost",
    "cache_read_input_image_token_cost",
    "cache_read_input_token_cost",
    "cache_read_input_token_cost_above_200k_tokens",
    "cache_read_input_token_cost_above_200k_tokens_priority",
    "cache_read_input_token_cost_above_512k_tokens",
    "cache_read_input_token_cost_above_272k_tokens",
    "cache_read_input_token_cost_above_272k_tokens_batches",
    "cache_read_input_token_cost_above_272k_tokens_flex",
    "cache_read_input_token_cost_above_272k_tokens_priority",
    "cache_read_input_token_cost_batches",
    "cache_read_input_token_cost_flex",
    "cache_read_input_token_cost_priority",
    "cache_read_input_token_cost_ultrafast",
    "citation_cost_per_token",
    "google_maps_grounding_cost_per_query",
    "input_cost_per_audio_per_second",
    "input_cost_per_audio_per_second_above_128k_tokens",
    "input_cost_per_audio_token",
    "input_cost_per_audio_token_batches",
    "input_cost_per_character",
    "input_cost_per_character_above_128k_tokens",
    "input_cost_per_image",
    "input_cost_per_image_above_128k_tokens",
    "input_cost_per_image_token",
    "input_cost_per_image_token_batches",
    "input_cost_per_pixel",
    "input_cost_per_query",
    "input_cost_per_second",
    "input_cost_per_token",
    "input_cost_per_token_above_128k_tokens",
    "input_cost_per_token_above_200k_tokens",
    "input_cost_per_token_above_200k_tokens_priority",
    "input_cost_per_token_above_272k_tokens",
    "input_cost_per_token_above_272k_tokens_batches",
    "input_cost_per_token_above_272k_tokens_flex",
    "input_cost_per_token_above_272k_tokens_priority",
    "input_cost_per_token_above_512k_tokens",
    "input_cost_per_token_batches",
    "input_cost_per_token_cache_hit",
    "input_cost_per_token_flex",
    "input_cost_per_token_priority",
    "input_cost_per_token_ultrafast",
    "input_cost_per_video_per_second",
    "input_cost_per_video_per_second_above_128k_tokens",
    "input_cost_per_video_per_second_above_15s_interval",
    "input_cost_per_video_per_second_above_8s_interval",
    "input_cost_per_video_token",
    "input_cost_per_video_token_batches",
    "ocr_cost_per_credit",
    "ocr_cost_per_page",
    "ocr_cost_per_page_batches",
    "output_cost_per_audio_per_second",
    "output_cost_per_audio_token",
    "output_cost_per_character",
    "output_cost_per_character_above_128k_tokens",
    "output_cost_per_image",
    "output_cost_per_image_1024",
    "output_cost_per_image_1536",
    "output_cost_per_image_512",
    "output_cost_per_image_token",
    "output_cost_per_pixel",
    "output_cost_per_reasoning_token",
    "output_cost_per_reasoning_token_flex",
    "output_cost_per_reasoning_token_priority",
    "output_cost_per_second",
    "output_cost_per_second_1080p",
    "output_cost_per_second_2k",
    "output_cost_per_second_480p",
    "output_cost_per_second_4k",
    "output_cost_per_second_720p",
    "output_cost_per_second_768p",
    "output_cost_per_token",
    "output_cost_per_token_above_128k_tokens",
    "output_cost_per_token_above_200k_tokens",
    "output_cost_per_token_above_200k_tokens_priority",
    "output_cost_per_token_above_272k_tokens",
    "output_cost_per_token_above_272k_tokens_batches",
    "output_cost_per_token_above_272k_tokens_flex",
    "output_cost_per_token_above_272k_tokens_priority",
    "output_cost_per_token_above_512k_tokens",
    "output_cost_per_token_batches",
    "output_cost_per_token_flex",
    "output_cost_per_token_priority",
    "output_cost_per_token_ultrafast",
    "output_cost_per_video_per_second",
    "output_cost_per_video_token",
    "search_context_cost_per_query",
];

#[derive(Clone, Debug, PartialEq)]
pub struct PtuTerms {
    pub team_id: String,
    pub ptu_count: i64,
    pub cost_per_ptu_per_hour: f64,
    pub effective_from: Timestamp,
    pub effective_to: Option<Timestamp>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AzureSpillover {
    pub from_deployment: Option<String>,
}

fn as_utc(value: &Value) -> Option<Timestamp> {
    let Value::String(text) = value else {
        return None;
    };
    if let Ok(stamped) = text.parse::<Timestamp>() {
        return Some(stamped);
    }
    let normalized = text.replace(' ', "T");
    if let Ok(naive) = normalized.parse::<DateTime>() {
        return naive
            .to_zoned(jiff::tz::TimeZone::UTC)
            .ok()
            .map(|zoned| zoned.timestamp());
    }
    normalized
        .parse::<Date>()
        .ok()
        .and_then(|date| {
            date.to_datetime(jiff::civil::time(0, 0, 0, 0))
                .to_zoned(jiff::tz::TimeZone::UTC)
                .ok()
        })
        .map(|zoned| zoned.timestamp())
}

fn named(reason: &str, model_name: Option<&str>) -> String {
    match model_name {
        None => reason.to_string(),
        Some(model_name) => {
            format!("PTU configuration on model '{model_name}' is invalid: {reason}")
        }
    }
}

pub fn declares_ptu(model_info: &Value) -> bool {
    PTU_MODEL_INFO_FIELDS
        .iter()
        .any(|field| model_info.get(field).is_some_and(|value| !value.is_null()))
}

pub fn ptu_config_error(model_info: &Value, model_name: Option<&str>) -> Option<String> {
    let effective_from = model_info.get("ptu_effective_from").and_then(as_utc);
    let effective_to = model_info.get("ptu_effective_to").and_then(as_utc);
    if let (Some(from), Some(to)) = (effective_from, effective_to)
        && to <= from
    {
        return Some(named(
            "ptu_effective_to must be after ptu_effective_from",
            model_name,
        ));
    }
    let has_count = model_info
        .get("ptu_count")
        .is_some_and(|value| !value.is_null());
    let has_rate = model_info
        .get("cost_per_ptu_per_hour")
        .is_some_and(|value| !value.is_null());
    if !has_count && !has_rate {
        return None;
    }
    if has_count != has_rate {
        return Some(named(
            "ptu_count and cost_per_ptu_per_hour must be set together",
            model_name,
        ));
    }
    if effective_from.is_none() {
        return Some(named(
            "ptu_effective_from is required when PTU fields are set. Flat cost accrues from that \
             instant, so without it the start would have to be inferred and a deployment \
             configured today could be billed for days it did not exist",
            model_name,
        ));
    }
    if !is_truthy(model_info.get("team_id").unwrap_or(&Value::Null)) {
        return Some(named(
            "team_id is required when PTU fields are set (one model maps to one team)",
            model_name,
        ));
    }
    None
}

pub fn ptu_identity_error(
    declared_id: Option<&str>,
    taken: bool,
    current_id: Option<&str>,
    model_name: Option<&str>,
) -> Option<String> {
    if declared_id.is_none_or(str::is_empty) {
        let current = current_id
            .filter(|current| !current.is_empty())
            .unwrap_or("shown by GET /model/info");
        return Some(named(
            &format!(
                "model_info.id is required when PTU fields are set. Without one the deployment is \
                 identified by a hash of its litellm_params, so rotating a credential bills the \
                 reservation a second time under the new identity. Set it to the id this \
                 deployment already uses, {current}, so the flat cost already written stays under \
                 one identity; any other value starts a second one"
            ),
            model_name,
        ));
    }
    if taken && let Some(declared_id) = declared_id {
        return Some(named(
            &format!(
                "model_info.id '{declared_id}' is declared on more than one deployment. Each would \
                 key the same flat-cost row, so one reservation would go unbilled"
            ),
            model_name,
        ));
    }
    None
}

pub fn ptu_terms(model_info: &Value) -> Option<PtuTerms> {
    let team_id = model_info.get("team_id").filter(|value| is_truthy(value))?;
    let ptu_count = py_int(model_info.get("ptu_count").unwrap_or(&Value::Null))?;
    let cost_per_hour = py_float(
        model_info
            .get("cost_per_ptu_per_hour")
            .unwrap_or(&Value::Null),
    )?;
    if ptu_count <= 0 || ptu_count > MAX_PTU_COUNT {
        return None;
    }
    if !(0.0..=MAX_COST_PER_PTU_PER_HOUR).contains(&cost_per_hour) {
        return None;
    }
    let raw_from = model_info
        .get("ptu_effective_from")
        .filter(|value| !value.is_null());
    let raw_to = model_info
        .get("ptu_effective_to")
        .filter(|value| !value.is_null());
    let effective_from = raw_from.and_then(as_utc)?;
    let effective_to = raw_to.and_then(as_utc);
    if raw_to.is_some() && effective_to.is_none() {
        return None;
    }
    if let Some(to) = effective_to
        && to <= effective_from
    {
        return None;
    }
    Some(PtuTerms {
        team_id: py_str(team_id),
        ptu_count,
        cost_per_ptu_per_hour: cost_per_hour,
        effective_from,
        effective_to,
    })
}

pub fn zeroed_ptu_pricing(
    model_info: &Value,
    declared: &Value,
    attribution_enabled: bool,
) -> Option<Map<String, Value>> {
    ptu_terms(model_info)?;
    if !attribution_enabled {
        return None;
    }
    let mut pricing = Map::new();
    for field in PTU_ZEROED_PRICING_FIELDS {
        pricing.insert((*field).to_string(), json!(0.0));
    }
    pricing.insert("tiered_pricing".to_string(), json!([]));
    let table = SEARCH_CONTEXT_SIZES
        .iter()
        .map(|size| ((*size).to_string(), json!(0.0)))
        .collect::<Map<String, Value>>();
    pricing.insert(
        "search_context_cost_per_query".to_string(),
        Value::Object(table),
    );
    if let Some(declared) = declared.as_object() {
        for key in declared.keys() {
            if CUSTOM_PRICING_FIELDS.contains(&key.as_str())
                && key != "search_context_cost_per_query"
                && key != "tiered_pricing"
            {
                pricing.insert(key.clone(), json!(0.0));
            }
        }
    }
    Some(pricing)
}

pub fn is_spilled_over_ptu_request(
    model_info: &Value,
    response_headers: Option<&Value>,
    additional_headers: Option<&Value>,
    attribution_enabled: bool,
) -> bool {
    ptu_terms(model_info).is_some()
        && attribution_enabled
        && azure_spillover(response_headers, additional_headers).is_some()
}

pub fn azure_spillover(
    response_headers: Option<&Value>,
    additional_headers: Option<&Value>,
) -> Option<AzureSpillover> {
    for (headers, prefix) in [
        (response_headers, ""),
        (additional_headers, "llm_provider-"),
    ] {
        let Some(headers) = headers.and_then(Value::as_object) else {
            continue;
        };
        let marker = headers
            .get(&format!("{prefix}{AZURE_SPILLOVER_HEADER}"))
            .map(py_str)
            .unwrap_or_else(|| "None".to_string());
        if marker.to_lowercase() != "true" {
            continue;
        }
        let from_deployment = headers
            .get(&format!("{prefix}{AZURE_SPILLOVER_FROM_HEADER}"))
            .filter(|value| !value.is_null())
            .map(py_str);
        return Some(AzureSpillover { from_deployment });
    }
    None
}
