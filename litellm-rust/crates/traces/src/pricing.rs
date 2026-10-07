use std::collections::BTreeMap;

use litellm_cost::{
    PromptConvention, Usage,
    catalog::{self, EstimateRequest},
};
use litellm_model_catalog::PricingCatalog;

use crate::normalize::openinference;

fn count(value: &str) -> Option<u64> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    value.parse().ok()
}

fn counts(attributes: &BTreeMap<String, String>, keys: &[&str]) -> Option<Option<u64>> {
    let values = keys
        .iter()
        .filter_map(|key| attributes.get(*key))
        .map(|value| count(value))
        .collect::<Option<Vec<_>>>()?;
    let first = values.first().copied();
    values
        .iter()
        .all(|value| Some(*value) == first)
        .then_some(first)
}

fn text<'a>(attributes: &'a BTreeMap<String, String>, keys: &[&str]) -> Option<Option<&'a str>> {
    let values: Vec<_> = keys
        .iter()
        .filter_map(|key| attributes.get(*key))
        .map(String::as_str)
        .collect();
    let first = values.first().copied();
    values
        .iter()
        .all(|value| Some(*value) == first)
        .then_some(first)
}

fn requires_additional_pricing(fields: &serde_json::Map<String, serde_json::Value>) -> bool {
    !matches!(
        fields.get("supports_token_only_pricing"),
        None | Some(serde_json::Value::Null | serde_json::Value::Bool(true))
    ) || ["input_cost_per_request", "input_cost_per_query"]
        .into_iter()
        .any(|key| {
            fields.get(key).is_some_and(|value| {
                !value.is_null()
                    && value
                        .as_f64()
                        .or_else(|| value.as_str()?.parse::<f64>().ok())
                        != Some(0.0)
            })
        })
}

pub fn estimate_cost(
    attributes: &BTreeMap<String, String>,
    catalog: &PricingCatalog,
) -> Option<f64> {
    if attributes
        .get("gen_ai.output.type")
        .is_some_and(|value| !matches!(value.as_str(), "text" | "json"))
    {
        return None;
    }
    if attributes.iter().any(|(key, value)| {
        (key.starts_with("gen_ai.usage.")
            || key.starts_with("llm.token_count.")
            || key.starts_with("anthropic.usage."))
            && [
                "audio",
                "image",
                "video",
                "search",
                "grounding",
                "browser",
                "code_interpreter",
                "code_execution",
                "computer_use",
                "citation",
            ]
            .iter()
            .any(|kind| key.contains(kind))
            && value.parse::<f64>().ok() != Some(0.0)
    }) {
        return None;
    }
    let model = attributes
        .get("gen_ai.response.model")
        .or_else(|| attributes.get(openinference::MODEL))
        .or_else(|| attributes.get("gen_ai.request.model"))?;
    let input = counts(
        attributes,
        &[
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.prompt_tokens",
            openinference::INPUT_TOKENS,
        ],
    )??;
    let output = counts(
        attributes,
        &[
            "gen_ai.usage.output_tokens",
            "gen_ai.usage.completion_tokens",
            openinference::OUTPUT_TOKENS,
        ],
    )??;
    let read = counts(
        attributes,
        &[
            "gen_ai.usage.cache_read.input_tokens",
            "llm.token_count.prompt_details.cache_read",
        ],
    )?
    .unwrap_or_default();
    let write = counts(
        attributes,
        &[
            "gen_ai.usage.cache_write.input_tokens",
            "gen_ai.usage.cache_creation.input_tokens",
            "llm.token_count.prompt_details.cache_write",
        ],
    )?
    .unwrap_or_default();
    let five = counts(
        attributes,
        &["anthropic.usage.cache_creation.ephemeral_5m_input_tokens"],
    )?;
    let one = counts(
        attributes,
        &["anthropic.usage.cache_creation.ephemeral_1h_input_tokens"],
    )?;
    let reasoning = counts(
        attributes,
        &[
            "gen_ai.usage.reasoning.output_tokens",
            "gen_ai.usage.details.reasoning_tokens",
            "gen_ai.usage.reasoning_tokens",
            "llm.token_count.completion_details.reasoning",
        ],
    )?;
    let response_tier = text(
        attributes,
        &[
            "openai.response.service_tier",
            "anthropic.response.service_tier",
            "gen_ai.openai.response.service_tier",
        ],
    )?;
    let request_tier = text(
        attributes,
        &[
            "openai.request.service_tier",
            "gen_ai.openai.request.service_tier",
        ],
    )?;
    let start = match attributes.get("litellm.trace.start_ns") {
        Some(value) => {
            count(value.strip_prefix('-').unwrap_or(value))?;
            Some(value.parse::<i64>().ok()?)
        }
        None => None,
    };
    let hosting: Vec<_> = ["gen_ai.provider.name", "gen_ai.system", "llm.provider"]
        .into_iter()
        .filter_map(|key| attributes.get(key).map(String::as_str))
        .collect();
    let providers = if hosting.is_empty() {
        attributes
            .get("llm.system")
            .map(String::as_str)
            .into_iter()
            .collect()
    } else {
        hosting
    };
    let matched = catalog.resolve(model, &providers)?;
    for key in [
        "gen_ai.response.model",
        "gen_ai.request.model",
        openinference::MODEL,
    ] {
        if let Some(observed) = attributes.get(key)
            && catalog
                .request_contexts(observed, &providers)?
                .iter()
                .any(|context| requires_additional_pricing(context.fields))
        {
            return None;
        }
    }
    if matched
        .fields
        .get("mode")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|mode| {
            matches!(
                mode,
                "audio_speech"
                    | "audio_transcription"
                    | "image_generation"
                    | "image_edit"
                    | "video_generation"
                    | "realtime"
                    | "ocr"
            )
        })
    {
        return None;
    }
    catalog::estimate(
        matched.fields,
        &EstimateRequest {
            usage: Usage {
                prompt_tokens: input,
                completion_tokens: output,
                cache_read_tokens: read,
                cache_write_tokens: write,
                cache_write_5m_tokens: five,
                cache_write_1h_tokens: one,
                prompt_convention: PromptConvention::IncludesCache,
            },
            service_tier: response_tier.or(request_tier),
            threshold_is_inclusive: None,
            reasoning_tokens: reasoning,
            billed_at_ns: start,
        },
    )
}
