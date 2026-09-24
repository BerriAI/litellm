use std::collections::BTreeMap;

use jiff::Timestamp;
use serde_json::Value;

use crate::generic_cost::calculate_generic_cost_from_model_info_with_region;
use crate::responses_usage::{ChatUsage, PromptTokenDetails};
use crate::wire::{py_float, py_int};

pub fn reported_cost(usage: &ChatUsage) -> Option<f64> {
    usage.cost.filter(|cost| cost.is_finite() && *cost >= 0.0)
}

pub fn web_search_cost_per_call_from_model_info(model_info: &Value, default_rate: f64) -> f64 {
    [
        "search_context_size_medium",
        "search_context_size_low",
        "search_context_size_high",
    ]
    .into_iter()
    .filter_map(|key| {
        model_info
            .get("search_context_cost_per_query")?
            .get(key)
            .and_then(py_float)
    })
    .find(|rate| *rate > 0.0)
    .unwrap_or(default_rate)
}

pub fn cost_per_web_search_request(
    usage: &ChatUsage,
    model_info: &Value,
    default_rate: f64,
) -> f64 {
    if reported_cost(usage).is_some() {
        return 0.0;
    }
    let calls = usage
        .extra
        .get("server_side_tool_usage_details")
        .and_then(|details| details.get("web_search_calls"))
        .and_then(|value| count(Some(value)))
        .unwrap_or(0);
    calls as f64 * web_search_cost_per_call_from_model_info(model_info, default_rate)
}

fn count(value: Option<&Value>) -> Option<u64> {
    value
        .and_then(py_int)
        .and_then(|count| u64::try_from(count).ok())
}

pub fn apply_server_side_tool_usage_details_to_usage(
    usage: &ChatUsage,
    details: Option<&Value>,
) -> ChatUsage {
    let Some(details) = details else {
        return usage.clone();
    };
    let web_search_requests = count(details.get("web_search_calls")).filter(|count| *count > 0);
    ChatUsage {
        prompt_tokens_details: web_search_requests.map_or_else(
            || usage.prompt_tokens_details.clone(),
            |count| {
                Some(PromptTokenDetails {
                    web_search_requests: Some(count),
                    ..usage.prompt_tokens_details.clone().unwrap_or_default()
                })
            },
        ),
        extra: usage
            .extra
            .iter()
            .filter(|(key, _)| key.as_str() != "server_side_tool_usage_details")
            .map(|(key, value)| (key.clone(), value.clone()))
            .chain([("server_side_tool_usage_details".to_owned(), details.clone())])
            .collect(),
        ..usage.clone()
    }
}

pub fn cost_per_token(usage: &ChatUsage, model_info: &Value, at: Timestamp) -> (f64, f64) {
    if let Some(cost) = reported_cost(usage) {
        return (0.0, cost);
    }
    let reasoning_tokens = usage
        .completion_tokens_details
        .as_ref()
        .and_then(|details| details.reasoning_tokens)
        .unwrap_or(0);
    let completion_tokens =
        if usage.total_tokens == usage.prompt_tokens.saturating_add(usage.completion_tokens) {
            usage.completion_tokens
        } else {
            usage.completion_tokens.saturating_add(reasoning_tokens)
        };
    let normalized = ChatUsage {
        completion_tokens,
        completion_tokens_details: None,
        cost: None,
        extra: BTreeMap::new(),
        ..usage.clone()
    };
    calculate_generic_cost_from_model_info_with_region(
        &normalized,
        model_info,
        None,
        true,
        None,
        None,
        at,
    )
}
