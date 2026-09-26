use serde_json::Value;

use crate::anthropic_cost::get_web_search_requests_from_usage;
use crate::responses_usage::ChatUsage;
use crate::wire::is_truthy;

const DEFAULT_WEB_SEARCH_COST: f64 = 35e-3;
const DEFAULT_MAPS_QUERY_COST: f64 = 14e-3;
const DEFAULT_MAPS_PROMPT_COST: f64 = 25e-3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WebSearchBillingUnit {
    PerPrompt,
    PerQuery,
    Unrecognized,
}

pub fn web_search_billing_unit(model_info: &Value) -> WebSearchBillingUnit {
    match model_info.get("web_search_billing_unit") {
        Some(unit) if is_truthy(unit) => match unit.as_str() {
            Some("per_prompt") => WebSearchBillingUnit::PerPrompt,
            Some("per_query") => WebSearchBillingUnit::PerQuery,
            _ => WebSearchBillingUnit::Unrecognized,
        },
        _ => WebSearchBillingUnit::PerPrompt,
    }
}

fn web_search_requests(usage: &ChatUsage) -> i64 {
    usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.web_search_requests)
        .and_then(|requests| i64::try_from(requests).ok())
        .filter(|requests| *requests != 0)
        .or_else(|| get_web_search_requests_from_usage(usage))
        .unwrap_or(0)
}

pub fn cost_per_web_search_request(usage: &ChatUsage, model_info: &Value) -> f64 {
    let rate = model_info
        .get("search_context_cost_per_query")
        .and_then(|rates| rates.get("search_context_size_medium"))
        .and_then(Value::as_f64)
        .unwrap_or(DEFAULT_WEB_SEARCH_COST);
    let requests = web_search_requests(usage);
    let billed =
        if requests > 0 && web_search_billing_unit(model_info) == WebSearchBillingUnit::PerPrompt {
            1
        } else {
            requests
        };
    rate * billed as f64
}

pub fn google_maps_grounding_requests(usage: Option<&ChatUsage>) -> Option<u64> {
    usage?
        .prompt_tokens_details
        .as_ref()?
        .google_maps_grounding_requests
}

pub fn cost_per_google_maps_grounding_request(usage: &ChatUsage, model_info: &Value) -> f64 {
    let Some(requests) = google_maps_grounding_requests(Some(usage)).filter(|count| *count > 0)
    else {
        return 0.0;
    };
    let per_query = web_search_billing_unit(model_info) == WebSearchBillingUnit::PerQuery;
    let default_rate = if per_query {
        DEFAULT_MAPS_QUERY_COST
    } else {
        DEFAULT_MAPS_PROMPT_COST
    };
    let rate = model_info
        .get("google_maps_grounding_cost_per_query")
        .and_then(Value::as_f64)
        .unwrap_or(default_rate);
    let billed = if per_query { requests } else { 1 };
    rate * billed as f64
}
