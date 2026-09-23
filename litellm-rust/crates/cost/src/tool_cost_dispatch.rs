use serde_json::Value;

use crate::anthropic_cost::{
    get_anthropic_web_search_requests_from_response, get_cost_for_anthropic_web_search,
    get_web_search_requests_from_usage,
};
use crate::gemini_cost::{cost_per_google_maps_grounding_request, cost_per_web_search_request};
use crate::responses_usage::ChatUsage;
use crate::tool_call_cost_tracking::{
    DefaultToolRates, ResponseKind, count_web_search_calls, extract_file_search_params,
    extract_token_counts, get_cost_for_code_interpreter, get_cost_for_computer_use,
    get_cost_for_file_search, get_cost_for_vector_store, get_cost_for_web_search,
    response_object_includes_file_search_call, response_object_includes_web_search_call,
    safe_convert_to_int,
};

#[derive(Clone, Copy, Debug)]
pub struct BuiltInToolCostRequest<'a> {
    pub response: &'a Value,
    pub response_kind: ResponseKind,
    pub usage: Option<&'a ChatUsage>,
    pub provider: Option<&'a str>,
    pub params: &'a Value,
    pub defaults: DefaultToolRates,
}

fn server_tool_count(usage: &ChatUsage, key: &str) -> Option<u64> {
    usage
        .extra
        .get("server_tool_use")
        .and_then(|tools| tools.get(key))
        .and_then(Value::as_u64)
}

fn usage_reports_web_search(usage: &ChatUsage) -> bool {
    usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.web_search_requests)
        .is_some()
        || server_tool_count(usage, "web_search_requests").is_some()
        || usage
            .extra
            .get("server_side_tool_usage_details")
            .and_then(|details| details.get("web_search_calls"))
            .and_then(Value::as_u64)
            .is_some_and(|calls| calls > 0)
}

fn web_search_requests(request: BuiltInToolCostRequest<'_>) -> Option<u64> {
    request
        .usage
        .and_then(|usage| server_tool_count(usage, "web_search_requests"))
        .or_else(|| {
            request
                .response
                .get("usage")
                .and_then(|usage| usage.get("server_tool_use"))
                .and_then(|tools| tools.get("web_search_requests"))
                .and_then(Value::as_u64)
        })
}

fn context_rate(model_info: &Value) -> f64 {
    model_info
        .get("search_context_cost_per_query")
        .and_then(|rates| rates.get("search_context_size_medium"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
}

fn provider_web_search_cost(
    request: BuiltInToolCostRequest<'_>,
    model_info: &Value,
) -> Option<f64> {
    let provider = request.provider?;
    let usage = request.usage;
    match provider {
        "gemini" => usage.map(|usage| cost_per_web_search_request(usage, model_info)),
        "anthropic" => match usage.and_then(get_web_search_requests_from_usage) {
            Some(_) => Some(get_cost_for_anthropic_web_search(Some(model_info), usage)),
            None => get_anthropic_web_search_requests_from_response(request.response)
                .map(|count| context_rate(model_info) * count as f64),
        },
        "perplexity" => Some(0.0),
        "xai" => usage.map(|usage| {
            crate::xai_cost::cost_per_web_search_request(
                usage,
                model_info,
                request.defaults.xai_web_search_per_call,
            )
        }),
        "groq" => usage.map(|usage| {
            server_tool_count(usage, "web_search_requests").unwrap_or(0) as f64
                * context_rate(model_info)
                + server_tool_count(usage, "browser_open_requests").unwrap_or(0) as f64
                    * request.defaults.groq_browser_open_per_call
        }),
        provider if provider.starts_with("vertex_ai") => {
            let is_claude = model_info
                .get("key")
                .and_then(Value::as_str)
                .is_some_and(|key| key.to_ascii_lowercase().contains("claude"));
            if is_claude {
                web_search_requests(request).map(|count| context_rate(model_info) * count as f64)
            } else {
                usage.map(|usage| cost_per_web_search_request(usage, model_info))
            }
        }
        _ => None,
    }
}

fn maps_cost(request: BuiltInToolCostRequest<'_>, model_info: Option<&Value>) -> f64 {
    let supports_maps = matches!(request.provider, Some("gemini"))
        || request
            .provider
            .is_some_and(|provider| provider.starts_with("vertex_ai"));
    if !supports_maps {
        return 0.0;
    }
    match (request.usage, model_info) {
        (Some(usage), Some(model_info)) => {
            cost_per_google_maps_grounding_request(usage, model_info)
        }
        _ => 0.0,
    }
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => false,
        Value::Bool(true) => true,
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn file_search_cost(request: BuiltInToolCostRequest<'_>, model_info: Option<&Value>) -> f64 {
    let file_search = request
        .params
        .get("file_search")
        .filter(|value| truthy(value));
    let (storage_gb, days) = file_search
        .map(extract_file_search_params)
        .unwrap_or((None, None));
    get_cost_for_file_search(
        file_search,
        request.provider,
        model_info,
        storage_gb,
        days,
        request.defaults,
    )
}

fn azure_assistant_cost(request: BuiltInToolCostRequest<'_>, model_info: Option<&Value>) -> f64 {
    if request.provider != Some("azure") {
        return 0.0;
    }
    let vector = request
        .params
        .get("vector_store_usage")
        .filter(|value| truthy(value));
    let computer = request
        .params
        .get("computer_use_usage")
        .filter(|value| truthy(value));
    let sessions = request
        .params
        .get("code_interpreter_sessions")
        .filter(|value| truthy(value))
        .and_then(|value| safe_convert_to_int(Some(value)));
    let (input_tokens, output_tokens) = computer.map(extract_token_counts).unwrap_or((None, None));
    get_cost_for_vector_store(vector, request.provider, model_info, request.defaults)
        + get_cost_for_computer_use(
            input_tokens,
            output_tokens,
            request.provider,
            model_info,
            request.defaults,
        )
        + get_cost_for_code_interpreter(sessions, model_info, request.defaults)
}

pub fn get_cost_for_built_in_tools(
    request: BuiltInToolCostRequest<'_>,
    model_info: Option<&Value>,
) -> f64 {
    let maps = maps_cost(request, model_info);
    let web_search =
        response_object_includes_web_search_call(request.response, request.response_kind, None)
            || (request.response_kind != ResponseKind::Responses
                && request.usage.is_some_and(usage_reports_web_search));
    if web_search {
        let routed =
            model_info.and_then(|model_info| provider_web_search_cost(request, model_info));
        let fallback =
            get_cost_for_web_search(request.params.get("web_search_options"), model_info)
                * count_web_search_calls(request.response, request.response_kind) as f64;
        return maps + routed.unwrap_or(fallback);
    }
    if response_object_includes_file_search_call(request.response, request.response_kind) {
        return maps + file_search_cost(request, model_info);
    }
    maps + azure_assistant_cost(request, model_info)
}
