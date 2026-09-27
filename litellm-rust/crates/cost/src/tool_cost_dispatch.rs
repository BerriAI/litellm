use std::borrow::Cow;

use serde_json::{Value, json};

use crate::anthropic_cost::{
    get_anthropic_web_search_requests_from_response, get_cost_for_anthropic_web_search,
    get_web_search_requests_from_usage,
};
use crate::catalog::ModelInfoCatalog;
use crate::gemini_cost::{
    cost_per_google_maps_grounding_request, cost_per_web_search_request,
    google_maps_grounding_requests,
};
use crate::groq_cost::cost_per_web_search_request as groq_web_search_cost;
use crate::provider::LlmProviders;
use crate::responses_usage::ChatUsage;
use crate::tool_call_cost_tracking::{
    DefaultToolRates, ResponseKind, count_web_search_calls, extract_file_search_params,
    extract_token_counts, get_cost_for_code_interpreter, get_cost_for_computer_use,
    get_cost_for_file_search, get_cost_for_vector_store, get_cost_for_web_search,
    response_object_includes_file_search_call, response_object_includes_web_search_call,
    safe_convert_to_int, usage_reports_server_side_web_search_calls,
};
use crate::wire::is_truthy;

#[derive(Clone, Copy, Debug)]
pub struct BuiltInToolCostRequest<'a> {
    pub response: &'a Value,
    pub response_kind: ResponseKind,
    pub usage: Option<&'a ChatUsage>,
    pub provider: Option<&'a str>,
    pub params: &'a Value,
    pub defaults: DefaultToolRates,
}

fn usage_reports_web_search(usage: &ChatUsage) -> bool {
    usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.web_search_requests)
        .is_some()
        || get_web_search_requests_from_usage(usage).is_some()
        || usage_reports_server_side_web_search_calls(
            usage.extra.get("server_side_tool_usage_details"),
        )
}

#[derive(Clone, Debug, Default)]
pub struct ResolvedModelInfo<'a> {
    pub model_info: Option<Cow<'a, Value>>,
    pub provider: Option<Cow<'a, str>>,
}

fn litellm_provider<'a>(model_info: &Value) -> Option<Cow<'a, str>> {
    model_info
        .get("litellm_provider")
        .and_then(Value::as_str)
        .map(|provider| Cow::Owned(provider.to_owned()))
}

pub fn resolve_model_info<'a>(
    catalog: &'a ModelInfoCatalog,
    model: &str,
    provider: Option<&'a str>,
    region: Option<&str>,
) -> ResolvedModelInfo<'a> {
    if let Some(direct) = catalog.entry(model, provider, region) {
        return ResolvedModelInfo {
            provider: provider
                .filter(|provider| !provider.is_empty())
                .map(Cow::Borrowed)
                .or_else(|| litellm_provider(&direct)),
            model_info: Some(direct),
        };
    }
    let by_prefix = model
        .contains('/')
        .then(|| catalog.entry(model, None, None))
        .flatten();
    match by_prefix {
        Some(by_prefix) => ResolvedModelInfo {
            provider: litellm_provider(&by_prefix),
            model_info: Some(by_prefix),
        },
        None => ResolvedModelInfo {
            model_info: None,
            provider: provider.map(Cow::Borrowed),
        },
    }
}

pub fn get_cost_for_web_search_request(
    provider: &str,
    usage: &ChatUsage,
    model_info: &Value,
    defaults: DefaultToolRates,
) -> Option<f64> {
    match provider.parse::<LlmProviders>().ok() {
        Some(LlmProviders::GEMINI) => Some(cost_per_web_search_request(usage, model_info)),
        Some(LlmProviders::ANTHROPIC) => Some(get_cost_for_anthropic_web_search(
            Some(model_info),
            Some(usage),
        )),
        _ if provider.starts_with(LlmProviders::VERTEX_AI.as_str()) => {
            let is_claude = model_info
                .get("key")
                .and_then(Value::as_str)
                .is_some_and(|key| key.to_ascii_lowercase().contains("claude"));
            Some(if is_claude {
                get_cost_for_anthropic_web_search(Some(model_info), Some(usage))
            } else {
                cost_per_web_search_request(usage, model_info)
            })
        }
        Some(LlmProviders::PERPLEXITY) => Some(0.0),
        Some(LlmProviders::XAI) => Some(crate::xai_cost::cost_per_web_search_request(
            usage,
            model_info,
            defaults.xai_web_search_per_call,
        )),
        Some(LlmProviders::GROQ) => Some(groq_web_search_cost(
            usage,
            model_info,
            defaults.groq_browser_open_per_call,
        )),
        _ => None,
    }
}

pub fn get_cost_for_google_maps_grounding_request(
    provider: &str,
    usage: &ChatUsage,
    model_info: &Value,
) -> Option<f64> {
    (LlmProviders::GEMINI.matches(Some(provider))
        || provider.starts_with(LlmProviders::VERTEX_AI.as_str()))
    .then(|| cost_per_google_maps_grounding_request(usage, model_info))
}

fn usage_with_anthropic_web_search<'a>(
    usage: Option<&'a ChatUsage>,
    response: &Value,
    kind: ResponseKind,
) -> Option<Cow<'a, ChatUsage>> {
    if usage.is_some_and(|usage| get_web_search_requests_from_usage(usage).is_some()) {
        return usage.map(Cow::Borrowed);
    }
    let from_response = (kind == ResponseKind::Anthropic)
        .then(|| get_anthropic_web_search_requests_from_response(response))
        .flatten();
    let Some(web_search_requests) = from_response else {
        return usage.map(Cow::Borrowed);
    };
    let base = usage.cloned().unwrap_or_default();
    let extra = base
        .extra
        .into_iter()
        .filter(|(key, _)| key != "server_tool_use")
        .chain([(
            "server_tool_use".to_owned(),
            json!({"web_search_requests": web_search_requests}),
        )])
        .collect();
    Some(Cow::Owned(ChatUsage { extra, ..base }))
}

fn maps_cost(usage: Option<&ChatUsage>, resolved: &ResolvedModelInfo<'_>) -> f64 {
    let Some(usage) = usage.filter(|usage| google_maps_grounding_requests(Some(usage)).is_some())
    else {
        return 0.0;
    };
    match (resolved.model_info.as_deref(), resolved.provider.as_deref()) {
        (Some(model_info), Some(provider)) => {
            get_cost_for_google_maps_grounding_request(provider, usage, model_info).unwrap_or(0.0)
        }
        _ => 0.0,
    }
}

fn web_search_cost(request: BuiltInToolCostRequest<'_>, resolved: &ResolvedModelInfo<'_>) -> f64 {
    let usage =
        usage_with_anthropic_web_search(request.usage, request.response, request.response_kind);
    let routed = match (
        resolved.model_info.as_deref(),
        usage.as_deref(),
        resolved.provider.as_deref(),
    ) {
        (Some(model_info), Some(usage), Some(provider)) => {
            get_cost_for_web_search_request(provider, usage, model_info, request.defaults)
        }
        _ => None,
    };
    routed.unwrap_or_else(|| {
        get_cost_for_web_search(
            request.params.get("web_search_options"),
            resolved.model_info.as_deref(),
        ) * count_web_search_calls(request.response, request.response_kind) as f64
    })
}

fn file_search_cost(request: BuiltInToolCostRequest<'_>, model_info: Option<&Value>) -> f64 {
    let file_search = request
        .params
        .get("file_search")
        .filter(|value| is_truthy(value));
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
    if !LlmProviders::AZURE.matches(request.provider) {
        return 0.0;
    }
    let vector = request
        .params
        .get("vector_store_usage")
        .filter(|value| is_truthy(value));
    let computer = request
        .params
        .get("computer_use_usage")
        .filter(|value| is_truthy(value));
    let sessions = request
        .params
        .get("code_interpreter_sessions")
        .filter(|value| is_truthy(value))
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
    catalog: &ModelInfoCatalog,
    model: &str,
    region: Option<&str>,
    request: BuiltInToolCostRequest<'_>,
) -> f64 {
    let resolved = resolve_model_info(catalog, model, request.provider, region);
    let maps = maps_cost(request.usage, &resolved);
    let web_search =
        response_object_includes_web_search_call(request.response, request.response_kind, None)
            || (request.response_kind != ResponseKind::Responses
                && request.usage.is_some_and(usage_reports_web_search));
    if web_search {
        return maps + web_search_cost(request, &resolved);
    }
    let direct = catalog.entry(model, request.provider, region);
    if response_object_includes_file_search_call(request.response, request.response_kind) {
        return maps + file_search_cost(request, direct.as_deref());
    }
    maps + azure_assistant_cost(request, direct.as_deref())
}
