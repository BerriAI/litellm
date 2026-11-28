//! Row fields LiteLLM derives from request stamps: the end user, the caller's session, whether the
//! client's oauth token was forwarded, the Kubernetes pod a request was routed to, Azure PTU
//! spillover, and the classifier audit fields. Each mirrors the Python helper named on it.

use litellm_python_compat::{Value, json};

use crate::consts::CLASSIFIER_AUDIT_FIELDS;
use crate::input::{LitellmParams, RowInput, text_of};
use crate::py::{Dict, NONE, get, get_or_none, has, text, to_str, truthy};

pub(crate) const CALLER_SESSION_ID_METADATA_KEY: &str = "litellm_caller_session_id";
const SESSION_ID_GENERATED_METADATA_KEY: &str = "litellm_session_id_generated";
const SESSION_ID_INHERITED_METADATA_KEY: &str = "litellm_session_id_inherited";
const KUBERNETES_POD_ROUTING_KEY: &str = "kubernetes_pod_routing";
const POD_SELECTIONS: [&str; 3] = ["round_robin", "session_affinity", "session_affinity_retry"];
const ANTHROPIC_OAUTH_FORWARD_PROVIDERS: [&str; 1] = ["anthropic"];
const AZURE_SPILLOVER_HEADER: &str = "x-ms-is-spilled-over";
const AZURE_SPILLOVER_FROM_HEADER: &str = "x-ms-spillover-from-deployment";

/// `get_end_user_id_for_cost_tracking(litellm_params)` for the spend log.
pub(crate) fn end_user_id(input: &RowInput, disabled: bool) -> Option<String> {
    if disabled {
        return None;
    }
    match &input.litellm_params.user_api_key_end_user_id {
        Some(end_user) if !end_user.is_empty() => Some(end_user.clone()),
        _ => input.auth.end_user_id.clone(),
    }
}

/// `as_str(value)` from the OTel model helpers: `None` stays `None`, anything else is `str(value)`.
fn as_text(value: &Value) -> Option<String> {
    text_of(value)
}

/// `caller_session_id(kwargs, TraceControls())`: the conversation id the caller sent.
fn caller_session_id(params: &LitellmParams) -> Option<String> {
    let bodies: Vec<&Value> = params.metadata_buckets().collect();
    let from_body: Vec<String> = bodies
        .iter()
        .filter_map(|body| as_text(get_or_none(body, "session_id")))
        .filter(|session| !session.is_empty())
        .collect();
    let minted: Vec<String> = bodies
        .iter()
        .filter(|body| truthy(get_or_none(body, SESSION_ID_GENERATED_METADATA_KEY)))
        .filter_map(|body| as_text(get_or_none(body, "session_id")))
        .filter(|session| !session.is_empty())
        .collect();
    if !minted.is_empty() {
        return from_body
            .into_iter()
            .find(|session| !minted.contains(session));
    }
    let explicit = params.litellm_session_id.clone();
    let echoes_trace_id = explicit.as_ref().is_some_and(|explicit| {
        bodies
            .iter()
            .any(|body| as_text(get_or_none(body, "trace_id")).as_ref() == Some(explicit))
    });
    if echoes_trace_id {
        None
    } else {
        explicit.filter(|session| !session.is_empty())
    }
}

/// `_caller_session_for_spend_log(kwargs, metadata=None)` when `inherits_parent_session(litellm_params)`.
pub(crate) fn inherited_parent_session(params: &LitellmParams) -> Option<String> {
    let inherited = params.metadata_buckets().any(|body| {
        matches!(
            get_or_none(body, SESSION_ID_INHERITED_METADATA_KEY),
            Value::Bool(true)
        )
    });
    if !inherited {
        return None;
    }
    caller_session_id(params).filter(|session| !session.contains('\0'))
}

/// `_caller_session_for_spend_log(kwargs, metadata)`.
pub(crate) fn caller_session(input: &RowInput) -> Option<String> {
    let session = input
        .auth
        .caller_session_id
        .clone()
        .or_else(|| caller_session_id(&input.litellm_params));
    session.filter(|session| !session.is_empty() && !session.contains('\0'))
}

/// `resolve_used_client_oauth_token(proxy_stamped_used_client_oauth_token(...), provider)`.
pub(crate) fn used_client_oauth_token(params: &LitellmParams, provider: Option<&str>) -> Value {
    let stamped = match params.litellm_metadata.mapping() {
        Some(bucket) if has(bucket, "used_client_oauth_token") => {
            get_or_none(bucket, "used_client_oauth_token")
        }
        _ => params.metadata.mapping().map_or(&NONE, |bucket| {
            get_or_none(bucket, "used_client_oauth_token")
        }),
    };
    match stamped {
        Value::Bool(false) => Value::Bool(false),
        Value::Bool(true) => Value::Bool(
            provider.is_some_and(|name| ANTHROPIC_OAUTH_FORWARD_PROVIDERS.contains(&name)),
        ),
        _ => Value::None,
    }
}

fn pod_routing_record(bucket: Option<&Value>) -> Option<Value> {
    let routing = get(bucket?, KUBERNETES_POD_ROUTING_KEY)?;
    let (
        Value::Str(service_host),
        Value::Str(pod_ip),
        Value::Int(pod_count),
        Value::Str(selection),
    ) = (
        get_or_none(routing, "service_host"),
        get_or_none(routing, "pod_ip"),
        get_or_none(routing, "pod_count"),
        get_or_none(routing, "selection"),
    )
    else {
        return None;
    };
    if *pod_count < 1.into() || !POD_SELECTIONS.contains(&selection.as_str()) {
        return None;
    }
    Some(Value::Dict(vec![
        (text("service_host"), text(service_host)),
        (text("pod_ip"), text(pod_ip)),
        (text("pod_count"), Value::Int(pod_count.clone())),
        (text("selection"), text(selection)),
    ]))
}

/// `proxy_stamped_kubernetes_pod_routing(metadata, litellm_params)`.
pub(crate) fn kubernetes_pod_routing(params: &LitellmParams) -> Value {
    pod_routing_record(params.litellm_metadata.mapping())
        .or_else(|| pod_routing_record(params.metadata.mapping()))
        .unwrap_or(Value::None)
}

/// `azure_spillover(response_headers, additional_headers)`.
pub(crate) fn azure_spillover(input: &RowInput) -> Value {
    let additional = input
        .payload
        .as_ref()
        .map(|payload| payload.hidden("additional_headers"))
        .filter(|headers| matches!(headers, Value::Dict(_)));
    for (headers, prefix) in [
        (input.kwargs.response_headers.as_ref(), ""),
        (additional, "llm_provider-"),
    ] {
        let Some(headers) = headers else {
            continue;
        };
        let spilled = get(headers, &format!("{prefix}{AZURE_SPILLOVER_HEADER}"))
            .is_some_and(|flag| to_str(flag).to_lowercase() == "true");
        if !spilled {
            continue;
        }
        let from_deployment =
            match get_or_none(headers, &format!("{prefix}{AZURE_SPILLOVER_FROM_HEADER}")) {
                Value::None => Value::None,
                found => Value::Str(to_str(found)),
            };
        return Value::Dict(vec![(text("from_deployment"), from_deployment)]);
    }
    Value::None
}

/// `classifier_input_snapshot(value)`: a JSON object, given as one or as its JSON text.
fn classifier_input_snapshot(value: &Value) -> Option<Value> {
    match value {
        Value::Dict(_) => Some(value.clone()),
        Value::Str(content) => json::loads(content)
            .ok()
            .filter(|parsed| matches!(parsed, Value::Dict(_))),
        _ => None,
    }
}

/// `classifier_audit_fields(payload)`.
pub(crate) fn classifier_audit(input: &RowInput) -> Value {
    let Some(payload) = &input.payload else {
        return Value::Dict(Vec::new());
    };
    let mut audit = Dict::default();
    for field in CLASSIFIER_AUDIT_FIELDS {
        if let Some(snapshot) = classifier_input_snapshot(payload.get(field)) {
            audit.set(field, snapshot);
        }
    }
    audit.into_value()
}
