//! The `LiteLLM_SpendLogs` row: `get_logging_payload` from
//! `litellm/proxy/spend_tracking/spend_tracking_utils.py`, applied to the typed [`RowInput`].
//! It resolves the row's identity columns (request id, api key hash, model, model group,
//! provider, session), the cleaned metadata JSON, token counts and timings, and delegates the
//! prompt-bearing columns to [`crate::request`].

use std::sync::LazyLock;

use litellm_python_compat::{BigInt, Value, datetime::IsoDateTime, truthy::truthy, unicode};
use num_traits::ToPrimitive;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::consts::{
    BATCH_COST_REQUEST_ID_SUFFIX, BATCH_TRACE_CALL_TYPES, CUSTOMER_IDENTITY_FIELDS,
    NON_SECRET_KEY_ALIASES, SCRUBBED_ERROR_TEXT_FIELDS, SPEND_LOGS_METADATA_FIELDS,
    STAMPED_METADATA_KEYS,
};
use crate::input::{LoggingPayload, RequestedModel, RowAuthFacts, RowInput, Timestamp, text_of};
use crate::py::{
    Dict, NONE, as_str, entries, get, get_or_none, has, is_true, non_blank_str, or, safe_dumps,
    text, to_str,
};
use crate::stamps;
use crate::text::{UNKNOWN_MODEL, raw_model_spellings, scrub_raw_model_from_error_text};
use crate::{RowError, SpendLogRowBuilder};

const MCP_SPEND_LOG_MODEL_PREFIX: &str = "MCP: ";
const MAX_SPEND_LOG_MODEL_NAME_LENGTH: usize = 256;
const CLI_SESSION_KEY_PREFIX: &str = "cli-session";
const SPECIAL_USAGE_FIELDS: [&str; 3] = ["completion_tokens", "prompt_tokens", "total_tokens"];

/// Whether the request succeeded, as the row's `status` column stores it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpendLogStatus {
    Success,
    Failure,
}

/// One `LiteLLM_SpendLogs` row, as `get_logging_payload` returns it on main.
///
/// The JSON columns (`metadata`, `request_tags`, `messages`, `response`, `proxy_server_request`)
/// hold the JSON text the row stores. Timestamps are the UTC `isoformat()` text Python writes.
/// Serialized with serde, the field names and order are those of Python's `SpendLogsPayload`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpendLogRow {
    pub request_id: String,
    pub call_type: String,
    pub api_key: String,
    /// `"True"`, `"False"`, or `"None"` when the `cache_hit` kwarg is not a bool.
    pub cache_hit: String,
    #[serde(rename = "startTime")]
    pub start_time: String,
    #[serde(rename = "endTime")]
    pub end_time: String,
    #[serde(rename = "completionStartTime")]
    pub completion_start_time: Option<String>,
    pub model: String,
    pub user: String,
    pub team_id: String,
    pub organization_id: String,
    pub metadata: String,
    pub cache_key: String,
    /// The request's `response_cost`; the ledger replaces it with the billable cost.
    pub spend: Option<f64>,
    pub total_tokens: i64,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub request_tags: String,
    pub end_user: String,
    pub api_base: String,
    pub model_group: String,
    pub model_id: String,
    pub mcp_namespaced_tool_name: Option<String>,
    pub agent_id: Option<String>,
    #[serde(default)]
    pub billing_agent_id: Option<String>,
    pub requester_ip_address: Option<String>,
    pub custom_llm_provider: String,
    pub messages: String,
    pub response: String,
    pub proxy_server_request: String,
    pub session_id: Option<String>,
    pub request_duration_ms: Option<i64>,
    pub status: SpendLogStatus,
    pub litellm_call_id: Option<String>,
}

static SHA256_HEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\A[a-fA-F0-9]{64}\z").expect("sha256 pattern compiles"));
static HASHED_JWT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\Ahashed-jwt-[a-fA-F0-9]{64}\z").expect("hashed jwt pattern compiles")
});

/// The value with any case-insensitive `Bearer ` prefix dropped.
pub fn strip_bearer(value: &str) -> &str {
    match value.get(..7) {
        Some(prefix) if prefix.eq_ignore_ascii_case("bearer ") => &value[7..],
        _ => value,
    }
}

/// `hash_token`: the key's SHA-256 as lowercase hex.
pub fn hash_token(token: &str) -> String {
    format!("{:x}", Sha256::digest(token.as_bytes()))
}

/// The value is a SHA-256 hex digest, so it is already the logged form of a key.
pub fn is_sha256_hex(value: &str) -> bool {
    SHA256_HEX.is_match(value)
}

fn is_non_secret_key_value(value: &str, key_alias: &Value) -> bool {
    NON_SECRET_KEY_ALIASES.contains(&value)
        || is_sha256_hex(value)
        || HASHED_JWT.is_match(value)
        || (value.starts_with(&format!("{CLI_SESSION_KEY_PREFIX}-"))
            && matches!(key_alias, Value::Str(alias) if alias == value))
}

fn redact_logged_api_key(value: &Value, already_redacted: bool, key_alias: &Value) -> Value {
    let Some(raw) = as_str(value).filter(|raw| !raw.is_empty()) else {
        return Value::None;
    };
    let stripped = strip_bearer(raw);
    if stripped.is_empty() {
        return Value::None;
    }
    if already_redacted && is_non_secret_key_value(stripped, key_alias) {
        return text(stripped);
    }
    Value::Str(hash_token(stripped))
}

fn key_already_redacted(auth: &RowAuthFacts, api_key: &Value) -> bool {
    match &auth.user_api_key_hash {
        Some(hash) => {
            is_non_secret_key_value(hash, &auth.user_api_key_alias)
                && matches!(api_key, Value::Str(key) if key == hash)
        }
        None => false,
    }
}

fn looks_like_model_name(model: &str) -> bool {
    let candidate = model
        .strip_prefix(MCP_SPEND_LOG_MODEL_PREFIX)
        .unwrap_or(model);
    candidate.chars().count() <= MAX_SPEND_LOG_MODEL_NAME_LENGTH
        && !candidate.chars().any(unicode::isspace)
}

fn reconstruct_model_name(
    model_name: &str,
    custom_llm_provider: Option<&str>,
    deployment: Option<&str>,
) -> String {
    if let Some(deployment) = deployment
        && deployment.contains('/')
    {
        return deployment.to_owned();
    }
    if !model_name.is_empty() && !model_name.contains('/') && custom_llm_provider == Some("bedrock")
    {
        return format!("bedrock/{model_name}");
    }
    model_name.to_owned()
}

fn scrub_raw_model_from_error_information(information: &Value, raw_model: &str) -> Value {
    if matches!(information, Value::None) || raw_model.is_empty() {
        return information.clone();
    }
    let Some(items) = entries(information) else {
        return information.clone();
    };
    let spellings = raw_model_spellings(raw_model);
    Value::Dict(
        items
            .iter()
            .map(|(key, value)| match (key, value) {
                (Value::Str(name), Value::Str(content))
                    if SCRUBBED_ERROR_TEXT_FIELDS.contains(&name.as_str()) =>
                {
                    (
                        key.clone(),
                        Value::Str(scrub_raw_model_from_error_text(content, &spellings)),
                    )
                }
                _ => (key.clone(), value.clone()),
            })
            .collect(),
    )
}

/// `_placeholder_stored_request_body`: when the row's model is a placeholder, the stored request
/// body's model, model group and error text must not leak the raw model either.
fn placeholder_stored_request(
    proxy_server_request: Option<&Value>,
    persisted_model_group: &str,
    raw_model: &str,
) -> Option<Value> {
    let Some(Value::Dict(request_items)) = proxy_server_request else {
        return proxy_server_request.cloned();
    };
    let Some(body @ Value::Dict(body_items)) =
        proxy_server_request.and_then(|found| get(found, "body"))
    else {
        return proxy_server_request.cloned();
    };
    let mut placeholdered = Dict::from_entries(body_items);
    if has(body, "model") {
        placeholdered.set("model", text(UNKNOWN_MODEL));
    }
    if let Some(Value::Dict(metadata_items)) = get(body, "metadata") {
        let stored_metadata = metadata_items
            .iter()
            .map(|(key, value)| match as_str(key) {
                Some("model_group") => (key.clone(), text(persisted_model_group)),
                Some("error_information") => (
                    key.clone(),
                    match value {
                        Value::Dict(_) => scrub_raw_model_from_error_information(value, raw_model),
                        other => other.clone(),
                    },
                ),
                _ => (key.clone(), value.clone()),
            })
            .collect();
        placeholdered.set("metadata", Value::Dict(stored_metadata));
    }
    let mut request = Dict::from_entries(request_items);
    request.set("body", placeholdered.into_value());
    Some(request.into_value())
}

fn router_metadata(
    internal_router_model: bool,
    requested_model: &str,
    selected_model: &str,
    selected_provider: Option<&str>,
    router_correlation_id: Option<&str>,
) -> Value {
    if !internal_router_model {
        return Value::None;
    }
    let present = |value: Option<&str>| {
        value
            .filter(|value| !value.is_empty())
            .map_or(Value::None, text)
    };
    Value::Dict(vec![
        (text("requested_model"), present(Some(requested_model))),
        (text("selected_model"), present(Some(selected_model))),
        (text("selected_provider"), present(selected_provider)),
        (
            text("router_correlation_id"),
            router_correlation_id.map_or(Value::None, text),
        ),
    ])
}

fn stored_user_email(auth: &RowAuthFacts) -> Value {
    auth.user_email.as_deref().map_or(Value::None, text)
}

/// `request_duration_ms`: whole milliseconds between start and end, when both are naive or both
/// are aware.
fn duration_ms(start: &Timestamp, end: &Timestamp) -> Option<i64> {
    match (start.raw, end.raw) {
        (Some(IsoDateTime::Aware(start)), Some(IsoDateTime::Aware(end)))
        | (Some(IsoDateTime::Naive(start)), Some(IsoDateTime::Naive(end))) => {
            Some(((end - start) as f64 / 1e6 * 1000.0) as i64)
        }
        _ => None,
    }
}

/// `first or second` over optional text, where the empty string is falsy.
fn first_non_empty(first: Option<&str>, second: impl FnOnce() -> Option<String>) -> Option<String> {
    match first {
        Some(first) if !first.is_empty() => Some(first.to_owned()),
        _ => second(),
    }
}

/// A token count as the integer column stores it. Counts that are not integers are read as their
/// integer part, and anything that is not a number as zero.
fn token_count(value: &Value) -> i64 {
    match value {
        Value::Int(number) => number.to_i64().unwrap_or_default(),
        Value::Bool(flag) => i64::from(*flag),
        Value::Float(number) => *number as i64,
        _ => 0,
    }
}

/// The logging payload's attribution fields, read the way `get_logging_payload` falls back to them.
struct Attribution<'a> {
    payload: Option<&'a LoggingPayload>,
}

impl Attribution<'_> {
    /// `_sl_attribution_fallback`: the payload's field when it is text, else `""`.
    fn fallback(&self, field: &str) -> String {
        self.payload
            .and_then(|payload| as_str(payload.get(field)))
            .unwrap_or_default()
            .to_owned()
    }

    fn payload_get(&self, field: &str) -> &Value {
        self.payload.map_or(&NONE, |payload| payload.get(field))
    }

    fn metadata(&self, field: &str) -> &Value {
        self.payload
            .map_or(&NONE, |payload| payload.metadata(field))
    }

    fn hidden(&self, field: &str) -> &Value {
        self.payload.map_or(&NONE, |payload| payload.hidden(field))
    }
}

/// `get_logging_payload(kwargs, response_obj, start_time, end_time)`: the whole row.
pub(crate) fn build(
    builder: &SpendLogRowBuilder,
    input: &RowInput,
) -> Result<SpendLogRow, RowError> {
    let config = &builder.config;
    let metadata = &input.metadata;
    let kwargs = &input.kwargs;
    let litellm_params = &input.litellm_params;
    let usage = &input.usage;
    let payload = input.payload.as_ref();
    let attribution = Attribution { payload };
    let rejected_as_unknown_model = input.unknown_model_error_message.is_some();

    let payload_token_count = |field: &str| {
        payload
            .and_then(|payload| payload.field(field))
            .map_or(0, token_count)
    };
    let usage_token_count =
        |field: &str| get(usage, field).map_or_else(|| payload_token_count(field), token_count);

    let auth = &input.auth;
    let api_key_value = auth.user_api_key.clone().unwrap_or_else(|| text(""));
    let already_redacted = key_already_redacted(auth, &api_key_value);
    let mut api_key =
        redact_logged_api_key(&api_key_value, already_redacted, &auth.user_api_key_alias);
    if matches!(api_key, Value::None) {
        api_key = text("");
    }
    let mut end_user_id = stamps::end_user_id(input, config.disable_end_user_cost_tracking);
    if payload.is_some() {
        if !truthy(&api_key) {
            let from_payload = redact_logged_api_key(
                attribution.metadata("user_api_key_hash"),
                true,
                attribution.metadata("user_api_key_alias"),
            );
            api_key = if truthy(&from_payload) {
                from_payload
            } else {
                text("")
            };
        }
        if !config.disable_end_user_cost_tracking
            && end_user_id.as_deref().is_none_or(str::is_empty)
        {
            end_user_id = text_of(attribution.metadata("user_api_key_end_user_id"));
        }
    }
    let tags = auth.tags.clone().unwrap_or_else(|| Value::List(Vec::new()));
    let mut request_tags = if matches!(tags, Value::List(_)) {
        safe_dumps(&tags, false)?
    } else {
        "[]".to_owned()
    };
    if let Some(payload) = payload
        && !matches!(payload.get("request_tags"), Value::None)
    {
        request_tags = safe_dumps(payload.get("request_tags"), false)?;
    }

    let model_id = match &auth.model_id {
        Some(model_id) => model_id.clone(),
        None => attribution.fallback("model_id"),
    };
    let model_group = first_non_empty(auth.model_group.as_deref(), || {
        Some(attribution.fallback("model_group"))
    })
    .unwrap_or_default();
    let api_base = first_non_empty(litellm_params.api_base.as_deref(), || {
        Some(attribution.fallback("api_base"))
    })
    .unwrap_or_default();

    let litellm_overhead_time_ms = attribution.hidden("litellm_overhead_time_ms").clone();
    let upstream_provider = match attribution.hidden("upstream_provider") {
        provider @ Value::Str(_) => provider.clone(),
        _ => Value::None,
    };

    let logged_provider = first_non_empty(kwargs.custom_llm_provider.as_deref(), || {
        Some(attribution.fallback("custom_llm_provider")).filter(|provider| !provider.is_empty())
    });
    let custom_llm_provider = logged_provider
        .clone()
        .or_else(|| input.model_group_provider.clone())
        .filter(|provider| !provider.is_empty());
    let raw_model = kwargs.model.name();
    let model_is_malformed = matches!(kwargs.model, RequestedModel::Malformed);
    let resolved_model = match attribution.payload_get("model") {
        Value::Str(model) if !model.is_empty() => model.clone(),
        _ => reconstruct_model_name(
            raw_model,
            logged_provider.as_deref(),
            auth.deployment.as_deref(),
        ),
    };
    let failed = auth.failed;
    let failed_with_prompt_shaped_model = failed
        && model_id.is_empty()
        && !looks_like_model_name(&resolved_model)
        && !input.is_configured_model_group;
    let model_name =
        if rejected_as_unknown_model || failed_with_prompt_shaped_model || model_is_malformed {
            UNKNOWN_MODEL.to_owned()
        } else {
            resolved_model
        };
    let model_is_placeholdered = model_name == UNKNOWN_MODEL;
    let persisted_model_group = if model_is_placeholdered
        && model_group == raw_model
        && !looks_like_model_name(raw_model)
    {
        String::new()
    } else {
        model_group
    };
    let persisted_metadata = if model_is_placeholdered {
        let mut scrubbed = Dict::from_entries(entries(metadata).unwrap_or_default());
        scrubbed.set(
            "error_information",
            scrub_raw_model_from_error_information(
                get_or_none(metadata, "error_information"),
                raw_model,
            ),
        );
        scrubbed.into_value()
    } else {
        metadata.clone()
    };
    let litellm_call_id = first_non_empty(kwargs.litellm_call_id.as_deref(), || {
        litellm_params.litellm_call_id.clone()
    });

    let silent_experiment = truthy(attribution.metadata("is_silent_experiment"));
    let guardrail_information = match payload {
        Some(payload) => payload.get("guardrail_information").clone(),
        None => get_or_none(metadata, "standard_logging_guardrail_information").clone(),
    };
    let model_map_information = attribution.payload_get("model_map_information").clone();

    let mut clean = Dict::default();
    for field in SPEND_LOGS_METADATA_FIELDS {
        if !STAMPED_METADATA_KEYS.contains(&field) {
            clean.set(field, get_or_none(&persisted_metadata, field).clone());
        }
    }
    for field in CUSTOMER_IDENTITY_FIELDS {
        if let Some(content) = non_blank_str(get_or_none(&persisted_metadata, field)) {
            clean.set(field, text(content));
        }
    }
    if !matches!(upstream_provider, Value::None) {
        clean.set("upstream_provider", upstream_provider);
    }
    if let Some(route_call_type) = &input.route_call_type {
        clean.set("route_call_type", text(route_call_type));
    }
    for field in [
        "autorouter_savings",
        "autorouter_savings_estimate",
        "autorouter_baseline_observation",
    ] {
        clean.set(field, attribution.payload_get(field).clone());
    }
    clean.set(
        "router_metadata",
        router_metadata(
            auth.internal_router_model,
            &persisted_model_group,
            &model_name,
            custom_llm_provider.as_deref(),
            litellm_call_id.as_deref(),
        ),
    );
    clean.set("azure_spillover", stamps::azure_spillover(input));
    clean.set(
        "used_client_oauth_token",
        stamps::used_client_oauth_token(litellm_params, custom_llm_provider.as_deref()),
    );
    clean.set(
        "claude_subscription_usage",
        input.subscription_usage.clone(),
    );
    clean.set(
        "kubernetes_pod_routing",
        stamps::kubernetes_pod_routing(litellm_params),
    );
    clean.set(
        "litellm_roi_estimator",
        Value::Bool(is_true(get_or_none(
            &persisted_metadata,
            "litellm_roi_estimator",
        ))),
    );
    if silent_experiment {
        clean.set("is_silent_experiment", Value::Bool(true));
    }
    clean.set(
        "billed_retry_attempt",
        Value::Bool(is_true(get_or_none(
            &persisted_metadata,
            "billed_retry_attempt",
        ))),
    );
    clean.set("user_api_key_user_email", stored_user_email(auth));

    let raw_key = clean.get("user_api_key").cloned().unwrap_or(Value::None);
    let clean_key_redacted = key_already_redacted(auth, &raw_key);
    clean.set(
        "user_api_key",
        redact_logged_api_key(
            &raw_key,
            clean_key_redacted,
            get_or_none(&persisted_metadata, "user_api_key_alias"),
        ),
    );
    clean.set(
        "applied_guardrails",
        attribution.metadata("applied_guardrails").clone(),
    );
    for field in [
        "batch_models",
        "batch_successful_requests",
        "batch_failed_requests",
    ] {
        clean.set(field, attribution.hidden(field).clone());
    }
    clean.set(
        "mcp_tool_call_metadata",
        attribution.metadata("mcp_tool_call_metadata").clone(),
    );
    clean.set(
        "vector_store_request_metadata",
        builder.vector_store_request_metadata(
            attribution.metadata("vector_store_request_metadata"),
            input,
        ),
    );
    clean.set(
        "guardrail_information",
        builder.sanitize_guardrail_information(&guardrail_information, input),
    );
    clean.set("usage_object", attribution.metadata("usage_object").clone());
    clean.set("model_map_information", model_map_information);
    clean.set(
        "cold_storage_object_key",
        attribution.metadata("cold_storage_object_key").clone(),
    );
    clean.set("litellm_overhead_time_ms", litellm_overhead_time_ms);
    clean.set(
        "cost_breakdown",
        match payload {
            Some(payload) => payload.get("cost_breakdown").clone(),
            None => input.snapshotless_cost_breakdown.clone(),
        },
    );
    clean.set(
        "litellm_call_id",
        litellm_call_id.as_deref().map_or(Value::None, text),
    );

    let mut additional_usage_values = Dict::default();
    for (key, value) in entries(usage).unwrap_or_default() {
        if !as_str(key).is_some_and(|name| SPECIAL_USAGE_FIELDS.contains(&name)) {
            additional_usage_values.set_value(key.clone(), value.clone());
        }
    }
    let mut prompt_tokens_details = additional_usage_values
        .get("prompt_tokens_details")
        .cloned();
    if !matches!(prompt_tokens_details, Some(Value::Dict(_)))
        && let Some(usage_object @ Value::Dict(_)) = clean.get("usage_object")
    {
        prompt_tokens_details = get(usage_object, "prompt_tokens_details").cloned();
    }
    if let Some(details @ Value::Dict(_)) = prompt_tokens_details {
        if additional_usage_values
            .get("cache_read_input_tokens")
            .is_none()
            && let cached @ (Value::Int(_) | Value::Bool(_)) =
                get_or_none(&details, "cached_tokens")
            && positive(cached)
        {
            additional_usage_values.set("cache_read_input_tokens", cached.clone());
        }
        if additional_usage_values
            .get("cache_creation_input_tokens")
            .is_none()
        {
            let written = or(
                get_or_none(&details, "cache_write_tokens"),
                get_or_none(&details, "cache_creation_tokens"),
            );
            if matches!(written, Value::Int(_) | Value::Bool(_)) && positive(written) {
                additional_usage_values.set("cache_creation_input_tokens", written.clone());
            }
        }
    }
    clean.set(
        "additional_usage_values",
        additional_usage_values.into_value(),
    );
    if config.drop_requester_ip_address {
        clean.set("requester_ip_address", Value::None);
    }
    let error_information = clean
        .get("error_information")
        .cloned()
        .unwrap_or(Value::None);
    clean.set(
        "error_information",
        builder.sanitize_error_information(
            &error_information,
            input.unknown_model_error_message.as_ref(),
            input,
        ),
    );

    let request_id = if input.cache_hit == Some(true) || input.served_from_cache {
        let base = input.request_id.as_deref().unwrap_or("None");
        Some(format!(
            "{base}_cache_hit{}",
            litellm_python_compat::repr::float_repr(input.now)
        ))
    } else {
        input.request_id.clone()
    };

    let mcp_namespaced_tool_name = match clean.get("mcp_tool_call_metadata") {
        Some(found @ Value::Dict(_)) => text_of(get_or_none(found, "namespaced_tool_name")),
        _ => None,
    };
    let agent_id = first_non_empty(kwargs.agent_id.as_deref(), || auth.agent_id.clone());
    let billing_agent_id = clean.get("billing_agent_id").and_then(text_of);
    let requester_ip_address = clean.get("requester_ip_address").and_then(text_of);
    let metadata_json = safe_dumps(&clean.into_value(), true)?;

    let placeholdered_request = model_is_placeholdered
        .then(|| {
            placeholder_stored_request(
                litellm_params.proxy_server_request.as_ref(),
                &persisted_model_group,
                raw_model,
            )
        })
        .flatten();
    let stored_request = if model_is_placeholdered {
        placeholdered_request.as_ref()
    } else {
        litellm_params.proxy_server_request.as_ref()
    };
    let session_id = session_id(builder, input, request_id.as_deref());
    let cache_hit = match input.cache_hit {
        Some(true) => "True",
        Some(false) => "False",
        None => "None",
    };
    Ok(SpendLogRow {
        request_id: request_id.unwrap_or_else(|| "None".to_owned()),
        call_type: input.call_type.clone().unwrap_or_default(),
        api_key: to_str(&api_key),
        cache_hit: cache_hit.to_owned(),
        start_time: input.start_time.utc.clone(),
        end_time: input.end_time.utc.clone(),
        completion_start_time: input
            .completion_start_time
            .as_ref()
            .map(|time| time.utc.clone()),
        model: model_name,
        user: auth.user_id.clone(),
        team_id: auth.team_id.clone(),
        organization_id: auth.org_id.clone(),
        metadata: metadata_json,
        cache_key: input.cache_key.clone(),
        spend: kwargs.response_cost,
        total_tokens: usage_token_count("total_tokens"),
        prompt_tokens: usage_token_count("prompt_tokens"),
        completion_tokens: usage_token_count("completion_tokens"),
        request_tags,
        end_user: end_user_id.unwrap_or_default(),
        api_base,
        model_group: persisted_model_group,
        model_id,
        mcp_namespaced_tool_name,
        agent_id,
        billing_agent_id,
        requester_ip_address,
        custom_llm_provider: custom_llm_provider.unwrap_or_default(),
        messages: builder.messages_json(input)?,
        response: builder.response_json(input)?,
        proxy_server_request: builder.proxy_server_request_json(input, stored_request)?,
        session_id,
        request_duration_ms: duration_ms(&input.start_time, &input.end_time),
        status: if failed {
            SpendLogStatus::Failure
        } else {
            SpendLogStatus::Success
        },
        litellm_call_id,
    })
}

fn positive(value: &Value) -> bool {
    match value {
        Value::Int(number) => *number > BigInt::from(0),
        Value::Bool(flag) => *flag,
        _ => false,
    }
}

/// The row's `session_id`: the stamped session, else the parent session a proxy sub-call
/// inherited, when missing sessions are omitted, the batch id for batch calls, the caller's
/// session under request correlation, else the trace id.
fn session_id(
    builder: &SpendLogRowBuilder,
    input: &RowInput,
    request_id: Option<&str>,
) -> Option<String> {
    let omit = input.auth.session_id_omitted || builder.config.omit_missing_session_id;
    if omit {
        return input
            .auth
            .session_id
            .clone()
            .or_else(|| stamps::inherited_parent_session(&input.litellm_params));
    }
    if let (Some(kind), Some(id)) = (input.call_type.as_deref(), request_id)
        && BATCH_TRACE_CALL_TYPES.contains(&kind)
        && !id.is_empty()
    {
        return Some(
            id.strip_suffix(BATCH_COST_REQUEST_ID_SUFFIX)
                .unwrap_or(id)
                .to_owned(),
        );
    }
    if builder.config.request_correlation_in_logs
        && let Some(caller) = stamps::caller_session(input)
    {
        return Some(caller);
    }
    if let Some(payload) = &input.payload
        && let Some(trace_id) = payload.field("trace_id")
        && !matches!(trace_id, Value::None)
    {
        return Some(to_str(trace_id));
    }
    Some(
        input
            .kwargs
            .litellm_trace_id
            .clone()
            .unwrap_or_else(|| input.fallback_session_id.clone()),
    )
}
