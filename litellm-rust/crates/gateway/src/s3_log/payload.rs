//! The proxy's `StandardLoggingPayload` for a request, from what the gateway knows of it, and
//! the transforms the proxy applies before logging it: base64 truncation, message redaction,
//! `s3_strip_base64_files` and `s3_log_prompts_only`.

use std::{
    collections::BTreeMap,
    time::{SystemTime, UNIX_EPOCH},
};

use litellm_gateway_hooks::{Record, Route};
use serde_json::{Map, Value, json};

use crate::{
    prometheus::error_class,
    spend::{logged_model, spend_logs_metadata, usage_values},
};

const REDACTED: &str = "redacted-by-litellm";
/// The proxy's `MAX_BASE64_LENGTH_FOR_LOGGING`: longer data URIs are logged as their size.
const MAX_BASE64_LENGTH: usize = 64;
const MAX_TRUNCATION_DEPTH: usize = 20;
/// `DEFAULT_MAX_RECURSE_DEPTH_SENSITIVE_DATA_MASKER`, the depth `s3_strip_base64_files` walks.
const MAX_STRIP_DEPTH: usize = 10;

/// The `StandardLoggingMetadata` keys, every one present and null when unset.
const METADATA_KEYS: [&str; 35] = [
    "user_api_key_hash",
    "user_api_key_alias",
    "user_api_key_spend",
    "user_api_key_max_budget",
    "user_api_key_budget_reset_at",
    "user_api_key_user_spend",
    "user_api_key_user_max_budget",
    "user_api_key_team_spend",
    "user_api_key_team_max_budget",
    "user_api_key_team_id",
    "user_api_key_org_id",
    "user_api_key_org_alias",
    "user_api_key_project_id",
    "user_api_key_project_alias",
    "user_api_key_user_id",
    "user_api_key_team_alias",
    "user_api_key_user_email",
    "user_api_key_end_user_id",
    "user_api_key_request_route",
    "spend_logs_metadata",
    "requester_ip_address",
    "user_agent",
    "requester_metadata",
    "prompt_management_metadata",
    "applied_guardrails",
    "mcp_tool_call_metadata",
    "vector_store_request_metadata",
    "routing_decision",
    "usage_object",
    "requester_custom_headers",
    "cold_storage_object_key",
    "user_api_key_auth_metadata",
    "team_alias",
    "team_id",
    "used_client_oauth_token",
];

/// What the proxy's settings change about every payload.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Redaction {
    /// `litellm_settings.turn_off_message_logging`.
    pub messages: bool,
    /// `litellm_settings.redact_user_api_key_info`.
    pub user_api_key_info: bool,
    /// The proxy's request tags carry the client's user agent unless
    /// `disable_add_user_agent_to_request_tags` is set.
    pub user_agent_tags: bool,
}

/// The request's `StandardLoggingPayload`.
pub(super) fn standard_logging_payload(
    record: &Record,
    redaction: Redaction,
) -> Map<String, Value> {
    let request = &record.request;
    let outcome = &record.outcome;
    let deployment = outcome.deployment.as_ref();
    let succeeded = record.succeeded();
    let body: Map<String, Value> = serde_json::from_slice(request.body()).unwrap_or_default();
    let response = match succeeded {
        true => response(record),
        false => Value::Object(Map::new()),
    };
    let id = response
        .get("id")
        .and_then(Value::as_str)
        .map_or_else(|| request.request_id.to_string(), str::to_owned);
    let usage = outcome.usage.unwrap_or_default();
    let start = seconds(request.started_at);
    let end = seconds(outcome.ended_at);
    let completion_start = match (request.stream, outcome.time_to_first_byte) {
        (true, Some(first_byte)) => seconds(request.started_at + first_byte),
        _ => end,
    };
    let status = match succeeded {
        true => "success",
        false => "failure",
    };
    let model = logged_model(record);
    let api_base = deployment
        .and_then(|deployment| deployment.api_base.as_deref())
        .map(strip_trailing_slash)
        .unwrap_or_default();
    let user_agent = request
        .client_headers
        .get(http::header::USER_AGENT)
        .and_then(|value| value.to_str().ok());
    let error_message = (!succeeded)
        .then(|| error_message(record))
        .flatten()
        .unwrap_or_default();
    let messages = body.get("messages").map(|messages| {
        let messages = with_system_prompt(messages.clone(), body.get("system"));
        truncate_base64(messages)
    });
    let hidden_params = json!({
        "model_id": deployment.map(|deployment| &*deployment.id),
        "cache_key": null,
        "api_base": deployment.and_then(|deployment| deployment.api_base.as_deref()),
        "response_cost": outcome.cost,
        "litellm_overhead_time_ms": null,
        "additional_headers": null,
        "batch_models": null,
        "batch_successful_requests": null,
        "batch_failed_requests": null,
        "litellm_model_name": deployment.map(|_| &model),
        "usage_object": null,
    });
    let error_information = json!({
        "error_code": if succeeded { String::new() } else { outcome.status.to_string() },
        "error_class": if succeeded { "" } else { error_class(record) },
        "llm_provider": deployment
            .filter(|_| !succeeded)
            .and_then(|deployment| deployment.provider())
            .unwrap_or_default(),
        "traceback": "",
        "error_message": &error_message,
    });
    let mut payload = json!({
        "id": id,
        "litellm_call_id": &*request.request_id,
        "trace_id": &*request.request_id,
        "session_id": "",
        "call_type": request.route.call_type(),
        "cache_hit": if request.stream { Value::Bool(false) } else { Value::Null },
        "stream": if request.stream && succeeded { Value::Bool(true) } else { Value::Null },
        "status": status,
        "status_fields": {"llm_api_status": status, "guardrail_status": "not_run"},
        "custom_llm_provider": deployment.and_then(|deployment| deployment.provider()),
        "saved_cache_cost": 0.0,
        "startTime": start,
        "endTime": end,
        "completionStartTime": completion_start,
        "response_time": match request.stream {
            true => completion_start - start,
            false => end - start,
        },
        "model": &model,
        "metadata": metadata(record, &body, user_agent, redaction),
        "cache_key": null,
        "response_cost": outcome.cost.unwrap_or(0.0),
        "cost_breakdown": null,
        "autorouter_savings": null,
        "autorouter_savings_estimate": null,
        "autorouter_baseline_observation": null,
    });
    let rest = json!({
        "total_tokens": usage.total_tokens(),
        "prompt_tokens": usage.prompt_tokens(),
        "completion_tokens": usage.output_tokens,
        "request_tags": request_tags(&body, user_agent.filter(|_| redaction.user_agent_tags)),
        "request_model_access_groups": [],
        "end_user": request.end_user.as_deref(),
        "api_base": api_base,
        "model_group": deployment.map_or("", |deployment| &deployment.model_group),
        "model_id": deployment.map_or("", |deployment| &deployment.id),
        "requester_ip_address": null,
        "user_agent": user_agent,
        "messages": messages,
        "response": response,
        "model_parameters": model_parameters(&body),
        "hidden_params": hidden_params,
        "model_map_information": {"model_map_key": &model, "model_map_value": null},
        "error_str": (!succeeded && !error_message.is_empty()).then_some(&error_message),
        "error_information": error_information,
        "response_cost_failure_debug_info": null,
        "zero_cost_diagnostic": null,
        "guardrail_information": null,
        "standard_built_in_tools_params": null,
    });
    if let (Value::Object(payload), Value::Object(rest)) = (&mut payload, rest) {
        payload.extend(rest);
    }
    if redaction.messages && message_logging_off(record) {
        redact(&mut payload);
    }
    match payload {
        Value::Object(payload) => payload,
        _ => unreachable!("the payload is built as an object"),
    }
}

fn seconds(time: SystemTime) -> f64 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn strip_trailing_slash(api_base: &str) -> &str {
    match api_base.ends_with("//") {
        true => api_base.trim_end_matches('/'),
        false => api_base.strip_suffix('/').unwrap_or(api_base),
    }
}

/// The proxy's `append_system_prompt_messages`: a string `system` becomes the first message.
fn with_system_prompt(messages: Value, system: Option<&Value>) -> Value {
    let Some(Value::String(system)) = system else {
        return messages;
    };
    let system_message = json!({"role": "system", "content": system});
    match messages {
        Value::Null => Value::Array(vec![system_message]),
        Value::Array(messages) => {
            if messages.first().is_some_and(|first| {
                first.get("role") == Some(&json!("system"))
                    && first.get("content") == Some(&json!(system))
            }) {
                return Value::Array(messages);
            }
            Value::Array(std::iter::once(system_message).chain(messages).collect())
        }
        Value::String(text) => json!([system_message, {"role": "user", "content": text}]),
        other => other,
    }
}

/// The proxy redacts unless the client opts out with `litellm-disable-message-redaction`.
fn message_logging_off(record: &Record) -> bool {
    !record
        .request
        .client_headers
        .get("litellm-disable-message-redaction")
        .is_some_and(|value| !value.is_empty())
}

/// The proxy's `_redact_standard_logging_object`.
fn redact(payload: &mut Value) {
    if !payload["messages"].is_null() {
        payload["messages"] = json!([{"role": "user", "content": REDACTED}]);
    }
    let response = &mut payload["response"];
    match response {
        Value::Null => {}
        Value::Object(fields) if fields.get("choices").is_some() => {
            if let Some(Value::Array(choices)) = fields.get_mut("choices") {
                choices.iter_mut().for_each(redact_choice);
            }
        }
        Value::String(_) => *response = Value::from(REDACTED),
        _ => *response = json!({"text": REDACTED}),
    }
}

fn redact_choice(choice: &mut Value) {
    let Value::Object(choice) = choice else {
        return;
    };
    let message = match (choice.contains_key("message"), choice.contains_key("delta")) {
        (true, _) => choice.get_mut("message"),
        (false, true) => choice.get_mut("delta"),
        (false, false) => {
            if choice.get("text").is_some_and(|text| !text.is_null()) {
                choice.insert("text".into(), Value::from(REDACTED));
            }
            return;
        }
    };
    let Some(Value::Object(message)) = message else {
        return;
    };
    for field in ["content", "reasoning_content"] {
        if message.get(field).is_some_and(|value| !value.is_null()) {
            message.insert(field.into(), Value::from(REDACTED));
        }
    }
    for field in ["thinking_blocks", "audio"] {
        if message.contains_key(field) {
            message.insert(field.into(), Value::Null);
        }
    }
    if let Some(Value::Array(calls)) = message.get_mut("tool_calls") {
        for call in calls {
            if let Some(Value::Object(function)) = call.get_mut("function") {
                function.insert("arguments".into(), Value::from(REDACTED));
            }
        }
    }
    if let Some(Value::Object(function)) = message.get_mut("function_call")
        && function.contains_key("arguments")
    {
        function.insert("arguments".into(), Value::from(REDACTED));
    }
}

fn metadata(
    record: &Record,
    body: &Map<String, Value>,
    user_agent: Option<&str>,
    redaction: Redaction,
) -> Value {
    let mut metadata: Map<String, Value> = METADATA_KEYS
        .iter()
        .map(|key| ((*key).to_owned(), Value::Null))
        .collect();
    let mut set = |key: &str, value: Value| {
        metadata.insert(key.to_owned(), value);
    };
    let request = &record.request;
    if let Some(key) = record.key().filter(|_| !redaction.user_api_key_info) {
        let budgets = &key.budgets;
        set("user_api_key_hash", json!(key.hash));
        set("user_api_key_alias", json!(key.alias));
        set("user_api_key_spend", json!(budgets.key.spend));
        set("user_api_key_max_budget", json!(budgets.key.max_budget));
        set(
            "user_api_key_user_spend",
            json!(budgets.user.map(|user| user.spend)),
        );
        set(
            "user_api_key_user_max_budget",
            json!(budgets.user.and_then(|user| user.max_budget)),
        );
        set(
            "user_api_key_team_spend",
            json!(budgets.team.map(|team| team.spend)),
        );
        set(
            "user_api_key_team_max_budget",
            json!(budgets.team.and_then(|team| team.max_budget)),
        );
        set("user_api_key_team_id", json!(key.team_id));
        set("user_api_key_org_id", json!(key.org_id));
        set("user_api_key_user_id", json!(key.user_id));
        set("user_api_key_team_alias", json!(key.team_alias));
    }
    if !redaction.user_api_key_info {
        set(
            "user_api_key_end_user_id",
            json!(request.end_user.as_deref()),
        );
        set(
            "user_api_key_request_route",
            json!(request.route.endpoint()),
        );
    }
    set("spend_logs_metadata", spend_logs_metadata(record));
    set("user_agent", json!(user_agent));
    if let Some(requester) = body.get("metadata").filter(|value| value.is_object()) {
        set("requester_metadata", requester.clone());
    }
    set(
        "usage_object",
        record
            .outcome
            .usage
            .map_or_else(|| json!({}), |usage| usage_values(&usage).0),
    );
    Value::Object(metadata)
}

/// The request's `metadata.tags`, then the proxy's user agent tags.
fn request_tags(body: &Map<String, Value>, user_agent: Option<&str>) -> Vec<Value> {
    let tags = body
        .get("metadata")
        .and_then(|metadata| metadata.get("tags"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let agent = user_agent.into_iter().flat_map(|agent| {
        agent
            .split_once('/')
            .map(|(product, _)| format!("User-Agent: {product}"))
            .into_iter()
            .chain(std::iter::once(format!("User-Agent: {agent}")))
            .map(Value::from)
    });
    tags.into_iter().chain(agent).collect()
}

/// The request's model parameters: everything but the model, the prompt and the metadata.
fn model_parameters(body: &Map<String, Value>) -> Map<String, Value> {
    body.iter()
        .filter(|(key, _)| {
            !matches!(
                key.as_str(),
                "model" | "messages" | "system" | "prompt" | "input" | "metadata"
            )
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

/// The provider's or the gateway's error message, from an error body when there is one.
fn error_message(record: &Record) -> Option<String> {
    let body: Value = serde_json::from_slice(record.response_body.as_deref()?).ok()?;
    let error = body.get("error").unwrap_or(&body);
    error
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| error.as_str())
        .map(str::to_owned)
}

/// The response the client got: the body itself, or the stream put back together the way the
/// proxy assembles it.
fn response(record: &Record) -> Value {
    let Some(body) = record.response_body.as_deref() else {
        return Value::Object(Map::new());
    };
    if let Ok(value @ Value::Object(_)) = serde_json::from_slice::<Value>(body) {
        return value;
    }
    let Ok(text) = std::str::from_utf8(body) else {
        return Value::Object(Map::new());
    };
    let events = text
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim)
        .filter_map(|data| serde_json::from_str::<Value>(data).ok());
    let assembled = match record.request.route {
        Route::ChatCompletions => chat_completion(events),
        Route::Messages => anthropic_message(events),
    };
    assembled.unwrap_or_else(|| Value::Object(Map::new()))
}

#[derive(Default)]
struct ChoiceParts {
    role: Option<Value>,
    content: Option<String>,
    reasoning: Option<String>,
    finish_reason: Value,
    tool_calls: BTreeMap<u64, Value>,
}

/// A chat completion from its chunks.
fn chat_completion(events: impl Iterator<Item = Value>) -> Option<Value> {
    let mut head: Option<Value> = None;
    let mut usage = Value::Null;
    let mut choices: BTreeMap<u64, ChoiceParts> = BTreeMap::new();
    for chunk in events {
        if head.is_none() {
            head = Some(json!({
                "id": chunk.get("id"),
                "created": chunk.get("created"),
                "model": chunk.get("model"),
                "system_fingerprint": chunk.get("system_fingerprint"),
            }));
        }
        if chunk.get("usage").is_some_and(|usage| !usage.is_null()) {
            usage = chunk["usage"].clone();
        }
        for choice in chunk["choices"].as_array().into_iter().flatten() {
            let parts = choices
                .entry(choice["index"].as_u64().unwrap_or(0))
                .or_default();
            let delta = &choice["delta"];
            if let Some(role) = delta.get("role").filter(|role| !role.is_null()) {
                parts.role = Some(role.clone());
            }
            if let Some(text) = delta["content"].as_str() {
                parts.content.get_or_insert_default().push_str(text);
            }
            if let Some(text) = delta["reasoning_content"].as_str() {
                parts.reasoning.get_or_insert_default().push_str(text);
            }
            if !choice["finish_reason"].is_null() {
                parts.finish_reason = choice["finish_reason"].clone();
            }
            for call in delta["tool_calls"].as_array().into_iter().flatten() {
                let slot = parts
                    .tool_calls
                    .entry(call["index"].as_u64().unwrap_or(0))
                    .or_insert_with(|| {
                        json!({"id": null, "type": "function", "function": {"name": null, "arguments": ""}})
                    });
                for field in ["id", "type"] {
                    if let Some(value) = call.get(field).filter(|value| !value.is_null()) {
                        slot[field] = value.clone();
                    }
                }
                if let Some(name) = call["function"]["name"].as_str() {
                    slot["function"]["name"] = Value::from(name);
                }
                if let Some(arguments) = call["function"]["arguments"].as_str() {
                    let joined = format!(
                        "{}{arguments}",
                        slot["function"]["arguments"].as_str().unwrap_or_default()
                    );
                    slot["function"]["arguments"] = Value::from(joined);
                }
            }
        }
    }
    let mut completion = head?;
    completion["object"] = Value::from("chat.completion");
    completion["choices"] = choices
        .into_iter()
        .map(|(index, parts)| {
            let tool_calls: Vec<Value> = parts.tool_calls.into_values().collect();
            let mut message = json!({
                "content": parts.content,
                "role": parts.role.unwrap_or_else(|| Value::from("assistant")),
                "tool_calls": if tool_calls.is_empty() { Value::Null } else { Value::Array(tool_calls) },
                "function_call": null,
            });
            if let Some(reasoning) = parts.reasoning {
                message["reasoning_content"] = Value::from(reasoning);
            }
            json!({"finish_reason": parts.finish_reason, "index": index, "message": message})
        })
        .collect();
    completion["usage"] = usage;
    Some(completion)
}

/// An Anthropic message from its stream events.
fn anthropic_message(events: impl Iterator<Item = Value>) -> Option<Value> {
    let mut message: Option<Value> = None;
    let mut blocks: BTreeMap<u64, Value> = BTreeMap::new();
    let mut partial_json: BTreeMap<u64, String> = BTreeMap::new();
    for event in events {
        let index = event["index"].as_u64().unwrap_or(0);
        match event["type"].as_str() {
            Some("message_start") => message = Some(event["message"].clone()),
            Some("content_block_start") => {
                blocks.insert(index, event["content_block"].clone());
            }
            Some("content_block_delta") => {
                let block = blocks.entry(index).or_insert_with(|| json!({}));
                let delta = &event["delta"];
                let append = |block: &mut Value, field: &str, text: &str| {
                    let joined = format!("{}{text}", block[field].as_str().unwrap_or_default());
                    block[field] = Value::from(joined);
                };
                match delta["type"].as_str() {
                    Some("text_delta") => {
                        append(block, "text", delta["text"].as_str().unwrap_or_default())
                    }
                    Some("thinking_delta") => append(
                        block,
                        "thinking",
                        delta["thinking"].as_str().unwrap_or_default(),
                    ),
                    Some("signature_delta") => block["signature"] = delta["signature"].clone(),
                    Some("input_json_delta") => partial_json
                        .entry(index)
                        .or_default()
                        .push_str(delta["partial_json"].as_str().unwrap_or_default()),
                    _ => {}
                }
            }
            Some("message_delta") => {
                let Some(message) = message.as_mut() else {
                    continue;
                };
                if let Some(delta) = event["delta"].as_object() {
                    for (field, value) in delta {
                        message[field.as_str()] = value.clone();
                    }
                }
                if let (Some(usage), Some(Value::Object(total))) =
                    (event["usage"].as_object(), message.get_mut("usage"))
                {
                    for (field, value) in usage {
                        if !value.is_null() {
                            total.insert(field.clone(), value.clone());
                        }
                    }
                }
            }
            _ => {}
        }
    }
    let mut message = message?;
    for (index, json) in partial_json {
        if let (Some(block), Ok(input)) = (blocks.get_mut(&index), serde_json::from_str(&json)) {
            block["input"] = input;
        }
    }
    message["content"] = Value::Array(blocks.into_values().collect());
    Some(message)
}

/// Replaces each data URI in `text` that `matcher` accepts; `matcher` gets the text from just
/// after `data:` and returns how much of it the URI spans and its replacement.
fn replace_data_uris(text: &str, matcher: impl Fn(&str) -> Option<(usize, String)>) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("data:") {
        let after = &rest[at + 5..];
        match matcher(after) {
            Some((length, replacement)) => {
                out.push_str(&rest[..at]);
                out.push_str(&replacement);
                rest = &after[length..];
            }
            None => {
                out.push_str(&rest[..at + 5]);
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// The length of the leading run of `text` whose characters satisfy `accept`.
fn run(text: &str, accept: impl Fn(char) -> bool) -> usize {
    text.find(|c: char| !accept(c)).unwrap_or(text.len())
}

fn is_base64(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '=')
}

/// The proxy's `truncate_base64_in_messages`: a data URI with more than 64 base64 characters
/// is logged as its size.
fn truncate_base64(value: Value) -> Value {
    fn walk(value: Value, depth: usize) -> Value {
        match value {
            Value::String(text) => Value::String(truncate_text(&text)),
            Value::Array(items) if depth <= MAX_TRUNCATION_DEPTH => Value::Array(
                items
                    .into_iter()
                    .map(|item| match item {
                        Value::String(_) => walk(item, depth),
                        item => walk(item, depth + 1),
                    })
                    .collect(),
            ),
            Value::Object(fields) if depth <= MAX_TRUNCATION_DEPTH => Value::Object(
                fields
                    .into_iter()
                    .map(|(key, item)| match item {
                        Value::String(_) => (key, walk(item, depth)),
                        item => (key, walk(item, depth + 1)),
                    })
                    .collect(),
            ),
            other => other,
        }
    }
    walk(value, 0)
}

fn truncate_text(text: &str) -> String {
    replace_data_uris(text, |after| {
        let mime = run(after, |c| !matches!(c, ';' | ',') && !c.is_whitespace());
        if mime == 0 || mime > 255 || !after[mime..].starts_with(";base64,") {
            return None;
        }
        let payload_start = mime + ";base64,".len();
        let payload = run(&after[payload_start..], is_base64);
        if payload == 0 {
            return None;
        }
        let length = payload_start + payload;
        if payload <= MAX_BASE64_LENGTH {
            return Some((length, format!("data:{}", &after[..length])));
        }
        Some((
            length,
            format!(
                "data:{};base64,[base64_data truncated: {}]",
                &after[..mime],
                base64_size(payload)
            ),
        ))
    })
}

/// The proxy's `format_base64_size`.
fn base64_size(characters: usize) -> String {
    let bytes = characters as f64 * 3.0 / 4.0;
    match bytes {
        bytes if bytes >= 1024.0 * 1024.0 => format!("{:.2}MB", bytes / (1024.0 * 1024.0)),
        bytes if bytes >= 1024.0 => format!("{:.1}KB", bytes / 1024.0),
        bytes => format!("{}B", bytes as u64),
    }
}

/// `s3_strip_base64_files`, as the proxy's `_strip_base64_from_messages_sync`: drops file and
/// non-text content parts and redacts inline base64 in what is left of each message.
pub(super) fn strip_base64_files(payload: &mut Map<String, Value>) {
    let Some(Value::Array(messages)) = payload.get_mut("messages") else {
        return;
    };
    if messages.is_empty() {
        return;
    }
    let stripped: Vec<Value> = std::mem::take(messages)
        .into_iter()
        .filter_map(|message| match message {
            Value::Object(message) => Some(Value::Object(strip_message(message))),
            _ => None,
        })
        .collect();
    *messages = stripped;
}

fn strip_message(mut message: Map<String, Value>) -> Map<String, Value> {
    let content = match message.remove("content") {
        Some(Value::Array(parts)) => Value::Array(
            parts
                .into_iter()
                .filter(keep_part)
                .map(|part| redact_base64(part, 0))
                .collect(),
        ),
        other => redact_base64(other.unwrap_or(Value::Null), 0),
    };
    let mut stripped: Map<String, Value> = message
        .into_iter()
        .map(|(key, value)| (key, redact_base64(value, 0)))
        .collect();
    stripped.insert("content".into(), content);
    stripped
}

fn keep_part(part: &Value) -> bool {
    let Value::Object(part) = part else {
        return true;
    };
    !part.contains_key("file")
        && part
            .get("type")
            .and_then(Value::as_str)
            .is_none_or(|kind| kind == "text")
}

fn redact_base64(value: Value, depth: usize) -> Value {
    if depth > MAX_STRIP_DEPTH {
        return Value::from("[MAX_DEPTH_REACHED]");
    }
    match value {
        Value::String(text) => Value::String(redact_text(&text)),
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(|item| redact_base64(item, depth + 1))
                .collect(),
        ),
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(key, item)| (key, redact_base64(item, depth + 1)))
                .collect(),
        ),
        other => other,
    }
}

/// Replaces `data:(application|image|audio|video)/<subtype>;base64,<payload>` with
/// `[BASE64_REDACTED]`, the payload running over base64 characters and whitespace.
fn redact_text(text: &str) -> String {
    replace_data_uris(text, |after| {
        let kind = ["application/", "image/", "audio/", "video/"]
            .into_iter()
            .find(|kind| after.starts_with(kind))?;
        let subtype = run(&after[kind.len()..], |c| {
            c.is_ascii_alphanumeric() || matches!(c, '.' | '+' | '-')
        });
        let rest = &after[kind.len() + subtype..];
        if subtype == 0 || !rest.starts_with(";base64,") {
            return None;
        }
        let payload_start = kind.len() + subtype + ";base64,".len();
        let payload = run(&after[payload_start..], |c| {
            is_base64(c) || c.is_whitespace()
        });
        (payload > 0).then(|| (payload_start + payload, "[BASE64_REDACTED]".to_owned()))
    })
}

/// `s3_log_prompts_only`: the payload without the response.
pub(super) fn prompts_only(payload: &mut Map<String, Value>) {
    payload.insert("response".into(), Value::Null);
}

#[cfg(test)]
pub(super) mod tests {
    use std::{sync::Arc, time::Duration};

    use axum::body::Bytes;
    use http::HeaderMap;
    use litellm_gateway_hooks::{
        Annotations, DeploymentFacts, KeyFacts, Outcome, RequestFacts, Usage,
    };
    use rstest::rstest;

    use super::*;

    pub(crate) const COMPLETION: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-x","choices":[{"index":0,"message":{"role":"assistant","content":"hi","tool_calls":[{"id":"t1","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}]},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":6,"total_tokens":11}}"#;

    pub(crate) fn record(
        route: Route,
        status: u16,
        stream: bool,
        request: Value,
        response: Option<&str>,
    ) -> Record {
        let started_at = UNIX_EPOCH + Duration::from_secs(1_000);
        let mut headers = HeaderMap::new();
        headers.insert("user-agent", "client/1.0".parse().unwrap());
        let key = KeyFacts {
            hash: "hash-1".into(),
            alias: Some("team-key".into()),
            user_id: Some("user-1".into()),
            team_id: Some("team-1".into()),
            team_alias: Some("research".into()),
            ..KeyFacts::default()
        };
        let mut facts = RequestFacts::new(
            "req-1".into(),
            route,
            Some(Arc::new(key)),
            Some("end-1".into()),
            "gpt".into(),
            Arc::new(headers),
            stream,
            None,
            Bytes::from(request.to_string()),
        );
        facts.started_at = started_at;
        Record {
            request: Arc::new(facts),
            outcome: Outcome {
                deployment: Some(DeploymentFacts {
                    id: "gpt-1".into(),
                    model_group: "gpt".into(),
                    model: "openai/gpt-x".into(),
                    provider: None,
                    api_base: Some("https://api.example/v1/".into()),
                }),
                status,
                usage: (status == 200).then_some(Usage {
                    input_tokens: 5,
                    output_tokens: 6,
                    ..Usage::default()
                }),
                cost: (status == 200).then_some(0.5),
                time_to_first_byte: Some(Duration::from_millis(250)),
                upstream_started_at: None,
                ended_at: started_at + Duration::from_secs(2),
                interrupted: false,
            },
            annotations: Annotations::default(),
            response_body: response.map(|body| Bytes::from(body.to_owned())),
        }
    }

    fn chat_request() -> Value {
        json!({
            "model": "gpt",
            "temperature": 0.2,
            "metadata": {"tags": ["prod"]},
            "messages": [{"role": "user", "content": "hi"}],
        })
    }

    #[test]
    fn a_success_carries_the_proxys_fields() {
        let record = record(
            Route::ChatCompletions,
            200,
            false,
            chat_request(),
            Some(COMPLETION),
        );
        let payload = Value::Object(standard_logging_payload(
            &record,
            Redaction {
                user_agent_tags: true,
                ..Redaction::default()
            },
        ));
        assert_eq!(payload["id"], "chatcmpl-1");
        assert_eq!(payload["litellm_call_id"], "req-1");
        assert_eq!(payload["call_type"], "acompletion");
        assert_eq!(payload["status"], "success");
        assert_eq!(payload["custom_llm_provider"], "openai");
        assert_eq!(payload["model"], "gpt-x");
        assert_eq!(payload["model_group"], "gpt");
        assert_eq!(payload["model_id"], "gpt-1");
        assert_eq!(payload["api_base"], "https://api.example/v1");
        assert_eq!(payload["response_cost"], 0.5);
        assert_eq!(payload["prompt_tokens"], 5);
        assert_eq!(payload["completion_tokens"], 6);
        assert_eq!(payload["total_tokens"], 11);
        assert_eq!(payload["startTime"], 1000.0);
        assert_eq!(payload["endTime"], 1002.0);
        assert_eq!(payload["completionStartTime"], 1002.0);
        assert_eq!(payload["response_time"], 2.0);
        assert_eq!(payload["stream"], Value::Null);
        assert_eq!(payload["end_user"], "end-1");
        assert_eq!(
            payload["messages"],
            json!([{"role": "user", "content": "hi"}])
        );
        assert_eq!(
            payload["response"],
            serde_json::from_str::<Value>(COMPLETION).unwrap()
        );
        assert_eq!(payload["model_parameters"], json!({"temperature": 0.2}));
        assert_eq!(
            payload["request_tags"],
            json!(["prod", "User-Agent: client", "User-Agent: client/1.0"])
        );
        let metadata = &payload["metadata"];
        assert_eq!(metadata.as_object().unwrap().len(), METADATA_KEYS.len());
        assert_eq!(metadata["user_api_key_hash"], "hash-1");
        assert_eq!(metadata["user_api_key_alias"], "team-key");
        assert_eq!(metadata["user_api_key_team_alias"], "research");
        assert_eq!(metadata["user_api_key_team_id"], "team-1");
        assert_eq!(metadata["user_api_key_user_id"], "user-1");
        assert_eq!(metadata["user_api_key_end_user_id"], "end-1");
        assert_eq!(metadata["requester_metadata"], json!({"tags": ["prod"]}));
        assert_eq!(metadata["usage_object"]["total_tokens"], 11);
        assert_eq!(payload["error_information"]["error_code"], "");
        assert_eq!(payload["error_str"], Value::Null);
    }

    #[test]
    fn a_failure_carries_the_error_and_an_empty_response() {
        let error = r#"{"error":{"message":"slow down","type":"rate_limit_error"}}"#;
        let record = record(
            Route::ChatCompletions,
            429,
            false,
            chat_request(),
            Some(error),
        );
        let payload = Value::Object(standard_logging_payload(&record, Redaction::default()));
        assert_eq!(payload["status"], "failure");
        assert_eq!(payload["id"], "req-1");
        assert_eq!(payload["response"], json!({}));
        assert_eq!(payload["response_cost"], 0.0);
        assert_eq!(payload["total_tokens"], 0);
        assert_eq!(payload["error_str"], "slow down");
        assert_eq!(
            payload["error_information"],
            json!({
                "error_code": "429",
                "error_class": "RateLimitError",
                "llm_provider": "openai",
                "traceback": "",
                "error_message": "slow down",
            })
        );
        assert_eq!(payload["metadata"]["usage_object"], json!({}));
        assert_eq!(payload["request_tags"], json!(["prod"]));
    }

    #[test]
    fn a_chat_stream_is_put_back_together() {
        let stream = concat!(
            "data: {\"id\":\"chatcmpl-9\",\"created\":7,\"model\":\"gpt-x\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"he\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl-9\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"llo\",\"tool_calls\":[{\"index\":0,\"id\":\"t1\",\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"a\\\"\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl-9\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\":1}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}\n\n",
            "data: [DONE]\n\n",
        );
        let record = record(
            Route::ChatCompletions,
            200,
            true,
            chat_request(),
            Some(stream),
        );
        let payload = Value::Object(standard_logging_payload(&record, Redaction::default()));
        assert_eq!(payload["id"], "chatcmpl-9");
        assert_eq!(payload["stream"], true);
        assert_eq!(payload["cache_hit"], false);
        assert_eq!(payload["completionStartTime"], 1000.25);
        assert_eq!(payload["response_time"], 0.25);
        let response = &payload["response"];
        assert_eq!(response["object"], "chat.completion");
        assert_eq!(response["created"], 7);
        assert_eq!(response["usage"]["total_tokens"], 3);
        assert_eq!(
            response["choices"],
            json!([{
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "hello",
                    "role": "assistant",
                    "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{\"a\":1}"}}],
                    "function_call": null,
                },
            }])
        );
    }

    #[test]
    fn a_messages_stream_is_put_back_together_with_its_system_prompt() {
        let stream = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-x\",\"stop_reason\":null,\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t\",\"name\":\"f\",\"input\":{}}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\\\":\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"1}\"}}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":42}}\n\n",
        );
        let request = json!({"model": "claude", "system": "be brief", "messages": [{"role": "user", "content": "hi"}]});
        let record = record(Route::Messages, 200, true, request, Some(stream));
        let payload = Value::Object(standard_logging_payload(&record, Redaction::default()));
        assert_eq!(payload["id"], "msg_1");
        assert_eq!(payload["call_type"], "anthropic_messages");
        assert_eq!(
            payload["messages"],
            json!([{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}])
        );
        let response = &payload["response"];
        assert_eq!(response["stop_reason"], "tool_use");
        assert_eq!(
            response["usage"],
            json!({"input_tokens": 10, "output_tokens": 42})
        );
        assert_eq!(
            response["content"],
            json!([{"type": "text", "text": "hi"}, {"type": "tool_use", "id": "t", "name": "f", "input": {"a": 1}}])
        );
    }

    #[test]
    fn turning_off_message_logging_redacts_messages_and_response() {
        let record = record(
            Route::ChatCompletions,
            200,
            false,
            chat_request(),
            Some(COMPLETION),
        );
        let redaction = Redaction {
            messages: true,
            ..Redaction::default()
        };
        let payload = Value::Object(standard_logging_payload(&record, redaction));
        assert_eq!(
            payload["messages"],
            json!([{"role": "user", "content": REDACTED}])
        );
        let message = &payload["response"]["choices"][0]["message"];
        assert_eq!(message["content"], REDACTED);
        assert_eq!(message["tool_calls"][0]["function"]["arguments"], REDACTED);
        assert_eq!(message["tool_calls"][0]["function"]["name"], "f");
        assert_eq!(payload["response"]["usage"]["total_tokens"], 11);
    }

    #[test]
    fn a_messages_response_is_redacted_to_text() {
        let message =
            r#"{"id":"msg_1","type":"message","content":[{"type":"text","text":"secret"}]}"#;
        let record = record(
            Route::Messages,
            200,
            false,
            json!({"messages": []}),
            Some(message),
        );
        let redaction = Redaction {
            messages: true,
            ..Redaction::default()
        };
        let payload = Value::Object(standard_logging_payload(&record, redaction));
        assert_eq!(payload["response"], json!({"text": REDACTED}));
        assert_eq!(payload["id"], "msg_1");
    }

    #[test]
    fn the_client_can_opt_out_of_redaction() {
        let mut record = record(
            Route::ChatCompletions,
            200,
            false,
            chat_request(),
            Some(COMPLETION),
        );
        let mut headers = HeaderMap::new();
        headers.insert("litellm-disable-message-redaction", "true".parse().unwrap());
        let request = Arc::get_mut(&mut record.request).unwrap();
        request.client_headers = Arc::new(headers);
        let redaction = Redaction {
            messages: true,
            ..Redaction::default()
        };
        let payload = Value::Object(standard_logging_payload(&record, redaction));
        assert_eq!(
            payload["messages"],
            json!([{"role": "user", "content": "hi"}])
        );
    }

    #[test]
    fn redacting_key_info_leaves_the_fields_null() {
        let record = record(
            Route::ChatCompletions,
            200,
            false,
            chat_request(),
            Some(COMPLETION),
        );
        let redaction = Redaction {
            user_api_key_info: true,
            ..Redaction::default()
        };
        let payload = Value::Object(standard_logging_payload(&record, redaction));
        assert_eq!(payload["metadata"]["user_api_key_team_alias"], Value::Null);
        assert_eq!(payload["metadata"]["user_api_key_hash"], Value::Null);
    }

    #[rstest]
    #[case::short("data:image/png;base64,AAAA", "data:image/png;base64,AAAA")]
    #[case::long(
        &format!("see data:image/png;base64,{} end", "A".repeat(4000)),
        "see data:image/png;base64,[base64_data truncated: 2.9KB] end"
    )]
    #[case::bytes(&format!("data:audio/wav;base64,{}", "A".repeat(100)), "data:audio/wav;base64,[base64_data truncated: 75B]")]
    #[case::not_base64("data:text/plain,hello", "data:text/plain,hello")]
    fn long_data_uris_in_messages_are_logged_as_their_size(
        #[case] text: &str,
        #[case] expected: &str,
    ) {
        assert_eq!(
            truncate_base64(json!([{"content": [{"image_url": {"url": text}}]}])),
            json!([{"content": [{"image_url": {"url": expected}}]}])
        );
    }

    #[test]
    fn stripping_base64_drops_file_parts_and_redacts_inline_data() {
        let mut payload = Map::new();
        payload.insert(
            "messages".into(),
            json!([
                {"role": "user", "content": [
                    {"type": "text", "text": "look data:image/png;base64,QUJD"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
                    {"file": {"file_data": "x"}},
                ]},
                {"role": "assistant", "content": "data:application/pdf;base64,QUJD\nQUJD"},
                {"role": "tool"},
                "not a message",
            ]),
        );
        strip_base64_files(&mut payload);
        assert_eq!(
            payload["messages"],
            json!([
                {"role": "user", "content": [{"type": "text", "text": "look [BASE64_REDACTED]"}]},
                {"role": "assistant", "content": "[BASE64_REDACTED]"},
                {"role": "tool", "content": null},
            ])
        );
    }

    #[test]
    fn prompts_only_drops_the_response() {
        let record = record(
            Route::ChatCompletions,
            200,
            false,
            chat_request(),
            Some(COMPLETION),
        );
        let mut payload = standard_logging_payload(&record, Redaction::default());
        prompts_only(&mut payload);
        assert_eq!(payload["response"], Value::Null);
        assert_eq!(
            payload["messages"],
            json!([{"role": "user", "content": "hi"}])
        );
        assert_eq!(payload["id"], "chatcmpl-1");
    }

    #[test]
    fn without_a_response_body_the_id_is_the_request_id() {
        let record = record(Route::ChatCompletions, 200, false, chat_request(), None);
        let payload = standard_logging_payload(&record, Redaction::default());
        assert_eq!(payload["id"], "req-1");
        assert_eq!(payload["response"], json!({}));
    }
}
