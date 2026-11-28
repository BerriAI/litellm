//! The prompt-bearing columns of the row: the stored request body (`proxy_server_request`), the
//! response and the realtime messages. Each is stored only when prompt storage is on, trimmed to
//! the latest turn when configured, redacted when message logging is off, then sanitized
//! (credentials masked, tokens redacted, strings truncated) before it is dumped.

use litellm_python_compat::{Value, truthy::truthy};

use crate::compaction::{compaction_summary, summary_only_compaction_content};
use crate::consts::{CLASSIFIER_AUDIT_FIELDS, CLIENT_IP_HEADERS, TOOL_OUTPUT_BLOCK_TYPES};
use crate::input::{LoggingPayload, RowInput};
use crate::privacy::error_information_without_ip_addresses;
use crate::py::{Dict, as_str, entries, get, get_or_none, has, safe_dumps, text};
use crate::redaction::{redacted_request_body, redacted_response, should_redact};
use crate::stamps::classifier_audit;
use crate::text::masked_ip_text;
use crate::{RowError, SpendLogRowBuilder};

const RESPONSES_CALL_TYPES: [&str; 2] = ["responses", "aresponses"];
const STORED_CONVERSATION_CALL_TYPES: [&str; 3] =
    ["responses", "aresponses", "_aresponses_websocket"];
const STATEFUL_REQUEST_KEYS: [&str; 2] = ["previous_response_id", "conversation"];
const TURNS_KEYS: [&str; 2] = ["messages", "input"];
const TEXT_BLOCK_TYPES: [&str; 2] = ["text", "input_text"];
const REQUEST_METADATA_KEYS: [&str; 2] = ["metadata", "litellm_metadata"];

fn omitted_block_placeholder(block_type: &str) -> Option<&'static str> {
    match block_type {
        "image" | "image_url" | "input_image" => Some("[image omitted]"),
        "file" | "input_file" | "document" => Some("[file omitted]"),
        "input_audio" => Some("[audio omitted]"),
        _ => None,
    }
}

fn parts_of(value: &Value) -> &[Value] {
    match value {
        Value::List(items) => items,
        _ => &[],
    }
}

fn block_type(block: &Value) -> Option<&str> {
    as_str(get_or_none(block, "type"))
}

fn content_block_text(block: &Value) -> Option<String> {
    if let Value::Str(content) = block {
        return Some(content.clone());
    }
    let kind = block_type(block)?;
    if TEXT_BLOCK_TYPES.contains(&kind) {
        return as_str(get_or_none(block, "text")).map(str::to_owned);
    }
    omitted_block_placeholder(kind).map(str::to_owned)
}

fn user_content_text(content: &Value) -> Option<String> {
    if let Value::Str(content) = content {
        return (!content.is_empty()).then(|| content.clone());
    }
    let blocks = parts_of(content);
    if blocks
        .iter()
        .any(|block| block_type(block).is_some_and(|kind| TOOL_OUTPUT_BLOCK_TYPES.contains(&kind)))
    {
        return None;
    }
    let texts: Vec<String> = blocks
        .iter()
        .filter_map(content_block_text)
        .filter(|content| !content.is_empty())
        .collect();
    let joined = texts.join("\n");
    (!joined.is_empty()).then_some(joined)
}

fn latest_user_message_text(turns: &Value) -> Option<String> {
    if let Value::Str(content) = turns {
        return (!content.is_empty()).then(|| content.clone());
    }
    let latest = parts_of(turns).last()?;
    if !matches!(get_or_none(latest, "role"), Value::Str(role) if role == "user") {
        return None;
    }
    user_content_text(get_or_none(latest, "content"))
}

fn continues_stored_conversation(body: &Value, call_type: &Value) -> bool {
    matches!(call_type, Value::Str(kind) if STORED_CONVERSATION_CALL_TYPES.contains(&kind.as_str()))
        && STATEFUL_REQUEST_KEYS
            .iter()
            .any(|key| truthy(get_or_none(body, key)))
}

/// Whether the Responses API bridge answered the request through chat completions, whose stored
/// request is kept whole.
fn replayed_by_responses_bridge(payload: Option<&LoggingPayload>) -> bool {
    let Some(payload) = payload else {
        return false;
    };
    matches!(payload.get("call_type"), Value::Str(kind) if RESPONSES_CALL_TYPES.contains(&kind.as_str()))
        && matches!(payload.get("response"), response @ Value::Dict(_) if has(response, "choices"))
}

fn kept_model(body: &Value) -> Dict {
    let mut kept = Dict::default();
    if let Some(model) = get(body, "model") {
        kept.set("model", model.clone());
    }
    kept
}

fn trimmed_turn(body: &Value, call_type: &Value) -> Value {
    if continues_stored_conversation(body, call_type) {
        return body.clone();
    }
    let mut kept = kept_model(body);
    let Some(turns_key) = TURNS_KEYS.iter().find(|key| has(body, key)) else {
        return kept.into_value();
    };
    let Some(latest) = latest_user_message_text(get_or_none(body, turns_key)) else {
        return kept.into_value();
    };
    kept.set(
        turns_key,
        Value::List(vec![Value::Dict(vec![
            (text("role"), text("user")),
            (text("content"), Value::Str(latest)),
        ])]),
    );
    kept.into_value()
}

fn latest_turn_body(body: &Value, call_type: &Value, originating: &Value) -> Value {
    if !matches!(originating, Value::Dict(_)) {
        return trimmed_turn(body, call_type);
    }
    let trimmed_originating = trimmed_turn(originating, &Value::None);
    let mut stored = Dict::from_entries(entries(&trimmed_originating).unwrap_or_default());
    if let Some(model) = get(body, "model") {
        stored.set("model", model.clone());
    }
    stored.set("originating_request_masked", trimmed_originating);
    stored.into_value()
}

fn without_requester_ip_address(value: &Value, headers: bool) -> Value {
    match value {
        Value::Dict(items) => Value::Dict(
            items
                .iter()
                .filter(|(key, _)| {
                    !(headers
                        && as_str(key).is_some_and(|name| {
                            CLIENT_IP_HEADERS.contains(&name.to_lowercase().as_str())
                        }))
                })
                .map(|(key, item)| {
                    let stored = match item {
                        Value::Str(content) if headers => Value::Str(masked_ip_text(content)),
                        _ => stored_value_without_requester_ip_address(key, item),
                    };
                    (key.clone(), stored)
                })
                .collect(),
        ),
        Value::List(items) => Value::List(
            items
                .iter()
                .map(|item| without_requester_ip_address(item, false))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn stored_value_without_requester_ip_address(key: &Value, value: &Value) -> Value {
    match (as_str(key), value) {
        (Some("requester_ip_address"), _) => Value::None,
        (Some("error_information"), Value::Dict(_)) => {
            error_information_without_ip_addresses(value)
        }
        (name, _) => without_requester_ip_address(value, name == Some("headers")),
    }
}

fn request_metadata_without_requester_ip_address(body: &Value) -> Value {
    Value::Dict(
        entries(body)
            .unwrap_or_default()
            .iter()
            .map(|(key, value)| {
                let stored =
                    if as_str(key).is_some_and(|name| REQUEST_METADATA_KEYS.contains(&name)) {
                        without_requester_ip_address(value, false)
                    } else {
                        value.clone()
                    };
                (key.clone(), stored)
            })
            .collect(),
    )
}

fn stored_request_without_requester_ip_address(body: &Value) -> Value {
    let without_metadata = request_metadata_without_requester_ip_address(body);
    Value::Dict(
        entries(&without_metadata)
            .unwrap_or_default()
            .iter()
            .map(|(key, value)| {
                let stored = match value {
                    Value::Dict(_)
                        if as_str(key)
                            .is_some_and(|name| CLASSIFIER_AUDIT_FIELDS.contains(&name)) =>
                    {
                        request_metadata_without_requester_ip_address(value)
                    }
                    _ => value.clone(),
                };
                (key.clone(), stored)
            })
            .collect(),
    )
}

impl SpendLogRowBuilder {
    /// `_get_messages_for_spend_logs_payload`: only realtime calls store their messages.
    pub(crate) fn messages_json(&self, input: &RowInput) -> Result<String, RowError> {
        let Some(payload) = &input.payload else {
            return Ok("{}".to_owned());
        };
        if !input.stores_prompts {
            return Ok("{}".to_owned());
        }
        if !matches!(payload.get("call_type"), Value::Str(kind) if kind == "_arealtime") {
            return Ok("{}".to_owned());
        }
        match payload.get("messages") {
            Value::None => Ok("{}".to_owned()),
            messages => Ok(self.without_tokens(&safe_dumps(messages, true)?)),
        }
    }

    /// `_get_response_for_spend_logs_payload`.
    pub(crate) fn response_json(&self, input: &RowInput) -> Result<String, RowError> {
        let Some(payload) = &input.payload else {
            return Ok("{}".to_owned());
        };
        if !input.stores_prompts {
            return Ok("{}".to_owned());
        }
        let response = payload.get("response");
        if matches!(response, Value::None) {
            return Ok("{}".to_owned());
        }
        let tool_calls = &input.kwargs.realtime_tool_calls;
        let response = match response {
            Value::Dict(items) if truthy(tool_calls) => {
                let mut with_tool_calls = Dict::from_entries(items);
                with_tool_calls.set("tool_calls", tool_calls.clone());
                with_tool_calls.into_value()
            }
            other => other.clone(),
        };
        let response = if should_redact(&self.config, input) {
            redacted_response(&response)
        } else {
            response
        };
        let wrapper = self.sanitize_body(&[(text("response"), response)], true, true);
        match get_or_none(&wrapper, "response") {
            Value::None => Ok("{}".to_owned()),
            Value::Str(content) => Ok(content.replace('\0', "")),
            sanitized => safe_dumps(sanitized, true),
        }
    }

    /// `_get_proxy_server_request_for_spend_logs_payload`, given the request's
    /// `proxy_server_request` (absent, `None`, or the proxy's record of the client request).
    pub(crate) fn proxy_server_request_json(
        &self,
        input: &RowInput,
        proxy_server_request: Option<&Value>,
    ) -> Result<String, RowError> {
        if !input.stores_prompts {
            return Ok("{}".to_owned());
        }
        if matches!(proxy_server_request, Some(Value::None)) {
            return Ok("{}".to_owned());
        }
        let found_body =
            proxy_server_request.map_or(&Value::None, |found| get_or_none(found, "body"));
        let mut body = Dict::from_entries(entries(found_body).unwrap_or_default());
        if input.payload.is_some() {
            body.extend(entries(&classifier_audit(input)).unwrap_or_default());
        }
        let realtime_tools = &input.kwargs.realtime_tools;
        if truthy(realtime_tools) {
            body.set("tools", realtime_tools.clone());
        }
        let body = body.into_value();
        let payload = input.payload.as_ref();
        let selected = if !self.config.latest_turn_only || replayed_by_responses_bridge(payload) {
            body
        } else {
            match payload {
                None => latest_turn_body(&body, &Value::None, &Value::None),
                Some(payload) => latest_turn_body(
                    &body,
                    payload.get("call_type"),
                    payload.get("originating_request_masked"),
                ),
            }
        };
        let selected = if should_redact(&self.config, input) {
            redacted_request_body(&selected)
        } else {
            selected
        };
        let stored = self.sanitized_request_body(&selected);
        let stored = if self.config.drop_requester_ip_address {
            stored_request_without_requester_ip_address(&stored)
        } else {
            stored
        };
        safe_dumps(&stored, true)
    }

    /// `_sanitize_proxy_server_request_body(body)`: the sanitized body, with each Responses API
    /// `compaction` input item reduced to its bounded summary.
    fn sanitized_request_body(&self, body: &Value) -> Value {
        let sanitized = self.sanitize_body(entries(body).unwrap_or_default(), true, false);
        let (Value::List(items), Some(Value::List(stored_items))) =
            (get_or_none(body, "input"), get(&sanitized, "input"))
        else {
            return sanitized;
        };
        let input = items
            .iter()
            .zip(stored_items)
            .map(
                |(item, stored_item)| match (compaction_summary(item), stored_item) {
                    (Some(summary), Value::Dict(stored_entries)) => {
                        let mut bounded = Dict::from_entries(stored_entries);
                        bounded.set(
                            "encrypted_content",
                            Value::Str(summary_only_compaction_content(
                                &self.compaction_summary_for_db(&summary),
                            )),
                        );
                        bounded.into_value()
                    }
                    _ => stored_item.clone(),
                },
            )
            .collect();
        let mut with_input = Dict::from_entries(entries(&sanitized).unwrap_or_default());
        with_input.set("input", Value::List(input));
        with_input.into_value()
    }
}
