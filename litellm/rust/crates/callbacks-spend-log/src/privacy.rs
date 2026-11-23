//! What the row may keep of prompts and credentials: the prompt storage decision, credential
//! masking and truncation of stored request and response bodies, token redaction, and the
//! redaction of guardrail records, vector store results and error text when prompts are not
//! stored in full.

use litellm_python_compat::Value;

use crate::SpendLogRowBuilder;
use crate::compaction::{compaction_entries_summary, is_compaction_item, with_compaction_summary};
use crate::consts::{
    NUMERIC_COMPRESSION_STAT_KEYS, PROMPT_CARRYING_GUARDRAIL_FIELDS, SCRUBBED_ERROR_TEXT_FIELDS,
    SENSITIVE_PATTERNS, TOOL_INPUT_BLOCK_TYPES, TOOL_OUTPUT_BLOCK_TYPES,
};
use crate::input::RowInput;
use crate::py::{Dict, as_str, char_len, entries, get_or_none, has, lookup, slice, text, truthy};
use crate::text::{masked_ip_text, redact_prompt_leaks_in_error_string};

pub(crate) const REDACTED: &str = "REDACTED_BY_LITELM";
/// The single key of the wrapper `rust_spend_log_rows._request_body` puts around a request body
/// value that was a Python object (a pydantic model such as `UserAPIKeyAuth`) rather than a dict.
/// `_sanitize_request_body_for_spend_logs_payload` only walks mappings, lists and strings, so such
/// a value is stored as the object serializes, unmasked and untruncated. The NUL makes the key one
/// no client body can hold, since the projection strips NUL from every real key.
pub(crate) const PYTHON_OBJECT_KEY: &str = "\0python_object";
pub(crate) const TRUNCATED_FIELD: &str = "litellm_truncated";
pub(crate) const TRUNCATION_NOTE: &str = "Truncation is a DB storage safeguard. Full, untruncated data is logged to logging callbacks (OTEL, Datadog, etc.). To increase the truncation limit, set `MAX_STRING_LENGTH_PROMPT_IN_DB` in your env.";
const MINIMUM_TOKEN_LENGTH: usize = 15;
const MAX_STRING_LENGTH_COMPACTION_SUMMARY_IN_DB: i64 = 64 * 1024;

impl SpendLogRowBuilder {
    /// `stores_full_prompts_in_spend_logs(metadata)`: prompts are stored for this request, and not
    /// trimmed to the latest turn.
    pub(crate) fn stores_full_prompts(&self, input: &RowInput) -> bool {
        input.stores_prompts && !self.config.latest_turn_only
    }

    pub(crate) fn without_tokens(&self, value: &str) -> String {
        if !self.config.redact_tokens {
            return value.to_owned();
        }
        let stripped = value.replace('\0', "");
        if char_len(&stripped) < MINIMUM_TOKEN_LENGTH {
            return stripped;
        }
        self.tokens.redact(&stripped, REDACTED).unwrap_or(stripped)
    }

    pub(crate) fn truncated_for_db(&self, value: &str) -> String {
        truncated(value, self.config.max_string_length_prompt_in_db)
    }

    /// `_compaction_summary_for_db(summary)`.
    pub(crate) fn compaction_summary_for_db(&self, summary: &str) -> String {
        truncated(
            &self.without_tokens(summary),
            self.config
                .max_string_length_prompt_in_db
                .max(MAX_STRING_LENGTH_COMPACTION_SUMMARY_IN_DB),
        )
    }

    /// `_bounded_compaction_summary(parent, key, value, keep_compaction_blocks)`.
    fn bounded_compaction_summary(
        &self,
        parent: &[(Value, Value)],
        key: &str,
        value: &Value,
    ) -> Option<String> {
        let Value::Str(content) = value else {
            return None;
        };
        match key {
            "content" => Some(self.compaction_summary_for_db(content)),
            "encrypted_content" => compaction_entries_summary(parent).map(|summary| {
                with_compaction_summary(content, &self.compaction_summary_for_db(&summary))
            }),
            _ => None,
        }
    }

    /// `_sanitize_request_body_for_spend_logs_payload(body, mask_credentials=...,
    /// keep_compaction_blocks=...)`: drops `secret_fields`, masks credential-named string values
    /// outside tool payloads, redacts tokens and truncates every string.
    pub(crate) fn sanitize_body(
        &self,
        body: &[(Value, Value)],
        mask_credentials: bool,
        keep_compaction_blocks: bool,
    ) -> Value {
        let compaction = keep_compaction_blocks && is_compaction_item(body);
        let content = ContentParent::of(body);
        let mut result = Vec::with_capacity(body.len());
        for (key, value) in body {
            if matches!(key, Value::Str(name) if name == "secret_fields") {
                continue;
            }
            let name = as_str(key).unwrap_or_default();
            let stored =
                if mask_credentials && matches!(value, Value::Str(_)) && is_sensitive_key(name) {
                    text(REDACTED)
                } else if let Some(bounded) = compaction
                    .then(|| self.bounded_compaction_summary(body, name, value))
                    .flatten()
                {
                    Value::Str(bounded)
                } else {
                    self.sanitize_value(
                        value,
                        mask_credentials && !content.holds(name),
                        keep_compaction_blocks,
                    )
                };
            result.push((key.clone(), stored));
        }
        Value::Dict(result)
    }

    fn sanitize_value(
        &self,
        value: &Value,
        mask_credentials: bool,
        keep_compaction_blocks: bool,
    ) -> Value {
        match value {
            Value::Dict(items) => match items.as_slice() {
                [(Value::Str(key), object)] if key == PYTHON_OBJECT_KEY => object.clone(),
                _ => self.sanitize_body(items, mask_credentials, keep_compaction_blocks),
            },
            Value::List(items) => Value::List(
                items
                    .iter()
                    .map(|item| self.sanitize_value(item, mask_credentials, keep_compaction_blocks))
                    .collect(),
            ),
            Value::Str(content) => Value::Str(self.truncated_for_db(&self.without_tokens(content))),
            other => other.clone(),
        }
    }

    pub(crate) fn sanitize_guardrail_information(
        &self,
        information: &Value,
        input: &RowInput,
    ) -> Value {
        if matches!(information, Value::None) || self.stores_full_prompts(input) {
            return information.clone();
        }
        let listed = match information {
            Value::Dict(_) => vec![information.clone()],
            Value::List(items) => items.clone(),
            Value::Str(_) => Vec::new(),
            other => return other.clone(),
        };
        Value::List(
            listed
                .iter()
                .filter_map(|entry| entries(entry).map(redact_guardrail_entry))
                .collect(),
        )
    }

    pub(crate) fn vector_store_request_metadata(
        &self,
        requests: &Value,
        input: &RowInput,
    ) -> Value {
        if self.stores_full_prompts(input) {
            return requests.clone();
        }
        match requests {
            Value::None => Value::None,
            Value::List(items) => {
                Value::List(items.iter().map(redacted_vector_store_request).collect())
            }
            other => other.clone(),
        }
    }

    pub(crate) fn sanitize_error_information(
        &self,
        information: &Value,
        unknown_model_message: Option<&Option<String>>,
        input: &RowInput,
    ) -> Value {
        let Some(items) = entries(information) else {
            return information.clone();
        };
        let mut sanitized = Dict::from_entries(items);
        if let Some(message) = unknown_model_message {
            sanitized.set(
                "error_message",
                message.as_deref().map_or(Value::None, text),
            );
        }
        if !self.stores_full_prompts(input) {
            for field in SCRUBBED_ERROR_TEXT_FIELDS {
                if let Some(Value::Str(content)) = sanitized.get(field) {
                    let redacted = redact_prompt_leaks_in_error_string(content);
                    sanitized.set(field, Value::Str(redacted));
                }
            }
        }
        let sanitized = self.sanitize_body(&dict_entries(sanitized.into_value()), true, false);
        if self.config.drop_requester_ip_address {
            error_information_without_ip_addresses(&sanitized)
        } else {
            sanitized
        }
    }
}

/// `_convert_to_json_serializable_dict`'s `max_depth`: a value nested this deep is stored as
/// [`MAX_DEPTH_EXCEEDED`].
pub(crate) const JSON_SERIALIZABLE_MAX_DEPTH: usize = 20;

/// What `_convert_to_json_serializable_dict` stores in place of a value nested past its depth.
const MAX_DEPTH_EXCEEDED: &str = "<max_depth_exceeded>";

/// `_convert_to_json_serializable_dict(value)`, which the redaction path runs first: every Python
/// object becomes the plain dict it dumps to, one level deeper than the object, so the sanitizer
/// then walks and masks it like any other mapping, and any value nested
/// [`JSON_SERIALIZABLE_MAX_DEPTH`] levels deep becomes `"<max_depth_exceeded>"`.
pub(crate) fn json_serializable(value: &Value) -> Value {
    json_serializable_within(value, JSON_SERIALIZABLE_MAX_DEPTH)
}

fn json_serializable_within(value: &Value, depth_left: usize) -> Value {
    if depth_left == 0 {
        return Value::Str(MAX_DEPTH_EXCEEDED.to_owned());
    }
    match value {
        Value::Dict(items) => match items.as_slice() {
            [(Value::Str(key), object)] if key == PYTHON_OBJECT_KEY => {
                json_serializable_within(object, depth_left - 1)
            }
            _ => Value::Dict(
                items
                    .iter()
                    .map(|(key, item)| {
                        (key.clone(), json_serializable_within(item, depth_left - 1))
                    })
                    .collect(),
            ),
        },
        Value::List(items) => Value::List(
            items
                .iter()
                .map(|item| json_serializable_within(item, depth_left - 1))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn dict_entries(value: Value) -> Vec<(Value, Value)> {
    match value {
        Value::Dict(items) => items,
        _ => Vec::new(),
    }
}

pub(crate) fn truncated(value: &str, max_length: i64) -> String {
    let length = char_len(value) as i64;
    if length <= max_length {
        return value.to_owned();
    }
    let start_chars = (max_length as f64 * 0.35) as i64;
    let end_chars = (max_length as f64 * 0.65) as i64;
    let tail = end_chars.min(max_length - start_chars);
    format!(
        "{}... ({TRUNCATED_FIELD} skipped {} chars. {TRUNCATION_NOTE}) ...{}",
        slice(value, None, Some(start_chars)),
        length - start_chars - end_chars,
        slice(value, Some(-tail), None),
    )
}

fn is_sensitive_key(key: &str) -> bool {
    let lowered = key.to_lowercase().replace('-', "_");
    let segments: Vec<&str> = lowered.split('_').collect();
    if segments.contains(&"cost") {
        return false;
    }
    SENSITIVE_PATTERNS
        .iter()
        .any(|pattern| segments.contains(pattern))
}

/// What `_is_spend_log_content` reads from a parent dict, looked up once per dict so a wide
/// object is walked in linear time.
struct ContentParent<'a> {
    block_type: Option<&'a str>,
    tool_role: bool,
}

impl<'a> ContentParent<'a> {
    fn of(parent: &'a [(Value, Value)]) -> Self {
        Self {
            block_type: lookup(parent, "type").and_then(as_str),
            tool_role: matches!(lookup(parent, "role"), Some(Value::Str(role)) if role == "tool"),
        }
    }

    /// `_is_spend_log_content(parent, key)`: the key holds tool payload or model output, which
    /// is kept unmasked.
    fn holds(&self, key: &str) -> bool {
        key == "arguments"
            || key == "logprobs"
            || (key == "input"
                && self
                    .block_type
                    .is_some_and(|kind| TOOL_INPUT_BLOCK_TYPES.contains(&kind)))
            || ((key == "content" || key == "output")
                && (self
                    .block_type
                    .is_some_and(|kind| TOOL_OUTPUT_BLOCK_TYPES.contains(&kind))
                    || self.tool_role))
    }
}

fn is_number(value: &Value) -> bool {
    matches!(value, Value::Int(_) | Value::Float(_))
}

fn redact_guardrail_entry(entry: &[(Value, Value)]) -> Value {
    let preserved: Option<Vec<(Value, Value)>> = lookup(entry, "guardrail_response")
        .and_then(entries)
        .map(|response| {
            response
                .iter()
                .filter(|(key, value)| {
                    as_str(key).is_some_and(|name| NUMERIC_COMPRESSION_STAT_KEYS.contains(&name))
                        && is_number(value)
                })
                .cloned()
                .collect::<Vec<_>>()
        })
        .filter(|stats| !stats.is_empty());
    let mut redacted = Dict::from_entries(entry);
    for field in PROMPT_CARRYING_GUARDRAIL_FIELDS {
        if lookup(entry, field).is_some() {
            redacted.set(field, text(REDACTED));
        }
    }
    if let Some(stats) = preserved {
        redacted.set("guardrail_response", Value::Dict(stats));
    }
    redacted.into_value()
}

fn redacted_vector_store_request(request: &Value) -> Value {
    let Some(items) = entries(request) else {
        return request.clone();
    };
    let found = get_or_none(request, "vector_store_search_response");
    let response = if truthy(found) {
        found.clone()
    } else {
        Value::Dict(Vec::new())
    };
    let data = sequence(get_or_none(&response, "data"))
        .iter()
        .map(redacted_vector_store_result)
        .collect();
    let mut stored_response = Dict::from_entries(entries(&response).unwrap_or_default());
    stored_response.set("data", Value::List(data));
    let mut stored = Dict::from_entries(items);
    stored.set("vector_store_search_response", stored_response.into_value());
    stored.into_value()
}

fn sequence(value: &Value) -> Vec<Value> {
    match value {
        Value::List(items) if !items.is_empty() => items.clone(),
        _ => Vec::new(),
    }
}

fn redacted_vector_store_result(result: &Value) -> Value {
    let contents = sequence(get_or_none(result, "content"))
        .iter()
        .map(|content| {
            if has(content, "text") {
                let mut stored = Dict::from_entries(entries(content).unwrap_or_default());
                stored.set("text", text(REDACTED));
                stored.into_value()
            } else {
                content.clone()
            }
        })
        .collect();
    let mut stored = Dict::from_entries(entries(result).unwrap_or_default());
    stored.set("content", Value::List(contents));
    stored.into_value()
}

pub(crate) fn error_information_without_ip_addresses(information: &Value) -> Value {
    let Some(items) = entries(information) else {
        return information.clone();
    };
    Value::Dict(
        items
            .iter()
            .map(|(key, value)| match (key, value) {
                (Value::Str(name), Value::Str(content))
                    if SCRUBBED_ERROR_TEXT_FIELDS.contains(&name.as_str()) =>
                {
                    (key.clone(), Value::Str(masked_ip_text(content)))
                }
                _ => (key.clone(), value.clone()),
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use litellm_python_compat::Value;

    use crate::py::{entries, get};
    use crate::test_support::{argument, case, field, helper_records, token_redactor};
    use crate::{RowConfig, SpendLogRowBuilder};

    fn flag(record: &Value, name: &str, default: bool) -> bool {
        match get(field(record, "kwargs"), name) {
            Some(Value::Bool(flag)) => *flag,
            _ => default,
        }
    }

    fn max_length(record: &Value) -> i64 {
        let recorded = get(field(record, "kwargs"), "max_string_length_prompt_in_db")
            .unwrap_or_else(|| field(record, "max_string_length_prompt_in_db"));
        match recorded {
            Value::Int(length) => i64::try_from(length).expect("length fits"),
            other => panic!("{other:?} is not a length"),
        }
    }

    // `_sanitize_request_body_for_spend_logs_payload` calls recorded by `scripts/record_goldens.py`
    #[test]
    fn sanitize_body_matches_every_recorded_python_call() {
        let tokens = token_redactor();
        for record in helper_records(include_str!(
            "../tests/goldens/helpers/sanitize_request_body_for_spend_logs_payload.jsonl"
        )) {
            let builder = SpendLogRowBuilder::new(
                RowConfig {
                    redact_tokens: matches!(field(&record, "redact_tokens"), Value::Bool(true)),
                    max_string_length_prompt_in_db: max_length(&record),
                    ..RowConfig::default()
                },
                tokens.clone(),
            );
            let body = entries(argument(&record, 0)).expect("the body is a dict");
            assert_eq!(
                &builder.sanitize_body(
                    body,
                    flag(&record, "mask_credentials", true),
                    flag(&record, "keep_compaction_blocks", false),
                ),
                field(&record, "output"),
                "{}",
                case(&record)
            );
        }
    }
}
