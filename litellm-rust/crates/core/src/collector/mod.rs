use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::error::{json_type_name, CoreError, CoreResult};

const COLLECTOR_CALL_TYPE: &str = "litellm-relay";
const DEFAULT_MODEL: &str = "local-ai-traffic";
const DEFAULT_API_KEY: &str = "litellm-relay";
const MAX_REQUEST_ID_HASH_CHARS: usize = 32;

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct CollectorAuthContext {
    pub key_hash: Option<String>,
    pub key_alias: Option<String>,
    pub team_id: Option<String>,
    pub team_alias: Option<String>,
    pub organization_id: Option<String>,
    pub user_id: Option<String>,
}

pub fn normalize_collector_spend_logs(
    logs: Vec<Value>,
    auth_context: CollectorAuthContext,
    now: &str,
) -> CoreResult<Vec<Value>> {
    logs.into_iter()
        .map(|log| normalize_collector_spend_log(log, &auth_context, now))
        .collect()
}

fn normalize_collector_spend_log(
    log: Value,
    auth_context: &CollectorAuthContext,
    now: &str,
) -> CoreResult<Value> {
    let input = match log {
        Value::Object(map) => map,
        other => {
            return Err(CoreError::InvalidType {
                expected: "object",
                actual: json_type_name(&other),
            })
        }
    };

    let collector_request_id = required_request_id(&input)?;
    let mut output = default_spend_log(now);

    for (key, value) in input.iter() {
        if is_collector_passthrough_field(key) {
            output.insert(key.clone(), value.clone());
        }
    }

    output.insert(
        "request_id".to_string(),
        json!(collector_request_id_for(auth_context, collector_request_id)),
    );
    output.insert("call_type".to_string(), json!(COLLECTOR_CALL_TYPE));
    output.insert(
        "api_key".to_string(),
        json!(auth_context.key_hash.as_deref().unwrap_or(DEFAULT_API_KEY)),
    );
    output.insert("spend".to_string(), json!(0.0));
    output.insert("custom_llm_provider".to_string(), json!(""));
    output.insert(
        "team_id".to_string(),
        optional_string_value(&auth_context.team_id),
    );
    output.insert(
        "organization_id".to_string(),
        optional_string_value(&auth_context.organization_id),
    );
    output.insert(
        "user".to_string(),
        optional_string_value(&auth_context.user_id),
    );

    let metadata = normalize_metadata(input.get("metadata"), auth_context, collector_request_id);
    output.insert("metadata".to_string(), Value::Object(metadata));
    output.insert(
        "request_tags".to_string(),
        json!(normalize_request_tags(input.get("request_tags"))),
    );

    Ok(sanitize_json_value(Value::Object(output)))
}

fn default_spend_log(now: &str) -> Map<String, Value> {
    let mut output = Map::new();
    output.insert("request_id".to_string(), json!(""));
    output.insert("call_type".to_string(), json!(COLLECTOR_CALL_TYPE));
    output.insert("api_key".to_string(), json!(DEFAULT_API_KEY));
    output.insert("spend".to_string(), json!(0.0));
    output.insert("total_tokens".to_string(), json!(0));
    output.insert("prompt_tokens".to_string(), json!(0));
    output.insert("completion_tokens".to_string(), json!(0));
    output.insert("startTime".to_string(), json!(now));
    output.insert("endTime".to_string(), json!(now));
    output.insert("model".to_string(), json!(DEFAULT_MODEL));
    output.insert("api_base".to_string(), json!(""));
    output.insert("custom_llm_provider".to_string(), json!(""));
    output.insert("user".to_string(), Value::Null);
    output.insert("metadata".to_string(), json!({}));
    output.insert("cache_hit".to_string(), json!("False"));
    output.insert("cache_key".to_string(), json!(""));
    output.insert("request_tags".to_string(), json!("[\"litellm-relay\"]"));
    output.insert("messages".to_string(), json!({}));
    output.insert("response".to_string(), json!({}));
    output.insert("proxy_server_request".to_string(), json!({}));
    output.insert("status".to_string(), json!("success"));
    output
}

fn required_request_id(input: &Map<String, Value>) -> CoreResult<&str> {
    match input.get("request_id") {
        Some(Value::String(request_id)) if !request_id.trim().is_empty() => Ok(request_id),
        Some(value) => Err(CoreError::InvalidType {
            expected: "non-empty string",
            actual: json_type_name(value),
        }),
        None => Err(CoreError::MissingField("request_id")),
    }
}

fn collector_request_id_for(
    auth_context: &CollectorAuthContext,
    collector_request_id: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(auth_context.key_hash.as_deref().unwrap_or(DEFAULT_API_KEY));
    hasher.update(b":");
    hasher.update(collector_request_id);
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(MAX_REQUEST_ID_HASH_CHARS);
    for byte in digest {
        encoded.push_str(&format!("{byte:02x}"));
        if encoded.len() >= MAX_REQUEST_ID_HASH_CHARS {
            encoded.truncate(MAX_REQUEST_ID_HASH_CHARS);
            break;
        }
    }
    format!("collector-{encoded}")
}

fn is_collector_passthrough_field(key: &str) -> bool {
    matches!(
        key,
        "total_tokens"
            | "prompt_tokens"
            | "completion_tokens"
            | "startTime"
            | "endTime"
            | "completionStartTime"
            | "model"
            | "model_id"
            | "model_group"
            | "mcp_namespaced_tool_name"
            | "agent_id"
            | "api_base"
            | "cache_hit"
            | "cache_key"
            | "end_user"
            | "requester_ip_address"
            | "messages"
            | "response"
            | "proxy_server_request"
            | "session_id"
            | "request_duration_ms"
            | "status"
    )
}

fn normalize_metadata(
    metadata: Option<&Value>,
    auth_context: &CollectorAuthContext,
    collector_request_id: &str,
) -> Map<String, Value> {
    let mut normalized = match metadata {
        Some(Value::Object(map)) => map.clone(),
        Some(Value::String(raw)) => {
            let mut map = Map::new();
            map.insert("relay_raw_metadata".to_string(), json!(raw));
            map
        }
        Some(Value::Null) | None => Map::new(),
        Some(raw) => {
            let mut map = Map::new();
            map.insert("relay_raw_metadata".to_string(), raw.clone());
            map
        }
    };

    normalized.insert("source".to_string(), json!(DEFAULT_API_KEY));
    normalized.insert(
        "collector_request_id".to_string(),
        json!(collector_request_id),
    );
    normalized.insert("relay_request_id".to_string(), json!(collector_request_id));
    normalized.insert(
        "user_api_key".to_string(),
        optional_string_value(&auth_context.key_hash),
    );
    normalized.insert(
        "user_api_key_alias".to_string(),
        optional_string_value(&auth_context.key_alias),
    );
    normalized.insert(
        "user_api_key_user_id".to_string(),
        optional_string_value(&auth_context.user_id),
    );
    normalized.insert(
        "user_api_key_team_id".to_string(),
        optional_string_value(&auth_context.team_id),
    );
    normalized.insert(
        "user_api_key_team_alias".to_string(),
        optional_string_value(&auth_context.team_alias),
    );
    normalized.insert(
        "user_api_key_org_id".to_string(),
        optional_string_value(&auth_context.organization_id),
    );
    normalized
}

fn optional_string_value(value: &Option<String>) -> Value {
    match value {
        Some(value) => json!(value),
        None => Value::Null,
    }
}

fn normalize_request_tags(value: Option<&Value>) -> String {
    let mut tags = match value {
        Some(Value::Array(tags)) => tags.clone(),
        Some(Value::String(raw)) => match serde_json::from_str::<Value>(raw) {
            Ok(Value::Array(tags)) => tags,
            _ if raw.trim().is_empty() => Vec::new(),
            _ => vec![json!(raw)],
        },
        Some(Value::Null) | None => Vec::new(),
        Some(other) => vec![other.clone()],
    };

    let has_relay_tag = tags.iter().any(|tag| tag.as_str() == Some(DEFAULT_API_KEY));
    if !has_relay_tag {
        tags.push(json!(DEFAULT_API_KEY));
    }

    serde_json::to_string(&tags).unwrap_or_else(|_| "[\"litellm-relay\"]".to_string())
}

fn sanitize_json_value(value: Value) -> Value {
    match value {
        Value::String(raw) => Value::String(raw.replace('\0', "")),
        Value::Array(items) => Value::Array(items.into_iter().map(sanitize_json_value).collect()),
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(key, value)| (key.replace('\0', ""), sanitize_json_value(value)))
                .collect(),
        ),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn auth_context() -> CollectorAuthContext {
        CollectorAuthContext {
            key_hash: Some("hashed-key".to_string()),
            key_alias: Some("relay-key".to_string()),
            team_id: Some("team-1".to_string()),
            team_alias: Some("Relay Team".to_string()),
            organization_id: Some("org-1".to_string()),
            user_id: Some("user-1".to_string()),
        }
    }

    #[test]
    fn normalizes_collector_log_with_authenticated_attribution() {
        let logs = vec![json!({
            "request_id": "raw-request-id",
            "model": "notion-ai",
            "api_key": "caller-supplied-key",
            "team_id": "caller-team",
            "organization_id": "caller-org",
            "user": "caller-user",
            "spend": 10,
            "custom_llm_provider": "caller-provider",
            "metadata": {
                "app": "notion",
                "source": "caller-source",
                "user_api_key": "caller-key"
            },
            "request_tags": ["notion"],
            "proxy_server_request": {"body_preview": "hi"},
            "response": {"body_preview": "hello"}
        })];

        let normalized =
            normalize_collector_spend_logs(logs, auth_context(), "2026-07-09T18:00:00+00:00")
                .expect("collector logs should normalize");
        let row = normalized[0].as_object().expect("row should be object");

        assert_eq!(row["call_type"], json!("litellm-relay"));
        assert_eq!(row["api_key"], json!("hashed-key"));
        assert_eq!(row["team_id"], json!("team-1"));
        assert_eq!(row["organization_id"], json!("org-1"));
        assert_eq!(row["user"], json!("user-1"));
        assert_eq!(row["spend"], json!(0.0));
        assert_eq!(row["custom_llm_provider"], json!(""));
        assert_ne!(row["request_id"], json!("raw-request-id"));
        assert!(row["request_id"]
            .as_str()
            .expect("request id should be string")
            .starts_with("collector-"));

        let metadata = row["metadata"]
            .as_object()
            .expect("metadata should be object");
        assert_eq!(metadata["app"], json!("notion"));
        assert_eq!(metadata["source"], json!("litellm-relay"));
        assert_eq!(metadata["collector_request_id"], json!("raw-request-id"));
        assert_eq!(metadata["user_api_key"], json!("hashed-key"));
        assert_eq!(metadata["user_api_key_alias"], json!("relay-key"));
        assert_eq!(metadata["user_api_key_team_alias"], json!("Relay Team"));
        assert_eq!(row["request_tags"], json!("[\"notion\",\"litellm-relay\"]"));
    }

    #[test]
    fn strips_nul_bytes_before_queueing() {
        let logs = vec![json!({
            "request_id": "nul-request",
            "proxy_server_request": {"body_preview": "hello\u{0}world"},
            "response": {"body_preview": "ok\u{0}"}
        })];

        let normalized =
            normalize_collector_spend_logs(logs, auth_context(), "2026-07-09T18:00:00+00:00")
                .expect("collector logs should normalize");
        let row_text = serde_json::to_string(&normalized[0]).expect("row should serialize");
        assert!(!row_text.contains('\0'));
    }

    #[test]
    fn rejects_missing_request_id() {
        let err = normalize_collector_spend_logs(
            vec![json!({"model": "notion-ai"})],
            auth_context(),
            "2026-07-09T18:00:00+00:00",
        )
        .expect_err("missing request id should fail");

        assert_eq!(err, CoreError::MissingField("request_id"));
    }
}
