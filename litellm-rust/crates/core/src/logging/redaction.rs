use std::collections::BTreeMap;

use serde_json::{Map, Value};

pub const PROVIDER_DEBUG_BODY_MAX_BYTES: usize = 64 * 1024;

pub struct BodySnapshot {
    pub body: Value,
    pub body_truncated: Option<bool>,
    pub body_original_bytes: Option<usize>,
}

pub fn snapshot_json(value: Value) -> BodySnapshot {
    let redacted = redact_value(value);
    let serialized = serde_json::to_vec(&redacted).unwrap_or_default();
    if serialized.len() <= PROVIDER_DEBUG_BODY_MAX_BYTES {
        return BodySnapshot {
            body: redacted,
            body_truncated: None,
            body_original_bytes: None,
        };
    }
    BodySnapshot {
        body: Value::String(
            String::from_utf8_lossy(&serialized[..PROVIDER_DEBUG_BODY_MAX_BYTES]).into_owned(),
        ),
        body_truncated: Some(true),
        body_original_bytes: Some(serialized.len()),
    }
}

pub fn redact_headers(headers: &[(String, String)]) -> BTreeMap<String, String> {
    headers
        .iter()
        .map(|(name, value)| {
            let value = if matches!(
                name.to_ascii_lowercase().as_str(),
                "authorization"
                    | "proxy-authorization"
                    | "x-api-key"
                    | "api-key"
                    | "x-amz-security-token"
                    | "cookie"
                    | "set-cookie"
            ) {
                "[REDACTED]".to_string()
            } else {
                value.clone()
            };
            (name.clone(), value)
        })
        .collect()
}

pub fn redact_url(url: &str) -> String {
    let Ok(mut parsed) = url::Url::parse(url) else {
        return url.to_string();
    };
    let Some(_) = parsed.query() else {
        return parsed.to_string();
    };
    let pairs = parsed
        .query_pairs()
        .map(|(key, value)| {
            let value = if matches!(
                key.to_ascii_lowercase().as_str(),
                "x-amz-signature"
                    | "x-amz-credential"
                    | "x-amz-security-token"
                    | "api-key"
                    | "key"
                    | "access_token"
                    | "signature"
            ) {
                "[REDACTED]"
            } else {
                value.as_ref()
            };
            (key.into_owned(), value.to_string())
        })
        .collect::<Vec<_>>();
    parsed.query_pairs_mut().clear().extend_pairs(pairs);
    parsed.to_string()
}

fn redact_value(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(key, value)| {
                    if is_secret_key(&key) {
                        (key, Value::String("[REDACTED]".to_string()))
                    } else {
                        (key, redact_value(value))
                    }
                })
                .collect::<Map<_, _>>(),
        ),
        Value::Array(values) => Value::Array(values.into_iter().map(redact_value).collect()),
        other => other,
    }
}

fn is_secret_key(key: &str) -> bool {
    matches!(
        key.to_ascii_lowercase().as_str(),
        "api_key"
            | "apikey"
            | "secret"
            | "password"
            | "token"
            | "access_token"
            | "client_secret"
            | "aws_secret_access_key"
            | "aws_access_key_id"
            | "aws_session_token"
            | "x-amz-security-token"
    )
}
