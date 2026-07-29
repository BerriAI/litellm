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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{PROVIDER_DEBUG_BODY_MAX_BYTES, redact_headers, redact_url, snapshot_json};

    #[test]
    fn redacts_sensitive_headers() {
        let headers = redact_headers(&[
            ("authorization".to_string(), "Bearer secret".to_string()),
            ("content-type".to_string(), "application/json".to_string()),
        ]);
        assert_eq!(headers["authorization"], "[REDACTED]");
        assert_eq!(headers["content-type"], "application/json");
    }

    #[test]
    fn redacts_sensitive_query_parameters() {
        let url = redact_url(
            "https://example.test/v1%3A0/invoke?X-Amz-Signature=secret&keep=value&key=hidden",
        );
        assert!(url.contains("X-Amz-Signature=%5BREDACTED%5D"));
        assert!(url.contains("key=%5BREDACTED%5D"));
        assert!(url.contains("keep=value"));
        assert!(url.contains("/v1%3A0/invoke"));
    }

    #[test]
    fn redacts_nested_body_keys() {
        let snapshot = snapshot_json(json!({
            "outer": {"token": "secret", "visible": "keep"},
            "items": [{"password": "hidden"}]
        }));
        assert_eq!(snapshot.body["outer"]["token"], "[REDACTED]");
        assert_eq!(snapshot.body["outer"]["visible"], "keep");
        assert_eq!(snapshot.body["items"][0]["password"], "[REDACTED]");
    }

    #[test]
    fn records_body_truncation_metadata() {
        let snapshot = snapshot_json(json!({"content": "x".repeat(PROVIDER_DEBUG_BODY_MAX_BYTES)}));
        assert_eq!(snapshot.body_truncated, Some(true));
        assert!(
            snapshot.body_original_bytes.expect("original bytes") > PROVIDER_DEBUG_BODY_MAX_BYTES
        );
    }
}
