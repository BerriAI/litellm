use serde_json::{Map, Value};

pub const REDACTED: &str = "[REDACTED]";

pub const CREDENTIAL_HEADERS: &[&str] = &[
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "x-goog-api-key",
    "x-amz-security-token",
    "cookie",
];

pub fn is_credential_header(name: &str) -> bool {
    CREDENTIAL_HEADERS
        .iter()
        .any(|credential| name.eq_ignore_ascii_case(credential))
}

pub fn headers(values: &[(String, String)]) -> Vec<(String, String)> {
    values
        .iter()
        .map(|(name, value)| {
            let redacted = if is_credential_header(name) {
                REDACTED.to_string()
            } else {
                value.clone()
            };
            (name.clone(), redacted)
        })
        .collect()
}

pub fn params(value: &Value, secret_fields: &[String]) -> Value {
    let Value::Object(object) = value else {
        return value.clone();
    };
    Value::Object(
        object
            .iter()
            .map(|(name, value)| {
                let redacted = if secret_fields.iter().any(|secret| secret == name) {
                    Value::String(REDACTED.to_string())
                } else {
                    value.clone()
                };
                (name.clone(), redacted)
            })
            .collect::<Map<_, _>>(),
    )
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use serde_json::json;

    use super::*;

    #[test]
    fn credential_header_matching_is_ascii_case_insensitive() {
        assert!(is_credential_header("AuThOrIzAtIoN"));
        assert!(!is_credential_header("x-request-id"));
    }

    #[test]
    fn redaction_replaces_values_without_changing_shape() {
        let input = vec![
            ("X-Request-ID".to_string(), "abc".to_string()),
            ("Authorization".to_string(), "secret".to_string()),
            ("authorization".to_string(), "other".to_string()),
        ];
        assert_eq!(
            headers(&input),
            vec![
                ("X-Request-ID".to_string(), "abc".to_string()),
                ("Authorization".to_string(), REDACTED.to_string()),
                ("authorization".to_string(), REDACTED.to_string()),
            ]
        );
        assert_eq!(
            params(
                &json!({"api_key": "secret", "model": "m"}),
                &["api_key".to_string()]
            ),
            json!({"api_key": REDACTED, "model": "m"})
        );
    }

    proptest! {
        #[test]
        fn redaction_is_idempotent(
            name in "[A-Za-z0-9-]{1,20}",
            value in "[A-Za-z0-9]{0,20}"
        ) {
            let input = vec![(name, value)];
            prop_assert_eq!(headers(&headers(&input)), headers(&input));
        }
    }
}
