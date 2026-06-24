use std::collections::HashMap;

const HEADER_PRESENT_PLACEHOLDER: &str = "[present]";

const DEFAULT_ALLOWLIST: &[&str] = &[
    "host",
    "accept-encoding",
    "connection",
    "accept",
    "content-type",
    "user-agent",
    "content-length",
];

const DEFAULT_GLOB_PATTERNS: &[&str] = &["x-stainless-", "x-litellm-"];

fn is_allowed(name: &str, extra_allowlist: &[String]) -> bool {
    let lower = name.to_ascii_lowercase();

    if DEFAULT_ALLOWLIST.contains(&lower.as_str()) {
        return true;
    }

    for prefix in DEFAULT_GLOB_PATTERNS {
        if lower.starts_with(prefix) {
            return true;
        }
    }

    for extra in extra_allowlist {
        if lower == extra.to_ascii_lowercase() {
            return true;
        }
    }

    false
}

/// Redact header values that are not on the allowlist, replacing them with a
/// placeholder so the generic guardrail API can see which headers were present
/// without receiving secrets.
pub fn sanitize_headers(
    raw: &HashMap<String, String>,
    extra_allowlist: &[String],
) -> HashMap<String, String> {
    raw.iter()
        .map(|(k, v)| {
            let value = if is_allowed(k, extra_allowlist) {
                v.clone()
            } else {
                HEADER_PRESENT_PLACEHOLDER.to_owned()
            };
            (k.clone(), value)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowlisted_headers_pass_through() {
        let raw = HashMap::from([
            ("Host".to_owned(), "example.com".to_owned()),
            ("Content-Type".to_owned(), "application/json".to_owned()),
            ("User-Agent".to_owned(), "test/1.0".to_owned()),
        ]);
        let result = sanitize_headers(&raw, &[]);
        assert_eq!(result.get("Host").unwrap(), "example.com");
        assert_eq!(result.get("Content-Type").unwrap(), "application/json");
        assert_eq!(result.get("User-Agent").unwrap(), "test/1.0");
    }

    #[test]
    fn non_allowlisted_headers_get_placeholder() {
        let raw = HashMap::from([
            ("Authorization".to_owned(), "Bearer secret123".to_owned()),
            ("X-Custom-Key".to_owned(), "my-api-key".to_owned()),
        ]);
        let result = sanitize_headers(&raw, &[]);
        assert_eq!(result.get("Authorization").unwrap(), "[present]");
        assert_eq!(result.get("X-Custom-Key").unwrap(), "[present]");
    }

    #[test]
    fn glob_patterns_match_prefixes() {
        let raw = HashMap::from([
            ("x-stainless-arch".to_owned(), "arm64".to_owned()),
            ("x-litellm-trace-id".to_owned(), "abc123".to_owned()),
        ]);
        let result = sanitize_headers(&raw, &[]);
        assert_eq!(result.get("x-stainless-arch").unwrap(), "arm64");
        assert_eq!(result.get("x-litellm-trace-id").unwrap(), "abc123");
    }

    #[test]
    fn extra_allowlist_extends_defaults() {
        let raw = HashMap::from([
            ("X-Custom-Key".to_owned(), "my-api-key".to_owned()),
            ("Authorization".to_owned(), "Bearer secret".to_owned()),
        ]);
        let extra = vec!["x-custom-key".to_owned()];
        let result = sanitize_headers(&raw, &extra);
        assert_eq!(result.get("X-Custom-Key").unwrap(), "my-api-key");
        assert_eq!(result.get("Authorization").unwrap(), "[present]");
    }

    #[test]
    fn empty_input_returns_empty() {
        let result = sanitize_headers(&HashMap::new(), &[]);
        assert!(result.is_empty());
    }
}
