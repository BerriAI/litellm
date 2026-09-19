use fancy_regex::Regex;

pub const REDACTED: &str = "REDACTED";

const DEFAULT_MINIMUM_CUSTOM_KEY_LENGTH: usize = 16;

fn minimum_custom_key_length() -> usize {
    std::env::var("MINIMUM_CUSTOM_KEY_LENGTH")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(DEFAULT_MINIMUM_CUSTOM_KEY_LENGTH)
}

fn secret_patterns(minimum_custom_key_length: usize) -> String {
    let sk_suffix_length = minimum_custom_key_length.saturating_sub("sk-".len());
    [
        r"-----BEGIN[A-Z \-]*PRIVATE KEY-----[\s\S]*?-----END[A-Z \-]*PRIVATE KEY-----",
        r"\bya29\.[A-Za-z0-9_.~+/-]+",
        r#"(?:client_secret|azure_password|azure_username)\s+[^\s,'"})\]{}>]+"#,
        r"(?:AKIA|ASIA)[0-9A-Z]{16}",
        r"Bearer\s+[A-Za-z0-9\-._~+/]{10,}=*",
        r"Basic\s+[A-Za-z0-9+/]{10,}={0,2}",
        &format!(r"sk-[A-Za-z0-9\-_]{{{sk_suffix_length},}}"),
        r#"(?<=[?&])(?:api[_-]?key|\w*(?:token|password|passwd|client_secret|secret_key|_secret))=[^\s&'"]+"#,
        r#"(?:api[_-]?key)['"]?\s*[:=]\s*['"]?[^\s,'"})\]{}>]{8,}"#,
        r#"(?:x-api-key|api-key)['"]?\s*[:=]\s*['"]?[^\s,'"})\]{}>]+"#,
        r"x-ak-[A-Za-z0-9\-_]{20,}",
        r"AIza[0-9A-Za-z\-_]{35}",
        r#"(?<=[?&])key=[^\s&'"]{8,}"#,
        r#"(?:^|(?<=\W))\w*(?:password|passwd|client_secret|secret_key|_secret)['"]?\s*[:=]\s*['"]?[^\s,'"})\]{}>]+"#,
        r#"(?<=://)[^\s'":]{0,4096}:[^\s'"]{1,4096}(?=@)"#,
        r"dapi[0-9a-f]{32}",
        r#"litellm\.[A-Za-z0-9_]*_key['"]?\s*[:=]\s*['"]?[^\s,'"})\]{}>]+"#,
        r#"private_key['"]?\s*[:=]\s*['"]?(?:-----BEGIN[A-Z \-]*PRIVATE KEY-----[\s\S]*?-----END[A-Z \-]*PRIVATE KEY-----|[^\s,'"})\]{}>]+)"#,
        concat!(
            r"(?:master_key|xai_key|database_url|db_url|connection_string|",
            r"aws_secret_access_key|aws_session_token|aws_access_key_id|",
            r"signing_key|encryption_key|",
            r"auth_token|access_token|refresh_token|",
            r"slack_webhook_url|webhook_url|",
            r"database_connection_string|",
            r"huggingface_token|jwt_secret)",
            r#"['"]?\s*[:=]\s*['"]?[^\s,'"})\]{}>]+"#,
        ),
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*",
        r"(?<=[?&])sig=[A-Za-z0-9%+/=]+",
        r#"\{[^{}]*"type"\s*:\s*"service_account"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"#,
    ]
    .join("|")
}

/// Python's `_ENABLE_SECRET_REDACTION` pattern set, compiled once per configuration.
#[derive(Clone, Debug)]
pub struct SecretRedactor {
    pattern: Regex,
}

impl SecretRedactor {
    pub fn new(minimum_custom_key_length: usize) -> Self {
        let pattern = Regex::new(&format!(
            "(?i){}",
            secret_patterns(minimum_custom_key_length)
        ))
        .expect("secret redaction patterns compile");
        Self { pattern }
    }

    /// `None` when `LITELLM_DISABLE_REDACT_SECRETS` turns redaction off.
    pub fn from_env() -> Option<Self> {
        let disabled = std::env::var("LITELLM_DISABLE_REDACT_SECRETS")
            .is_ok_and(|value| value.eq_ignore_ascii_case("true"));
        (!disabled).then(|| Self::new(minimum_custom_key_length()))
    }

    pub fn redact(&self, value: &str) -> String {
        self.pattern.replace_all(value, REDACTED).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    #[case::bearer("auth failed: Bearer abcdefghijklmnop", "auth failed: REDACTED")]
    #[case::sk_key("key sk-abcdefghijklmnopqrstuvwxyz rejected", "key REDACTED rejected")]
    #[case::short_sk_key_is_kept("sk-abc", "sk-abc")]
    #[case::query_param("GET /v1?api_key=secret123&x=1", "GET /v1?REDACTED&x=1")]
    #[case::dict_repr("{'api_key': 'abcdefghij'}", "{'REDACTED'}")]
    #[case::url_credentials("postgres://user:pass@host/db", "postgres://REDACTED@host/db")]
    #[case::case_insensitive("BEARER ABCDEFGHIJKLMNOP", "REDACTED")]
    #[case::aws_key("AKIAABCDEFGHIJKLMNOP", "REDACTED")]
    #[case::sas_signature("https://x.blob/a?sv=1&sig=abc%2B=", "https://x.blob/a?sv=1&REDACTED")]
    #[case::password_needs_word_boundary("db_password=hunter2", "REDACTED")]
    #[case::plain_text_is_kept(r#"{"message": "rejected"}"#, r#"{"message": "rejected"}"#)]
    fn redacts_the_same_spans_as_the_python_patterns(#[case] input: &str, #[case] expected: &str) {
        assert_eq!(
            SecretRedactor::new(DEFAULT_MINIMUM_CUSTOM_KEY_LENGTH).redact(input),
            expected
        );
    }

    #[test]
    fn sk_threshold_follows_the_minimum_custom_key_length() {
        let redactor = SecretRedactor::new(8);
        assert_eq!(redactor.redact("sk-abcde"), REDACTED);
        assert_eq!(redactor.redact("sk-abcd"), "sk-abcd");
    }
}
