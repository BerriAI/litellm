use fancy_regex::{NoExpand, Regex};

pub const REDACTED: &str = "REDACTED";

#[cfg(test)]
const DEFAULT_MINIMUM_CUSTOM_KEY_LENGTH: usize = 16;

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
            r"aws_secret_access_key|aws_session_token|aws_access_key_id|s3_secret_access_key|s3_access_key_id|",
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

#[derive(Clone, Debug)]
pub struct SecretRedactor {
    pattern: Regex,
    internal_pattern: Regex,
}

impl SecretRedactor {
    pub fn new(minimum_custom_key_length: usize) -> Self {
        let pattern = Regex::new(&format!(
            "(?i){}",
            secret_patterns(minimum_custom_key_length)
        ))
        .expect("secret redaction patterns compile");
        let internal_pattern = Regex::new(concat!(
            r#"(?i)/(?:etc|var|opt|usr|home|root|private|Users|tmp|mnt|srv)/[^\s'"\)\]}>,]+|"#,
            r#"[A-Za-z]:\\[^\s'"\)\]}>,]+|"#,
            r"\b(?:10(?:\.\d{1,3}){3}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|",
            r"192\.168(?:\.\d{1,3}){2}|127(?:\.\d{1,3}){3})\b|",
            r"\b[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:internal|local|corp|lan|intra|private)\b",
        ))
        .expect("internal detail patterns compile");
        Self {
            pattern,
            internal_pattern,
        }
    }

    pub fn redact(&self, value: &str) -> String {
        self.try_redact(value)
            .unwrap_or_else(|_| REDACTED.to_owned())
    }

    pub fn try_redact(&self, value: &str) -> fancy_regex::Result<String> {
        self.pattern
            .try_replacen(value, 0, NoExpand(REDACTED))
            .map(|value| value.into_owned())
    }

    pub fn try_redact_structured(
        &self,
        key: Option<&str>,
        value: &str,
    ) -> fancy_regex::Result<String> {
        let scrubbed = self.try_redact(value)?;
        if scrubbed != value || key.is_none() {
            return Ok(scrubbed);
        }
        let rendered = format!("'{}': '{value}'", key.unwrap_or_default());
        Ok(if self.try_redact(&rendered)? != rendered {
            REDACTED.to_owned()
        } else {
            value.to_owned()
        })
    }

    pub fn try_redact_internal(&self, value: &str) -> fancy_regex::Result<String> {
        let without_traceback = value
            .split_once("Traceback (most recent call last):")
            .map_or(value, |(prefix, _)| prefix.trim_end());
        self.internal_pattern
            .try_replacen(&self.try_redact(without_traceback)?, 0, NoExpand(REDACTED))
            .map(|value| value.into_owned())
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
