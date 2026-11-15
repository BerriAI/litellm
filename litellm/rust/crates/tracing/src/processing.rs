use fancy_regex::Result;
use percent_encoding::percent_decode_str;

use crate::{REDACTED, SecretRedactor};

#[derive(Clone, Copy, Debug)]
pub struct Policy {
    pub redact: bool,
    pub base64_limit: i64,
    pub text_limit: i64,
}

#[derive(Clone, Debug)]
pub struct DiagnosticInput {
    pub message: String,
    pub exception: Option<String>,
    pub stack: Option<String>,
    pub leaves: Vec<(Option<String>, String)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiagnosticOutput {
    pub message: String,
    pub exception: Option<String>,
    pub stack: Option<String>,
    pub leaves: Vec<String>,
    pub changed: bool,
}

pub struct Processor {
    redactor: SecretRedactor,
}

impl Processor {
    pub fn new(minimum_custom_key_length: usize) -> Self {
        Self {
            redactor: SecretRedactor::new(minimum_custom_key_length),
        }
    }

    pub fn redact_text(&self, text: &str) -> Result<String> {
        self.redactor.try_redact(text)
    }

    pub fn redact_structured_text(&self, key: Option<&str>, text: &str) -> Result<String> {
        self.redactor.try_redact_structured(key, text)
    }

    pub fn redact_client_message(&self, text: &str) -> Result<String> {
        self.redactor.try_redact_internal(text)
    }

    pub fn process_diagnostic(
        &self,
        input: &DiagnosticInput,
        policy: Policy,
    ) -> Result<DiagnosticOutput> {
        let message = self.process_text(&input.message, policy)?;
        let exception = input
            .exception
            .as_deref()
            .map(|text| self.process_text(text, policy))
            .transpose()?;
        let stack = input
            .stack
            .as_deref()
            .map(|text| {
                if policy.redact {
                    self.redact_text(text)
                } else {
                    Ok(text.to_owned())
                }
            })
            .transpose()?;
        let leaves = input
            .leaves
            .iter()
            .map(|(key, text)| {
                if policy.redact {
                    self.redact_structured_text(key.as_deref(), text)
                } else {
                    Ok(text.clone())
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let changed = message != input.message
            || exception != input.exception
            || stack != input.stack
            || leaves
                .iter()
                .zip(&input.leaves)
                .any(|(processed, (_, original))| processed != original);
        Ok(DiagnosticOutput {
            message,
            exception,
            stack,
            leaves,
            changed,
        })
    }

    pub fn scrub_access_arguments(&self, arguments: &[String]) -> Result<Vec<String>> {
        arguments
            .iter()
            .map(|argument| self.scrub_access_arg(argument))
            .collect()
    }

    fn process_text(&self, text: &str, policy: Policy) -> Result<String> {
        let collapsed = if policy.base64_limit > 0 {
            collapse_base64(text, policy.base64_limit as usize)
        } else {
            text.to_owned()
        };
        let redacted = if policy.redact {
            self.redact_text(&collapsed)?
        } else {
            collapsed
        };
        Ok(
            if policy.text_limit > 0 && redacted.chars().count() > policy.text_limit as usize {
                truncate_text(&redacted, policy.text_limit as usize)
            } else {
                redacted
            },
        )
    }

    fn scrub_access_arg(&self, value: &str) -> Result<String> {
        let length = value.chars().count();
        let scanned = if length <= 512 {
            value
        } else {
            let head = &value[..char_offset(value, 512)];
            if head.contains('?') {
                &head[..head.rfind(['?', '&']).unwrap_or(0)]
            } else {
                head
            }
        };
        let scrubbed = self.redact_text(scanned)?;
        let (path, query) = scrubbed
            .split_once('?')
            .map_or((scrubbed.as_str(), None), |(path, query)| {
                (path, Some(query))
            });
        let safe = if self.hides_encoded_credential(path)? {
            REDACTED.to_owned()
        } else if query.is_some() && self.hides_encoded_credential(&scrubbed)? {
            format!("{path}?{REDACTED}")
        } else {
            scrubbed
        };
        Ok(if length > 512 {
            format!(
                "{safe}... ({} more chars truncated) ...",
                length - scanned.chars().count()
            )
        } else {
            safe
        })
    }

    fn hides_encoded_credential(&self, value: &str) -> Result<bool> {
        if !value.as_bytes().contains(&b'%') {
            return Ok(false);
        }
        let decoded = percent_decode_str(value).decode_utf8_lossy();
        Ok(self.redact_text(&decoded)? != decoded)
    }
}

fn char_offset(text: &str, count: usize) -> usize {
    text.char_indices()
        .nth(count)
        .map_or(text.len(), |(index, _)| index)
}

fn marker(skipped_chars: usize) -> String {
    format!(
        "... (litellm_truncated skipped {skipped_chars} chars. Truncation is a stdout logging safeguard. Full, untruncated data is logged to logging callbacks (OTEL, Datadog, etc.) and at DEBUG level. To increase the truncation limit, set `MAX_STRING_LENGTH_STDOUT_LOG` in your env.) ..."
    )
}

fn truncate_text(text: &str, limit: usize) -> String {
    let length = text.chars().count();
    let kept = limit.saturating_sub(marker(length).len());
    if kept == 0 {
        return text[..char_offset(text, limit)].to_owned();
    }
    let head = kept / 2;
    let tail = kept - head;
    format!(
        "{}{}{}",
        &text[..char_offset(text, head)],
        marker(length - kept),
        &text[char_offset(text, length - tail)..]
    )
}

fn base64_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'+' || byte == b'/'
}

fn looks_like_base64(run: &str) -> bool {
    let unpadded = run.trim_end_matches('=');
    let lower_hex = unpadded
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    let upper_hex = unpadded
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'A'..=b'F').contains(&byte));
    let repeated = unpadded.bytes().all(|byte| byte == unpadded.as_bytes()[0]);
    (!lower_hex && !upper_hex) || repeated
}

fn base64_size(chars: usize) -> String {
    let bytes = chars as f64 * 3.0 / 4.0;
    if bytes >= 1024.0 * 1024.0 {
        return format!("{:.2}MB", bytes / (1024.0 * 1024.0));
    }
    if bytes >= 1024.0 {
        return format!("{:.1}KB", bytes / 1024.0);
    }
    format!("{}B", bytes as usize)
}

fn collapse_base64(text: &str, limit: usize) -> String {
    let bytes = text.as_bytes();
    let mut position = 0;
    let mut previous = 0;
    let mut output = String::new();
    while position < bytes.len() {
        if !base64_byte(bytes[position]) || (position > 0 && base64_byte(bytes[position - 1])) {
            position += 1;
            continue;
        }
        let start = position;
        while position < bytes.len() && base64_byte(bytes[position]) {
            position += 1;
        }
        let run_end = position;
        while position < bytes.len() && position - run_end < 2 && bytes[position] == b'=' {
            position += 1;
        }
        let run = &text[start..position];
        if run_end - start > limit && looks_like_base64(run) {
            output.push_str(&text[previous..start]);
            output.push_str(&format!(
                "[base64_data truncated: {}]",
                base64_size(run.len())
            ));
            previous = position;
        }
    }
    output.push_str(&text[previous..]);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redaction_precedes_the_text_bound_and_preserves_unicode_character_limits() {
        let processor = Processor::new(16);
        let secret = format!("sk-{}", "q".repeat(48));
        let text = format!("{}{}{}", "é".repeat(110), secret, "界".repeat(1000));
        let input = DiagnosticInput {
            message: text,
            exception: None,
            stack: None,
            leaves: vec![],
        };
        let output = processor
            .process_diagnostic(
                &input,
                Policy {
                    redact: true,
                    base64_limit: 0,
                    text_limit: 500,
                },
            )
            .unwrap();
        assert!(output.message.chars().count() <= 500);
        assert!(!output.message.contains("sk-qq"));
        assert!(output.changed);
    }

    #[test]
    fn base64_collapse_applies_to_debug_and_exceptions_without_touching_hex() {
        let processor = Processor::new(16);
        let input = DiagnosticInput {
            message: format!("image={} digest={}", "Q".repeat(100), "a1".repeat(50)),
            exception: Some(format!("upload failed: {}", "Q".repeat(100))),
            stack: Some("api_key=secret123".to_owned()),
            leaves: vec![(Some("api_key".to_owned()), "secret123".to_owned())],
        };
        let output = processor
            .process_diagnostic(
                &input,
                Policy {
                    redact: true,
                    base64_limit: 20,
                    text_limit: 0,
                },
            )
            .unwrap();
        assert!(output.message.contains("[base64_data truncated: 75B]"));
        assert!(output.message.contains(&"a1".repeat(50)));
        assert!(
            output
                .exception
                .unwrap()
                .contains("[base64_data truncated: 75B]")
        );
        assert_eq!(output.stack.as_deref(), Some(REDACTED));
        assert_eq!(output.leaves, vec![REDACTED]);
    }

    #[test]
    fn access_arguments_keep_encoded_paths_and_drop_decoded_credentials() {
        let processor = Processor::new(16);
        let arguments = vec![
            "/v1/models?filter=gpt%2D4o&page=2".to_owned(),
            "/v1/models?k%65y=sk%2Dabcdefghijklmnopqrstuvwxyz&page=2".to_owned(),
        ];
        assert_eq!(
            processor.scrub_access_arguments(&arguments).unwrap(),
            vec![arguments[0].clone(), "/v1/models?REDACTED".to_owned()]
        );
    }
}
