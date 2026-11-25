use aho_corasick::AhoCorasick;
use regex::Regex;

const PREFIXES: &[&str] = &[
    "-----BEGIN",
    "eyJ",
    "ya29.",
    "AKIA",
    "ASIA",
    "sk-",
    "sk_live_",
    "sk_test_",
    "rk_live_",
    "rk_test_",
    "AIza",
    "dapi",
    "ghp_",
    "gho_",
    "ghu_",
    "ghs_",
    "ghr_",
    "github_pat_",
    "glpat-",
    "lin_api_",
    "xoxa-",
    "xoxb-",
    "xoxp-",
    "xoxo-",
    "xoxs-",
    "xoxr-",
    "hf_",
    "npm_",
    "pypi-AgEIcHlwaS5vcmc",
    "SG.",
];

pub struct TokenRedactor {
    prefixes: AhoCorasick,
    tokens: Regex,
    word_ranges: Vec<(u32, u32)>,
}

impl TokenRedactor {
    pub fn new(minimum_custom_key_length: usize, word_ranges: Vec<(u32, u32)>) -> Self {
        let sk_length = minimum_custom_key_length.saturating_sub(3);
        let patterns = [
            r"ya29\.[A-Za-z0-9_.~+/-]{20,}",
            r"(?:AKIA|ASIA)[0-9A-Z]{16}",
            &format!(r"sk-[A-Za-z0-9_-]{{{sk_length},}}"),
            r"(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{24,}",
            r"AIza[0-9A-Za-z_-]{35}",
            r"dapi[0-9a-f]{32}",
            r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}",
            r"github_pat_[A-Za-z0-9_]{82}",
            r"glpat-[A-Za-z0-9_-]{20,}",
            r"lin_api_[A-Za-z0-9]{40}",
            r"xox[abposr]-[A-Za-z0-9-]{10,}",
            r"hf_[A-Za-z0-9]{34,}",
            r"npm_[A-Za-z0-9]{36}",
            r"pypi-AgEIcHlwaS5vcmc[A-Za-z0-9_-]{50,}",
            r"SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}",
        ];
        Self {
            prefixes: AhoCorasick::new(PREFIXES).expect("token prefixes compile"),
            tokens: Regex::new(&format!(r"\A(?:{})", patterns.join("|")))
                .expect("token patterns compile"),
            word_ranges,
        }
    }

    pub fn redact(&self, text: &str, replacement: &str) -> Option<String> {
        let mut output = None;
        let mut copied_until = 0;
        let mut failed_jwt_until = 0;
        let mut failed_jwt_scope_end = 0;
        for found in self.prefixes.find_overlapping_iter(text) {
            let start = found.start();
            if start < copied_until {
                continue;
            }
            let prefix = PREFIXES[found.pattern().as_usize()];
            let end = if prefix == "-----BEGIN" {
                pem_end(text, found.end())
            } else if !self.starts_token(text, start) {
                None
            } else if prefix == "eyJ" {
                if start < failed_jwt_until {
                    continue;
                }
                let header_end = jwt_run_end(text, found.end());
                let end = jwt_end(text, found.end(), header_end);
                if end.is_none() {
                    // Every later JWT start in this header shares the same invalid suffix.
                    failed_jwt_until = header_end;
                    failed_jwt_scope_end = self.failed_jwt_scope_end(text, start, header_end);
                }
                end
            } else {
                let scope = if start < failed_jwt_scope_end {
                    &text[..failed_jwt_scope_end]
                } else {
                    text
                };
                self.tokens
                    .find(&scope[start..])
                    .map(|token| start + token.end())
                    .filter(|&end| self.ends_token(scope, end, prefix))
            };
            if let Some(end) = end {
                let result = output.get_or_insert_with(String::new);
                result.push_str(&text[copied_until..start]);
                result.push_str(replacement);
                copied_until = end;
            }
        }
        if let Some(result) = &mut output {
            result.push_str(&text[copied_until..]);
        }
        output
    }

    fn is_word(&self, character: char) -> bool {
        if character.is_ascii() {
            return character.is_ascii_alphanumeric() || character == '_';
        }
        let code = u32::from(character);
        let index = self.word_ranges.partition_point(|&(_, end)| end < code);
        self.word_ranges
            .get(index)
            .is_some_and(|&(start, _)| start <= code)
    }

    fn failed_jwt_scope_end(&self, text: &str, start: usize, header_end: usize) -> usize {
        if let Some(marker) = text[start..header_end].find("-----BEGIN") {
            return start + marker;
        }
        let dotted_end = ["ya29", "SG"]
            .into_iter()
            .find_map(|prefix| {
                let position = header_end.checked_sub(prefix.len())?;
                if position < start
                    || !text[position..].starts_with(prefix)
                    || !self.starts_token(text, position)
                {
                    return None;
                }
                let token = self.tokens.find(&text[position..])?;
                let end = position + token.end();
                self.ends_token(text, end, if prefix == "SG" { "SG." } else { "ya29." })
                    .then_some(end)
            })
            .unwrap_or(header_end);
        dotted_end
            + text[dotted_end..]
                .chars()
                .next()
                .filter(|c| !c.is_ascii() || !jwt_byte(*c as u8))
                .map_or(0, char::len_utf8)
    }

    fn starts_token(&self, text: &str, start: usize) -> bool {
        let before = &text[..start];
        if before.chars().next_back().is_none_or(|c| !self.is_word(c)) {
            return true;
        }
        let bytes = before.as_bytes();
        bytes.ends_with(b"\\n")
            || bytes.ends_with(b"\\r")
            || bytes.ends_with(b"\\t")
            || (bytes.len() >= 6
                && bytes[bytes.len() - 6..].starts_with(b"\\u")
                && bytes[bytes.len() - 4..].iter().all(u8::is_ascii_hexdigit))
    }

    fn ends_token(&self, text: &str, end: usize, prefix: &str) -> bool {
        match prefix {
            "AKIA" | "ASIA" | "dapi" | "lin_api_" | "npm_" => {
                text[end..].chars().next().is_none_or(|c| !self.is_word(c))
            }
            "SG." => text.as_bytes().get(end).is_none_or(|&b| !jwt_byte(b)),
            _ => true,
        }
    }
}

fn jwt_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-'
}

fn jwt_run_end(text: &str, start: usize) -> usize {
    start
        + text.as_bytes()[start..]
            .iter()
            .take_while(|&&byte| jwt_byte(byte))
            .count()
}

fn jwt_end(text: &str, header_start: usize, header_end: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if header_end - header_start < 10 || bytes.get(header_end) != Some(&b'.') {
        return None;
    }
    let body_start = header_end + 1;
    let body_end = jwt_run_end(text, body_start);
    if body_start == body_end || bytes.get(body_end) != Some(&b'.') {
        return None;
    }
    let signature_start = body_end + 1;
    let signature_end = jwt_run_end(text, signature_start);
    (signature_end > signature_start).then_some(signature_end)
}

fn pem_label_end(text: &str, start: usize) -> Option<usize> {
    let marker = start + text[start..].find("-----")?;
    let label = &text[start..marker];
    (label.ends_with("PRIVATE KEY")
        && label
            .bytes()
            .all(|b| b.is_ascii_uppercase() || b == b' ' || b == b'-'))
    .then_some(marker + 5)
}

fn pem_end(text: &str, label_start: usize) -> Option<usize> {
    let body_start = pem_label_end(text, label_start)?;
    let marker = body_start + text[body_start..].find("-----")?;
    text[marker..]
        .strip_prefix("-----END")
        .and_then(|_| pem_label_end(text, marker + 8))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::{fixture, rstest};

    #[fixture]
    fn redactor() -> TokenRedactor {
        TokenRedactor::new(
            16,
            vec![
                (u32::from('²'), u32::from('²')),
                (u32::from('é'), u32::from('é')),
            ],
        )
    }

    #[rstest]
    #[case::jwt("eyJaaaaaaaaaaaa.x.signature", Some("hidden"))]
    #[case::failed_jwt("eyJaaaaaaaaaaaa..signature", None)]
    #[case::unicode_word("éghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", None)]
    #[case::unicode_number("²ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", None)]
    #[case::combining_mark(
        "\u{301}ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        Some("\u{301}hidden")
    )]
    #[case::escape(
        r"\u2022ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        Some(r"\u2022hidden")
    )]
    #[case::pem(
        "-----BEGIN RSA PRIVATE KEY-----\nbody\n-----END PRIVATE KEY-----",
        Some("hidden")
    )]
    #[case::nested_pem(
        "-----BEGIN PRIVATE KEY-----\n-----BEGIN PRIVATE KEY-----\n-----END PRIVATE KEY-----",
        Some("-----BEGIN PRIVATE KEY-----\nhidden")
    )]
    #[case::short_key_before_pem(
        "eyJa-sk-short-----BEGIN PRIVATE KEY-----x-----END PRIVATE KEY-----",
        Some("eyJa-sk-shorthidden")
    )]
    fn preserves_token_boundaries(
        redactor: TokenRedactor,
        #[case] input: &str,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(redactor.redact(input, "hidden").as_deref(), expected);
    }

    #[rstest]
    #[case::missing_body("..signature")]
    #[case::missing_signature(".body.")]
    #[case::no_dots("")]
    fn failed_jwt_runs_do_not_hide_inner_keys(redactor: TokenRedactor, #[case] suffix: &str) {
        let run = "eyJaaaaaaaaaa-AKIAAAAAAAAAAAAAAAAA-".repeat(24_000);
        let expected = "eyJaaaaaaaaaa-hidden-".repeat(24_000);
        assert_eq!(
            redactor.redact(&format!("{run}{suffix}"), "hidden"),
            Some(format!("{expected}{suffix}"))
        );
    }
}
