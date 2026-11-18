//! Error text scrubbing: prompt bodies echoed in provider errors, client IP addresses, and the raw
//! model name a request asked for when the row stores a placeholder model.

use std::sync::LazyLock;

use fancy_regex::{Captures, Regex as FancyRegex};
use litellm_python_compat::{Value, json, repr};
use regex::Regex;

use crate::privacy::{REDACTED, TRUNCATED_FIELD, TRUNCATION_NOTE};
use crate::py::{char_len, slice};

pub(crate) const UNKNOWN_MODEL: &str = "unknown-model";
const IP_ADDRESS_CHARACTERS: &str = "0123456789abcdefABCDEF.:";
const MAX_IP_ADDRESS_LENGTH: usize = 45;

static PROMPT_LEAK_KEY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?:['"](?:input|messages|prompt)['"]\s*:\s*|\b(?:input_value)\s*=\s*)"#)
        .expect("prompt leak pattern compiles")
});
static TRUNCATION_MARKER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"\.\.\. \({} skipped \d+ chars\. {}\) \.\.\.",
        regex::escape(TRUNCATED_FIELD),
        regex::escape(TRUNCATION_NOTE)
    ))
    .expect("truncation marker compiles")
});
static LONG_HEX_GROUP: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[0-9A-Fa-f]{5,}").expect("hex group pattern compiles"));
static IPV4_ADDRESS: LazyLock<FancyRegex> = LazyLock::new(|| {
    FancyRegex::new(r"(?<![0-9A-Za-z])(?:\d{1,3}\.){3}\d{1,3}(?![0-9A-Za-z])")
        .expect("ipv4 pattern compiles")
});
static IPV6_ADDRESS_CANDIDATE: LazyLock<FancyRegex> = LazyLock::new(|| {
    FancyRegex::new(
        r"(?:(?<![0-9A-Za-z:])|(?<=\w:))(?:[0-9A-Fa-f]{0,4}:){2,9}[0-9A-Fa-f]{0,4}(?![0-9A-Za-z:])",
    )
    .expect("ipv6 pattern compiles")
});

fn char_at(text: &str, at: usize) -> Option<char> {
    text.get(at..).and_then(|rest| rest.chars().next())
}

fn after_char(text: &str, at: usize) -> usize {
    at + char_at(text, at).map_or(1, char::len_utf8)
}

fn scan_quoted_string_end(text: &str, start: usize, quote: char) -> Option<usize> {
    let mut index = after_char(text, start);
    while index < text.len() {
        let current = char_at(text, index)?;
        if current == '\\' {
            index = after_char(text, after_char(text, index));
            continue;
        }
        if current == quote {
            return Some(after_char(text, index));
        }
        index = after_char(text, index);
    }
    None
}

fn scan_balanced_value_end(text: &str, start: usize) -> Option<usize> {
    let first = char_at(text, start)?;
    let close = match first {
        '\'' | '"' => return scan_quoted_string_end(text, start, first),
        '[' => ']',
        '{' => '}',
        _ => return None,
    };
    let mut depth = 0usize;
    let mut index = start;
    while index < text.len() {
        let current = char_at(text, index)?;
        if current == '\'' || current == '"' {
            index = scan_quoted_string_end(text, index, current)?;
            continue;
        }
        if current == first {
            depth += 1;
        } else if current == close {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return Some(after_char(text, index));
            }
        }
        index = after_char(text, index);
    }
    None
}

pub(crate) fn redact_prompt_leaks_in_error_string(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let redaction = format!("\"{REDACTED}\"");
    let mut out = String::new();
    let mut position = 0;
    while position < text.len() {
        let Some(found) = PROMPT_LEAK_KEY.find_at(text, position) else {
            out.push_str(&text[position..]);
            break;
        };
        out.push_str(&text[position..found.end()]);
        let value_start = found.end();
        if value_start >= text.len() {
            break;
        }
        match char_at(text, value_start) {
            Some('[' | '{' | '\'' | '"') => match scan_balanced_value_end(text, value_start) {
                Some(value_end) => {
                    out.push_str(&redaction);
                    position = value_end;
                }
                None => {
                    out.push_str(&redaction);
                    break;
                }
            },
            _ => position = value_start,
        }
    }
    out
}

fn ipv6_address(text: &str) -> bool {
    !text.is_empty() && text.parse::<std::net::Ipv6Addr>().is_ok()
}

fn rpartition_head(text: &str) -> &str {
    text.rsplit_once(':').map_or("", |(head, _)| head)
}

fn masked_ipv6_address(captures: &Captures<'_, str>) -> String {
    let candidate = captures.get(0).map_or("", |found| found.as_str());
    let unlabelled = candidate.split_once(':').map_or("", |(_, rest)| rest);
    let addresses = [
        candidate,
        rpartition_head(candidate),
        unlabelled,
        rpartition_head(unlabelled),
    ];
    if addresses.iter().any(|address| ipv6_address(address)) {
        REDACTED.to_owned()
    } else {
        candidate.to_owned()
    }
}

fn without_ip_addresses(text: &str) -> String {
    let without_ipv4 = IPV4_ADDRESS.replace_all(text, REDACTED);
    IPV6_ADDRESS_CANDIDATE
        .replace_all(&without_ipv4, |captures: &Captures<'_, str>| {
            masked_ipv6_address(captures)
        })
        .into_owned()
}

fn is_ip_address_piece(piece: &str) -> bool {
    piece
        .chars()
        .any(|character| character.is_ascii_digit() || character == ':')
}

fn is_ip_character(character: char) -> bool {
    IP_ADDRESS_CHARACTERS.contains(character)
}

fn without_ip_piece_before_cut(text: &str) -> String {
    let run = &text[text.trim_end_matches(is_ip_character).len()..];
    let last_long = LONG_HEX_GROUP
        .find_iter(run)
        .last()
        .map_or(0, |found| found.end());
    let after_long = &run[last_long..];
    let bounded = &after_long[after_long.len().saturating_sub(MAX_IP_ADDRESS_LENGTH)..];
    let piece = bounded.trim_start_matches(['.', ':']);
    if is_ip_address_piece(piece) {
        format!("{}{REDACTED}", &text[..text.len() - piece.len()])
    } else {
        text.to_owned()
    }
}

fn without_ip_piece_after_cut(text: &str) -> String {
    let run = &text[..text.len() - text.trim_start_matches(is_ip_character).len()];
    let first_long = LONG_HEX_GROUP
        .find(run)
        .map_or(run.len(), |found| found.start());
    let before_long = &run[..first_long];
    let bounded = &before_long[..before_long.len().min(MAX_IP_ADDRESS_LENGTH)];
    let piece = bounded.trim_end_matches(['.', ':']);
    if is_ip_address_piece(piece) {
        format!("{REDACTED}{}", &text[piece.len()..])
    } else {
        text.to_owned()
    }
}

pub(crate) fn masked_ip_text(text: &str) -> String {
    let masked = without_ip_addresses(text);
    let mut parts: Vec<&str> = Vec::new();
    let mut last = 0;
    for marker in TRUNCATION_MARKER.find_iter(&masked) {
        parts.push(&masked[last..marker.start()]);
        parts.push(marker.as_str());
        last = marker.end();
    }
    parts.push(&masked[last..]);
    let final_index = parts.len() - 1;
    parts
        .iter()
        .enumerate()
        .map(|(index, part)| {
            if index % 2 == 1 {
                return (*part).to_owned();
            }
            let after = if index > 0 {
                without_ip_piece_after_cut(part)
            } else {
                (*part).to_owned()
            };
            if index < final_index {
                without_ip_piece_before_cut(&after)
            } else {
                after
            }
        })
        .collect()
}

pub(crate) fn raw_model_spellings(raw_model: &str) -> Vec<String> {
    let model = Value::Str(raw_model.to_owned());
    let repr_text = repr::repr(&model);
    let json_text = json::dumps(&model).unwrap_or_default();
    let mut spellings: Vec<String> = Vec::new();
    for spelling in [
        raw_model.to_owned(),
        slice(&repr_text, Some(1), Some(-1)),
        slice(&json_text, Some(1), Some(-1)),
    ] {
        if !spellings.contains(&spelling) {
            spellings.push(spelling);
        }
    }
    spellings
}

fn overlap_at_end(text: &str, spelling: &str) -> usize {
    let spelling_chars: Vec<char> = spelling.chars().collect();
    let longest = char_len(text).min(spelling_chars.len().saturating_sub(1));
    (1..=longest)
        .rev()
        .find(|length| text.ends_with(&spelling_chars[..*length].iter().collect::<String>()))
        .unwrap_or(0)
}

fn overlap_at_start(text: &str, spelling: &str) -> usize {
    let spelling_chars: Vec<char> = spelling.chars().collect();
    let longest = char_len(text).min(spelling_chars.len().saturating_sub(1));
    (1..=longest)
        .rev()
        .find(|length| {
            text.starts_with(
                &spelling_chars[spelling_chars.len() - length..]
                    .iter()
                    .collect::<String>(),
            )
        })
        .unwrap_or(0)
}

fn scrub_raw_model_split_by_truncation(text: &str, spellings: &[String]) -> String {
    let Some(marker) = TRUNCATION_MARKER.find(text) else {
        return text.to_owned();
    };
    let head = &text[..marker.start()];
    let tail = &text[marker.end()..];
    let head_cut = spellings
        .iter()
        .map(|spelling| overlap_at_end(head, spelling))
        .max()
        .unwrap_or(0);
    let tail_cut = spellings
        .iter()
        .map(|spelling| overlap_at_start(tail, spelling))
        .max()
        .unwrap_or(0);
    let head_length = char_len(head) as i64;
    format!(
        "{}{}{}{}{}",
        slice(head, None, Some(head_length - head_cut as i64)),
        if head_cut > 0 { UNKNOWN_MODEL } else { "" },
        marker.as_str(),
        if tail_cut > 0 { UNKNOWN_MODEL } else { "" },
        slice(tail, Some(tail_cut as i64), None),
    )
}

pub(crate) fn scrub_raw_model_from_error_text(text: &str, spellings: &[String]) -> String {
    let whole_occurrences_scrubbed = spellings
        .iter()
        .fold(text.to_owned(), |scrubbed, spelling| {
            scrubbed.replace(spelling.as_str(), UNKNOWN_MODEL)
        });
    scrub_raw_model_split_by_truncation(&whole_occurrences_scrubbed, spellings)
}

#[cfg(test)]
mod tests {
    use litellm_python_compat::Value;
    use rstest::rstest;

    use super::{
        masked_ip_text, redact_prompt_leaks_in_error_string, scrub_raw_model_from_error_text,
    };
    use crate::test_support::{argument, case, field, helper_records};

    fn text(value: &Value) -> &str {
        match value {
            Value::Str(text) => text,
            other => panic!("{other:?} is not text"),
        }
    }

    fn spellings(value: &Value) -> Vec<String> {
        match value {
            Value::List(items) => items.iter().map(|item| text(item).to_owned()).collect(),
            other => panic!("{other:?} is not a list"),
        }
    }

    // Each file holds the calls `scripts/record_goldens.py` recorded from the Python helper
    #[rstest]
    #[case::prompt_leaks(
        include_str!("../tests/goldens/helpers/redact_prompt_leaks_in_error_string.jsonl"),
        |record: &Value| redact_prompt_leaks_in_error_string(text(argument(record, 0)))
    )]
    #[case::masked_ip_text(
        include_str!("../tests/goldens/helpers/masked_ip_text.jsonl"),
        |record: &Value| masked_ip_text(text(argument(record, 0)))
    )]
    #[case::raw_model_scrubbing(
        include_str!("../tests/goldens/helpers/scrub_raw_model_from_error_text.jsonl"),
        |record: &Value| scrub_raw_model_from_error_text(
            text(argument(record, 0)),
            &spellings(argument(record, 1)),
        )
    )]
    fn matches_every_recorded_python_call(
        #[case] goldens: &str,
        #[case] port: fn(&Value) -> String,
    ) {
        for record in helper_records(goldens) {
            assert_eq!(
                port(&record),
                text(field(&record, "output")),
                "{}",
                case(&record)
            );
        }
    }
}
