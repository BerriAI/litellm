//! `yaml.resolver.Resolver`'s implicit resolvers that give a config value another type, checked in
//! PyYAML's registration order. Each matcher is the resolver's regular expression written out by
//! hand. Timestamps and `=` are left as text, see `AGENTS.md`.

use crate::compose::Tag;

/// The tag PyYAML resolves a plain scalar's text to.
pub(crate) fn implicit(text: &str) -> Tag {
    match text {
        _ if is_bool(text) => Tag::Bool,
        _ if is_float(text) => Tag::Float,
        _ if is_int(text) => Tag::Int,
        "<<" => Tag::Merge,
        "~" | "null" | "Null" | "NULL" | "" => Tag::Null,
        _ => Tag::Str,
    }
}

fn is_bool(text: &str) -> bool {
    matches!(
        text,
        "yes"
            | "Yes"
            | "YES"
            | "no"
            | "No"
            | "NO"
            | "true"
            | "True"
            | "TRUE"
            | "false"
            | "False"
            | "FALSE"
            | "on"
            | "On"
            | "ON"
            | "off"
            | "Off"
            | "OFF"
    )
}

/// `[-+]?[0-9][0-9_]*\.[0-9_]*(?:[eE][-+][0-9]+)?`, `\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?`,
/// `[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*`, `[-+]?\.(?:inf|Inf|INF)` or `\.(?:nan|NaN|NAN)`.
fn is_float(text: &str) -> bool {
    if matches!(text, ".nan" | ".NaN" | ".NAN") {
        return true;
    }
    if text.starts_with('.') && digit_run(&text[1..]) {
        return true;
    }
    let unsigned = unsigned(text);
    if matches!(unsigned, ".inf" | ".Inf" | ".INF") {
        return true;
    }
    let Some(rest) = leading_number(unsigned) else {
        return false;
    };
    match rest.strip_prefix('.') {
        Some(fraction) => exponent_tail(fraction),
        None => sexagesimal(rest)
            .and_then(|rest| rest.strip_prefix('.'))
            .is_some_and(|fraction| fraction.bytes().all(digit_or_underscore)),
    }
}

/// `[-+]?0b[0-1_]+`, `[-+]?0[0-7_]+`, `[-+]?(?:0|[1-9][0-9_]*)`, `[-+]?0x[0-9a-fA-F_]+` or
/// `[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+`.
fn is_int(text: &str) -> bool {
    let unsigned = unsigned(text);
    if let Some(digits) = unsigned.strip_prefix("0b") {
        return !digits.is_empty()
            && digits
                .bytes()
                .all(|byte| matches!(byte, b'0' | b'1' | b'_'));
    }
    if let Some(digits) = unsigned.strip_prefix("0x") {
        return !digits.is_empty()
            && digits
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() || byte == b'_');
    }
    if let Some(digits) = unsigned.strip_prefix('0') {
        return digits
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'7' | b'_'));
    }
    if !unsigned.starts_with(|first: char| first.is_ascii_digit()) {
        return false;
    }
    let Some(rest) = leading_number(unsigned) else {
        return false;
    };
    rest.is_empty() || sexagesimal(rest).is_some_and(str::is_empty)
}

fn unsigned(text: &str) -> &str {
    text.strip_prefix(['-', '+']).unwrap_or(text)
}

fn digit_or_underscore(byte: u8) -> bool {
    byte.is_ascii_digit() || byte == b'_'
}

/// What follows `[0-9][0-9_]*`, when the text starts with a digit.
fn leading_number(text: &str) -> Option<&str> {
    if !text.starts_with(|first: char| first.is_ascii_digit()) {
        return None;
    }
    Some(text.trim_start_matches(|character: char| character.is_ascii_digit() || character == '_'))
}

/// What follows one or more `:[0-5]?[0-9]` groups, when the text starts with one.
fn sexagesimal(text: &str) -> Option<&str> {
    let mut rest = text;
    let mut groups = 0;
    while let Some(after_colon) = rest.strip_prefix(':') {
        let run = after_colon.bytes().take_while(u8::is_ascii_digit).count();
        let valid = match run {
            1 => true,
            2 => matches!(after_colon.as_bytes()[0], b'0'..=b'5'),
            _ => false,
        };
        if !valid {
            return None;
        }
        rest = &after_colon[run..];
        groups += 1;
    }
    (groups > 0).then_some(rest)
}

/// `[0-9][0-9_]*` followed by an optional exponent, the end of a float that starts with `.`.
fn digit_run(text: &str) -> bool {
    text.starts_with(|first: char| first.is_ascii_digit()) && exponent_tail(text)
}

/// `[0-9_]*(?:[eE][-+][0-9]+)?` to the end of the text.
fn exponent_tail(text: &str) -> bool {
    let rest =
        text.trim_start_matches(|character: char| character.is_ascii_digit() || character == '_');
    match rest.strip_prefix(['e', 'E']) {
        None => rest.is_empty(),
        Some(exponent) => exponent.strip_prefix(['-', '+']).is_some_and(|power| {
            !power.is_empty() && power.bytes().all(|byte| byte.is_ascii_digit())
        }),
    }
}
