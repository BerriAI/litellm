//! Byte caps for stored span payloads. Message arrays stay valid JSON: they keep the first message,
//! an elision marker and the newest messages that fit.

use indexmap::IndexMap;
use serde::Serialize;
use serde_json::Value;

use crate::normalize::encode;

const MAX_JSON_ESCAPE_BYTES: usize = 6;
const MARKER_ROOM: usize = 48;

type Message = IndexMap<String, Value>;

pub fn truncate_value(value: String, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value;
    }
    let kept = prefix(&value, max_bytes);
    format!("{kept}…[truncated {} bytes]", value.len() - kept.len())
}

pub fn truncate_messages(value: String, max_bytes: usize) -> String {
    if value.len() <= max_bytes || !value.starts_with('[') {
        return truncate_value(value, max_bytes);
    }
    let messages = match serde_json::from_str::<Vec<Message>>(&value) {
        Ok(messages) if messages.len() >= 2 => messages,
        _ => return truncate_value(value, max_bytes),
    };
    let encoded: Vec<String> = messages.iter().map(encode).collect();
    let marker_bytes = elided(messages.len()).len();
    let fixed = 4 + encoded[0].len() + marker_bytes;
    let kept =
        newest_that_fit(&encoded[1..], max_bytes.saturating_sub(fixed)).min(messages.len() - 2);
    if kept > 0 {
        let marker = elided(messages.len() - 1 - kept);
        let tail = &encoded[encoded.len() - kept..];
        return array(
            std::iter::once(encoded[0].as_str())
                .chain([marker.as_str()])
                .chain(tail.iter().map(String::as_str)),
        );
    }
    let half = max_bytes.saturating_sub(marker_bytes + 4) / 2;
    let first = shrunk(&messages[0], half);
    let last = shrunk(&messages[messages.len() - 1], half);
    let middle = (messages.len() > 2).then(|| elided(messages.len() - 2));
    let shortened = array(
        std::iter::once(first.as_str())
            .chain(middle.as_deref())
            .chain([last.as_str()]),
    );
    if shortened.len() <= max_bytes {
        shortened
    } else {
        array([elided(messages.len()).as_str()])
    }
}

fn prefix(value: &str, max_bytes: usize) -> &str {
    let end = (0..=max_bytes.min(value.len()))
        .rev()
        .find(|index| value.is_char_boundary(*index))
        .unwrap_or_default();
    &value[..end]
}

fn array<'a>(parts: impl IntoIterator<Item = &'a str>) -> String {
    format!("[{}]", parts.into_iter().collect::<Vec<_>>().join(", "))
}

#[derive(Serialize)]
struct ElisionMarker {
    role: &'static str,
    content: String,
}

fn elided(count: usize) -> String {
    encode(&ElisionMarker {
        role: "system",
        content: format!("…[{count} earlier messages truncated]"),
    })
}

/// How many trailing messages fit in `budget` bytes, counting the `, ` separator before each.
fn newest_that_fit(encoded: &[String], budget: usize) -> usize {
    encoded
        .iter()
        .rev()
        .scan(0, |total, message| {
            *total += message.len() + 2;
            Some(*total)
        })
        .take_while(|total| *total <= budget)
        .count()
}

/// One message cut to `budget` bytes. Shortens `content` first; if other fields (e.g. huge
/// tool_calls) still don't fit, keeps only role and content.
fn shrunk(message: &Message, budget: usize) -> String {
    let text = match message.get("content") {
        Some(Value::String(text)) => text.clone(),
        content => encode(&content.unwrap_or(&Value::Null)),
    };
    let role_only = Message::from([(
        "role".to_owned(),
        message
            .get("role")
            .cloned()
            .unwrap_or_else(|| Value::from("user")),
    )]);
    let attempts = [
        cut(message, &text, budget, 1),
        cut(&role_only, &text, budget, 1),
        cut(&role_only, &text, budget, MAX_JSON_ESCAPE_BYTES),
    ];
    let fallback = attempts[2].clone();
    attempts
        .into_iter()
        .find(|attempt| attempt.len() <= budget)
        .unwrap_or(fallback)
}

fn cut(message: &Message, text: &str, budget: usize, escape_factor: usize) -> String {
    let overhead = with_content(message, String::new()).len();
    let room = budget.saturating_sub(overhead + MARKER_ROOM) / escape_factor;
    let kept = prefix(text, room);
    with_content(
        message,
        format!("{kept}…[truncated {} bytes]", text.len() - kept.len()),
    )
}

fn with_content(message: &Message, content: String) -> String {
    let mut replaced = message.clone();
    replaced.insert("content".to_owned(), Value::String(content));
    encode(&replaced)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    fn parsed(value: &str) -> Vec<Message> {
        serde_json::from_str(value).expect("truncated message arrays stay valid JSON")
    }

    #[rstest]
    #[case::fits("short", 10, "short")]
    #[case::ascii("abcdefghij", 4, "abcd…[truncated 6 bytes]")]
    #[case::splits_no_character("雪雪", 4, "雪…[truncated 3 bytes]")]
    fn values_keep_a_whole_character_prefix(
        #[case] value: &str,
        #[case] max_bytes: usize,
        #[case] expected: &str,
    ) {
        assert_eq!(truncate_value(value.to_owned(), max_bytes), expected);
    }

    #[rstest]
    fn long_history_drops_middle_messages_and_counts_them() {
        let history = (0..12).map(
            |turn| json!({"role": "user", "content": format!("turn {turn} {}", "x".repeat(60))}),
        );
        let messages: Vec<Value> =
            std::iter::once(json!({"role": "system", "content": "be brief"}))
                .chain(history)
                .collect();
        let original_count = messages.len();
        let output = truncate_messages(Value::Array(messages).to_string(), 400);
        let kept = parsed(&output);
        assert!(output.len() <= 400);
        assert_eq!(kept[0]["content"], "be brief");
        assert!(
            kept.last().unwrap()["content"]
                .as_str()
                .unwrap()
                .starts_with("turn 11 ")
        );
        let elided: usize = kept[1]["content"].as_str().unwrap()["…[".len()..]
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(elided + kept.len() - 1, original_count);
    }

    #[rstest]
    fn kept_messages_count_their_separators_against_the_limit() {
        let messages: Vec<Value> = std::iter::once(json!({"role": "system", "content": "s"}))
            .chain((0..50).map(|_| json!({"role": "user", "content": ""})))
            .collect();
        for max_bytes in 120..400 {
            let output = truncate_messages(Value::Array(messages.clone()).to_string(), max_bytes);
            assert!(output.len() <= max_bytes, "{max_bytes}: {output}");
            parsed(&output);
        }
    }

    #[rstest]
    #[case::huge_first(json!([{"role": "system", "content": "s".repeat(2000)}, {"role": "user", "content": "short question"}]))]
    #[case::two_messages(json!([{"role": "user", "content": "a".repeat(900)}, {"role": "assistant", "content": "b".repeat(900)}]))]
    #[case::huge_first_and_last(json!([{"role": "system", "content": "s".repeat(900)}, {"role": "user", "content": "middle"}, {"role": "user", "content": "q".repeat(900)}]))]
    fn oversized_messages_are_shortened_not_cut(#[case] messages: Value) {
        let output = truncate_messages(messages.to_string(), 400);
        let kept = parsed(&output);
        assert!(output.len() <= 400);
        assert_eq!(kept[0]["role"], messages[0]["role"]);
        assert_eq!(
            kept.last().unwrap()["role"],
            messages.as_array().unwrap().last().unwrap()["role"]
        );
        assert!(kept.iter().all(|message| message["content"].is_string()));
    }

    #[rstest]
    fn oversized_non_content_fields_fall_back_to_role_and_content() {
        let messages = json!([
            {"role": "assistant", "content": "x", "tool_calls": [{"name": "t", "args": {"blob": "z".repeat(3000)}}]},
            {"role": "user", "content": "—".repeat(900)},
        ]);
        let output = truncate_messages(messages.to_string(), 400);
        let kept = parsed(&output);
        assert!(output.len() <= 400);
        assert_eq!(
            kept.iter()
                .map(|message| message["role"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["assistant", "user"]
        );
        assert!(kept[0]["content"].as_str().unwrap().starts_with('x'));
        assert!(kept[1]["content"].as_str().unwrap().starts_with('—'));
    }

    #[rstest]
    #[case::object(r#"{"role": "user", "content": "long"}"#)]
    #[case::single_message(r#"[{"role": "user", "content": "long"}]"#)]
    #[case::not_messages("[1, 2, 3, 4, 5, 6, 7, 8]")]
    fn other_payloads_are_byte_truncated(#[case] value: &str) {
        assert_eq!(
            truncate_messages(value.to_owned(), 8),
            truncate_value(value.to_owned(), 8)
        );
    }
}
