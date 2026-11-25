//! The payload's `messages`: the request's messages with its top-level `system` prompt in front,
//! and every long base64 payload in them replaced by a note of its size, as
//! `get_standard_logging_object_payload` logs them
//! (`truncate_base64_in_messages(StandardLoggingPayloadSetup.append_system_prompt_messages(...))`
//! in litellm/litellm_core_utils/litellm_logging.py and logging_utils.py).

use std::sync::LazyLock;

use litellm_python_compat::{format::fixed, pystr};
use regex::{Captures, Regex};
use serde_json::{Map, Value, json};

/// `MAX_BASE64_LENGTH_FOR_LOGGING`'s default: the longest base64 payload logged whole.
pub const MAX_BASE64_LENGTH_FOR_LOGGING: usize = 64;

/// `TRUNCATED_BASE64_MARKER`, which opens the note a truncated payload is replaced by.
const TRUNCATED_BASE64_MARKER: &str = "[base64_data truncated: ";

/// `_MAX_TRUNCATION_DEPTH`: objects and lists nested deeper than this are logged as they are.
const MAX_TRUNCATION_DEPTH: usize = 20;

const BYTES_PER_KIB: f64 = 1024.0;
const BYTES_PER_MIB: f64 = 1024.0 * 1024.0;

/// `_DATA_URI_RE`: a data URI's media type and its base64 payload.
static DATA_URI: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"data:([^;,\s]{1,255});base64,([A-Za-z0-9+/=]+)")
        .expect("the data URI pattern compiles")
});

/// `StandardLoggingPayloadSetup.append_system_prompt_messages`: `messages` with the request's
/// top-level `system` prompt, text or a non-empty list of blocks, in front as a system message.
/// Messages that already open with that same system message are left as they are, a text prompt
/// becomes the user's message after it, and any other value that is not a list is left alone.
pub fn with_system_prompt(messages: Value, system: Option<&Value>) -> Value {
    let Some(system) = system.filter(|system| match system {
        Value::String(_) => true,
        Value::Array(blocks) => !blocks.is_empty(),
        _ => false,
    }) else {
        return messages;
    };
    let system_message = json!({"role": "system", "content": system});
    match messages {
        Value::Null => Value::Array(vec![system_message]),
        Value::String(text) => json!([system_message, {"role": "user", "content": text}]),
        Value::Array(mut listed) => {
            let first = listed.first();
            let opens_with_it = first.and_then(|message| message.get("role"))
                == Some(&Value::from("system"))
                && first.and_then(|message| message.get("content")) == Some(system);
            if !opens_with_it {
                listed.insert(0, system_message);
            }
            Value::Array(listed)
        }
        other => other,
    }
}

/// `truncate_base64_in_messages`: `messages` with each base64 data URI whose payload is longer
/// than `max_length` cut to its media type and a note of its size, and the `data` of an object
/// whose `type` is `base64` (an Anthropic base64 source) longer than that replaced by the note
/// alone. Objects and lists nested past `_MAX_TRUNCATION_DEPTH` are left whole. A `max_length` of
/// 0 turns truncation off, as Python's `MAX_BASE64_LENGTH_FOR_LOGGING <= 0` does.
pub fn truncate_base64(mut messages: Value, max_length: usize) -> Value {
    if max_length == 0 {
        return messages;
    }
    match &mut messages {
        Value::String(text) => truncate_data_uris(text, max_length),
        Value::Object(_) | Value::Array(_) => truncate_within(&mut messages, 0, max_length),
        _ => {}
    }
    messages
}

/// `_truncate_base64_in_value`'s walk over one object or list `depth` levels below the root.
fn truncate_within(container: &mut Value, depth: usize, max_length: usize) {
    if depth > MAX_TRUNCATION_DEPTH {
        return;
    }
    match container {
        Value::Object(fields) => truncate_fields(fields, depth, max_length),
        Value::Array(items) => {
            for item in items {
                match item {
                    Value::String(text) => truncate_data_uris(text, max_length),
                    Value::Object(_) | Value::Array(_) => {
                        truncate_within(item, depth + 1, max_length);
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

/// `_truncate_dict_string` for each text field of one object: an Anthropic base64 source's
/// `data` is truncated whole, any other text by the data URIs in it.
fn truncate_fields(fields: &mut Map<String, Value>, depth: usize, max_length: usize) {
    let base64_source = fields.get("type").and_then(Value::as_str) == Some("base64");
    for (key, field) in fields.iter_mut() {
        match field {
            Value::String(data) if base64_source && key == "data" => {
                if pystr::len(data) > max_length {
                    *data = format!(
                        "{TRUNCATED_BASE64_MARKER}{}]",
                        base64_size(pystr::len(data))
                    );
                }
            }
            Value::String(text) => truncate_data_uris(text, max_length),
            Value::Object(_) | Value::Array(_) => truncate_within(field, depth + 1, max_length),
            _ => {}
        }
    }
}

/// `_truncate_base64_in_string`: each data URI in `text` whose payload is longer than
/// `max_length` becomes `data:<media type>;base64,[base64_data truncated: <size>]`.
fn truncate_data_uris(text: &mut String, max_length: usize) {
    if !(text.contains(';') && text.contains("data:") && text.contains(";base64,")) {
        return;
    }
    let truncated = DATA_URI.replace_all(text, |uri: &Captures<'_>| {
        let payload = &uri[2];
        if payload.len() <= max_length {
            return uri[0].to_owned();
        }
        format!(
            "data:{};base64,{TRUNCATED_BASE64_MARKER}{}]",
            &uri[1],
            base64_size(payload.len())
        )
    });
    if let std::borrow::Cow::Owned(truncated) = truncated {
        *text = truncated;
    }
}

/// `format_base64_size` (litellm/_logging.py): the bytes `chars` base64 characters decode to, in
/// `MB` to two places, `KB` to one, or whole `B`.
fn base64_size(chars: usize) -> String {
    let bytes = (chars as f64) * 3.0 / 4.0;
    if bytes >= BYTES_PER_MIB {
        format!("{}MB", fixed(bytes / BYTES_PER_MIB, 2))
    } else if bytes >= BYTES_PER_KIB {
        format!("{}KB", fixed(bytes / BYTES_PER_KIB, 1))
    } else {
        format!("{}B", chars * 3 / 4)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    fn base64(chars: usize) -> String {
        "A".repeat(chars)
    }

    #[rstest]
    #[case::no_system(json!([{"role": "user", "content": "hi"}]), None, json!([{"role": "user", "content": "hi"}]))]
    #[case::text_system(
        json!([{"role": "user", "content": "hi"}]),
        Some(json!("be brief")),
        json!([{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}])
    )]
    #[case::empty_text_system(
        json!([{"role": "user", "content": "hi"}]),
        Some(json!("")),
        json!([{"role": "system", "content": ""}, {"role": "user", "content": "hi"}])
    )]
    #[case::block_system(
        json!([{"role": "user", "content": "hi"}]),
        Some(json!([{"type": "text", "text": "be brief"}])),
        json!([
            {"role": "system", "content": [{"type": "text", "text": "be brief"}]},
            {"role": "user", "content": "hi"}
        ])
    )]
    #[case::empty_block_system(
        json!([{"role": "user", "content": "hi"}]),
        Some(json!([])),
        json!([{"role": "user", "content": "hi"}])
    )]
    #[case::system_of_another_type(json!([]), Some(json!(1)), json!([]))]
    #[case::already_first(
        json!([{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]),
        Some(json!("be brief")),
        json!([{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}])
    )]
    #[case::a_different_system_first(
        json!([{"role": "system", "content": "other"}]),
        Some(json!("be brief")),
        json!([{"role": "system", "content": "be brief"}, {"role": "system", "content": "other"}])
    )]
    #[case::no_messages(Value::Null, Some(json!("be brief")), json!([{"role": "system", "content": "be brief"}]))]
    #[case::text_messages(
        json!("hi"),
        Some(json!("be brief")),
        json!([{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}])
    )]
    #[case::messages_of_another_type(json!({"role": "user"}), Some(json!("be brief")), json!({"role": "user"}))]
    fn the_system_prompt_is_put_in_front_as_python_does(
        #[case] messages: Value,
        #[case] system: Option<Value>,
        #[case] expected: Value,
    ) {
        assert_eq!(with_system_prompt(messages, system.as_ref()), expected);
    }

    #[rstest]
    #[case::short_payload(format!("data:image/png;base64,{}", base64(64)), format!("data:image/png;base64,{}", base64(64)))]
    #[case::long_payload(
        format!("see data:image/png;base64,{} here", base64(100)),
        "see data:image/png;base64,[base64_data truncated: 75B] here".to_owned()
    )]
    #[case::kilobytes(
        format!("data:image/png;base64,{}", base64(4000)),
        "data:image/png;base64,[base64_data truncated: 2.9KB]".to_owned()
    )]
    #[case::megabytes(
        format!("data:image/png;base64,{}", base64(2_000_000)),
        "data:image/png;base64,[base64_data truncated: 1.43MB]".to_owned()
    )]
    #[case::two_uris(
        format!("data:a/b;base64,{0} data:c/d;base64,{0}", base64(65)),
        "data:a/b;base64,[base64_data truncated: 48B] data:c/d;base64,[base64_data truncated: 48B]".to_owned()
    )]
    #[case::no_uri(base64(100), base64(100))]
    fn a_long_data_uri_is_cut_to_a_note_of_its_size(
        #[case] text: String,
        #[case] expected: String,
    ) {
        assert_eq!(
            truncate_base64(
                json!([{"role": "user", "content": text}]),
                MAX_BASE64_LENGTH_FOR_LOGGING
            ),
            json!([{"role": "user", "content": expected}])
        );
    }

    #[rstest]
    #[case::base64_source(
        json!({"type": "base64", "media_type": "image/png", "data": base64(100)}),
        json!({"type": "base64", "media_type": "image/png", "data": "[base64_data truncated: 75B]"})
    )]
    #[case::short_base64_source(
        json!({"type": "base64", "data": base64(64)}),
        json!({"type": "base64", "data": base64(64)})
    )]
    #[case::data_of_another_type(
        json!({"type": "text", "data": base64(100)}),
        json!({"type": "text", "data": base64(100)})
    )]
    fn an_anthropic_base64_sources_data_is_replaced_whole(
        #[case] source: Value,
        #[case] expected: Value,
    ) {
        let block =
            |source| json!([{"role": "user", "content": [{"type": "image", "source": source}]}]);
        assert_eq!(
            truncate_base64(block(source), MAX_BASE64_LENGTH_FOR_LOGGING),
            block(expected)
        );
    }

    fn nested(depth: usize, leaf: Value) -> Value {
        (0..depth).fold(leaf, |inner, _| json!([inner]))
    }

    #[rstest]
    #[case::at_the_limit(20, true)]
    #[case::past_the_limit(21, false)]
    fn text_nested_past_the_depth_limit_is_left_whole(
        #[case] depth: usize,
        #[case] truncated: bool,
    ) {
        let uri = format!("data:image/png;base64,{}", base64(100));
        let note = "data:image/png;base64,[base64_data truncated: 75B]";
        let expected = if truncated { json!(note) } else { json!(uri) };
        assert_eq!(
            truncate_base64(nested(depth, json!([uri])), MAX_BASE64_LENGTH_FOR_LOGGING),
            nested(depth, json!([expected]))
        );
    }

    #[rstest]
    fn a_max_length_of_zero_truncates_nothing() {
        let messages = json!([{"type": "base64", "data": base64(100)}]);
        assert_eq!(truncate_base64(messages.clone(), 0), messages);
    }
}
