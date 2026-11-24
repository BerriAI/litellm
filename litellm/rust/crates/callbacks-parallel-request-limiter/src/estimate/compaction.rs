//! `billable_input_since_newest_compaction` from
//! `litellm/responses/litellm_completion_transformation/compaction_items.py`: the Responses input a
//! request is charged for once a compaction item stands in for the turns before it.

use serde_json::{Map, Value};

use super::{
    Out, PythonServed,
    python::{Py, py, text},
};

const WRAPPED_PREFIX: &str = "litellm_enc:";
const INSTRUCTION_ROLES: [&str; 2] = ["system", "developer"];

/// `_json_object`: the item when it is a mapping.
fn object(value: &Value) -> Out<Option<&Map<String, Value>>> {
    Ok(match py(value)? {
        Py::Dict(fields) => Some(fields),
        _ => None,
    })
}

/// `fields.get(name, default)` read as a `str`, `Ok(None)` for any other value.
fn member_text<'a>(
    fields: &'a Map<String, Value>,
    name: &str,
    default: Option<&'a str>,
) -> Out<Option<&'a str>> {
    match fields.get(name) {
        None => Ok(default),
        Some(value) => text(value),
    }
}

fn base64_value(byte: u8) -> Option<u8> {
    match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// `binascii.a2b_base64` outside strict mode, which `base64.b64decode` runs: bytes outside the
/// alphabet are skipped, a complete pad sequence ends the data, and a dangling quad raises.
fn python_b64decode(encoded: &[u8]) -> Option<Vec<u8>> {
    let mut decoded = Vec::with_capacity(encoded.len() * 3 / 4);
    let (mut quad, mut left, mut pads) = (0u8, 0u8, 0u8);
    for &byte in encoded {
        if byte == b'=' {
            if quad >= 2 {
                pads += 1;
                if quad + pads >= 4 {
                    return Some(decoded);
                }
            }
            continue;
        }
        let Some(value) = base64_value(byte) else {
            continue;
        };
        pads = 0;
        match quad {
            0 => {
                quad = 1;
                left = value;
            }
            1 => {
                quad = 2;
                decoded.push((left << 2) | (value >> 4));
                left = value & 0x0f;
            }
            2 => {
                quad = 3;
                decoded.push((left << 4) | (value >> 2));
                left = value & 0x03;
            }
            _ => {
                quad = 0;
                decoded.push((left << 6) | value);
                left = 0;
            }
        }
    }
    (quad == 0).then_some(decoded)
}

/// `ResponsesAPIRequestUtils.unwrap_encrypted_content_with_model_id(...)[1]`: the content behind a
/// `litellm_enc:<base64 model id>;` prefix, or the whole text when the prefix does not decode.
fn without_affinity_wrap(encrypted: &str) -> &str {
    if !encrypted.starts_with(WRAPPED_PREFIX) {
        return encrypted;
    }
    let Some((metadata, original)) = encrypted.split_once(';') else {
        return encrypted;
    };
    let mut metadata = metadata.replace(WRAPPED_PREFIX, "");
    let missing = metadata.chars().count() % 4;
    if missing != 0 {
        metadata.push_str(&"=".repeat(4 - missing));
    }
    match python_b64decode(metadata.as_bytes()).map(String::from_utf8) {
        Some(Ok(_)) => original,
        _ => encrypted,
    }
}

/// `decode_compaction_item`: the summary of the `{"type": "compaction", "content": ...}` block the
/// encrypted content holds. A text pydantic might parse where serde does not (NaN, Infinity, lone
/// surrogates) is left to Python.
fn decoded_summary(encrypted: &str) -> Out<Option<String>> {
    let original = without_affinity_wrap(encrypted);
    let block: Map<String, Value> = match serde_json::from_str(original) {
        Ok(block) => block,
        Err(_)
            if original.trim_start().starts_with('{')
                && ["NaN", "Infinity", "\\u"]
                    .iter()
                    .any(|found| original.contains(found)) =>
        {
            return Err(PythonServed);
        }
        Err(_) => return Ok(None),
    };
    if block.get("type").and_then(Value::as_str) != Some("compaction") {
        return Ok(None);
    }
    Ok(block
        .get("content")
        .and_then(Value::as_str)
        .filter(|summary| !summary.is_empty())
        .map(str::to_owned))
}

/// `compaction_summary`: the summary a `type: compaction` input item carries.
fn summary(item: &Value) -> Out<Option<String>> {
    let Some(fields) = object(item)? else {
        return Ok(None);
    };
    if member_text(fields, "type", None)? != Some("compaction") {
        return Ok(None);
    }
    match member_text(fields, "encrypted_content", None)? {
        Some(encrypted) => decoded_summary(encrypted),
        None => Ok(None),
    }
}

/// `_call_id`.
fn call_id(item: &Value) -> Out<Option<&str>> {
    match object(item)? {
        Some(fields) => member_text(fields, "call_id", None),
        None => Ok(None),
    }
}

/// `_is_instruction_item`: a system or developer message.
fn is_instruction(item: &Value) -> Out<bool> {
    let Some(fields) = object(item)? else {
        return Ok(false);
    };
    if fields.get("type").is_some() && member_text(fields, "type", None)? != Some("message") {
        return Ok(false);
    }
    Ok(member_text(fields, "role", None)?.is_some_and(|role| INSTRUCTION_ROLES.contains(&role)))
}

/// `_is_turn_item`: an assistant message, a reasoning item, or a call that is not an output.
fn is_turn(item: &Value) -> Out<bool> {
    let Some(fields) = object(item)? else {
        return Ok(false);
    };
    let Some(kind) = member_text(fields, "type", Some("message"))? else {
        return Ok(false);
    };
    if kind == "message" {
        return Ok(fields
            .get("role")
            .is_some_and(|role| text(role) == Ok(Some("assistant"))));
    }
    Ok(kind == "reasoning" || (call_id(item)?.is_some() && !kind.ends_with("_output")))
}

fn call_ids<'a>(items: &[&'a Value]) -> Out<Vec<&'a str>> {
    let mut found = Vec::new();
    for item in items {
        if let Some(id) = call_id(item)? {
            found.push(id);
        }
    }
    Ok(found)
}

/// `_earlier_turn_indices_answered_later`: the indices of the earlier turns holding a call whose
/// last occurrence is answered after the compaction, with that turn's non-call items.
fn moved_indices(earlier: &[&Value], answered: &[&str]) -> Out<Vec<usize>> {
    let mut last_index: Vec<(&str, usize)> = Vec::new();
    for (index, item) in earlier.iter().enumerate() {
        if let Some(id) = call_id(item)? {
            match last_index.iter_mut().find(|(known, _)| *known == id) {
                Some(entry) => entry.1 = index,
                None => last_index.push((id, index)),
            }
        }
    }
    let answered_later: Vec<usize> = last_index
        .iter()
        .filter(|(id, _)| answered.contains(id))
        .map(|(_, index)| *index)
        .collect();
    let mut moved = Vec::new();
    let mut turn: Vec<usize> = Vec::new();
    let close = |turn: &mut Vec<usize>, moved: &mut Vec<usize>| -> Out<()> {
        if turn.iter().any(|index| answered_later.contains(index)) {
            for &index in turn.iter() {
                if call_id(earlier[index])?.is_none() || answered_later.contains(&index) {
                    moved.push(index);
                }
            }
        }
        turn.clear();
        Ok(())
    };
    for (index, item) in earlier.iter().enumerate() {
        if is_turn(item)? {
            turn.push(index);
        } else {
            close(&mut turn, &mut moved)?;
        }
    }
    close(&mut turn, &mut moved)?;
    Ok(moved)
}

/// `_with_answered_turns_after_newest_compaction(input, _is_instruction_item)`.
fn since_newest_compaction(input: &[Value]) -> Out<Vec<&Value>> {
    let mut newest = 0;
    for index in (1..input.len()).rev() {
        if summary(&input[index])?.is_some() {
            newest = index;
            break;
        }
    }
    let earlier: Vec<&Value> = input[..newest].iter().collect();
    let later: Vec<&Value> = input[newest..].iter().collect();
    let moved = moved_indices(&earlier, &call_ids(&later)?)?;
    let moved_items: Vec<&Value> = moved.iter().map(|&index| earlier[index]).collect();
    let moved_call_ids = call_ids(&moved_items)?;
    let mut first_answer = later.len();
    for (index, item) in later.iter().enumerate() {
        if call_id(item)?.is_some_and(|id| moved_call_ids.contains(&id)) {
            first_answer = index;
            break;
        }
    }
    let mut kept = Vec::with_capacity(input.len());
    for (index, item) in earlier.iter().enumerate() {
        if !moved.contains(&index) && is_instruction(item)? {
            kept.push(*item);
        }
    }
    kept.extend(&later[..first_answer]);
    kept.extend(moved_items);
    kept.extend(&later[first_answer..]);
    Ok(kept)
}

/// `billable_input_since_newest_compaction`: a list input from its newest compaction on, each
/// compaction item replaced by its summary text; any other input unchanged.
pub fn billable_input(input: &Value) -> Out<Value> {
    let Py::List(items) = py(input)? else {
        return Ok(input.clone());
    };
    let kept = since_newest_compaction(items)?;
    let mut billable = Vec::with_capacity(kept.len());
    for item in kept {
        billable.push(match summary(item)? {
            Some(summary) => Value::String(summary),
            None => item.clone(),
        });
    }
    Ok(Value::Array(billable))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn compaction(summary: &str) -> Value {
        let block = json!({"type": "compaction", "content": summary}).to_string();
        json!({"type": "compaction", "encrypted_content": block})
    }

    // Expected values below were derived by reading `compaction_items.py`, not by running it.
    #[rstest]
    fn input_without_a_compaction_is_unchanged() {
        let input = json!([{"role": "user", "content": "hi"}, {"type": "reasoning"}]);
        assert_eq!(billable_input(&input), Ok(input));
    }

    #[rstest]
    fn a_text_input_is_unchanged() {
        assert_eq!(billable_input(&json!("hello")), Ok(json!("hello")));
    }

    #[rstest]
    fn earlier_turns_drop_except_instructions_and_the_compaction_becomes_its_summary() {
        let input = json!([
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "old question"},
            {"type": "message", "role": "assistant", "content": "old answer"},
            compaction("the summary"),
            {"role": "user", "content": "new question"},
        ]);
        assert_eq!(
            billable_input(&input),
            Ok(json!([
                {"role": "system", "content": "be brief"},
                "the summary",
                {"role": "user", "content": "new question"},
            ]))
        );
    }

    #[rstest]
    fn a_compaction_at_the_first_index_is_summarized_but_keeps_every_item() {
        let input = json!([compaction("s"), {"role": "user", "content": "q"}]);
        assert_eq!(
            billable_input(&input),
            Ok(json!(["s", {"role": "user", "content": "q"}]))
        );
    }

    #[rstest]
    fn an_earlier_call_answered_after_the_compaction_moves_before_its_answer() {
        let input = json!([
            {"role": "user", "content": "old"},
            {"type": "reasoning", "id": "r"},
            {"type": "function_call", "call_id": "c1", "name": "f"},
            compaction("s"),
            {"role": "user", "content": "new"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        ]);
        assert_eq!(
            billable_input(&input),
            Ok(json!([
                "s",
                {"role": "user", "content": "new"},
                {"type": "reasoning", "id": "r"},
                {"type": "function_call", "call_id": "c1", "name": "f"},
                {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            ]))
        );
    }

    #[rstest]
    fn an_undecodable_compaction_is_kept_as_an_item() {
        let input = json!([
            {"role": "user", "content": "q"},
            {"type": "compaction", "encrypted_content": "opaque-provider-blob"},
        ]);
        assert_eq!(billable_input(&input), Ok(input));
    }

    #[rstest]
    #[case::wrapped(
        "litellm_enc:bW9kZWxfaWQ6YWJj;{\"type\":\"compaction\",\"content\":\"s\"}",
        Some("s")
    )]
    #[case::bad_base64("litellm_enc:a;{\"type\":\"compaction\",\"content\":\"s\"}", None)]
    #[case::no_separator("litellm_enc:bW9kZWxfaWQ6YWJj", None)]
    #[case::empty_content("{\"type\":\"compaction\",\"content\":\"\"}", None)]
    fn the_affinity_wrap_is_removed_only_when_its_prefix_decodes(
        #[case] encrypted: &str,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(decoded_summary(encrypted), Ok(expected.map(str::to_owned)));
    }

    #[rstest]
    #[case::plain(b"aGk=".as_slice(), Some(b"hi".to_vec()))]
    #[case::unpadded_tail(b"aGk".as_slice(), None)]
    #[case::one_extra(b"aGkxa".as_slice(), None)]
    #[case::junk_skipped(b"a G!k=".as_slice(), Some(b"hi".to_vec()))]
    #[case::data_after_padding(b"aGk=QUJD".as_slice(), Some(b"hi".to_vec()))]
    fn base64_decodes_like_binascii_outside_strict_mode(
        #[case] encoded: &[u8],
        #[case] expected: Option<Vec<u8>>,
    ) {
        assert_eq!(python_b64decode(encoded), expected);
    }
}
