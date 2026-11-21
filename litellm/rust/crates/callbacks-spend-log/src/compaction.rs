//! Compaction blocks in stored requests and responses.
//!
//! A Responses API `compaction` item carries the conversation summary inside `encrypted_content`,
//! as the JSON of the provider's compaction block, possibly wrapped with LiteLLM's model affinity
//! prefix (`litellm_enc:<base64 model id>;`). The row keeps the block but bounds the summary to
//! the larger of the prompt limit and 64 KiB, so the row stays loadable.

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use litellm_python_compat::{Value, json};

use crate::py::{Dict, entries, get_or_none, lookup, text};

const AFFINITY_PREFIX: &str = "litellm_enc:";

/// `ResponsesAPIRequestUtils.unwrap_encrypted_content_with_model_id(content)[1]`.
fn without_affinity_wrap(content: &str) -> &str {
    let Some(wrapped) = content.strip_prefix(AFFINITY_PREFIX) else {
        return content;
    };
    let Some((metadata, original)) = wrapped.split_once(';') else {
        return content;
    };
    let padded = format!("{metadata}{}", "=".repeat((4 - metadata.len() % 4) % 4));
    let decodes = STANDARD
        .decode(padded)
        .ok()
        .is_some_and(|bytes| String::from_utf8(bytes).is_ok());
    if decodes { original } else { content }
}

/// A compaction block decoded from `encrypted_content`: its summary and the block itself.
struct CompactionBlock {
    summary: String,
    wire_block: Vec<(Value, Value)>,
}

/// `decode_compaction_item(encrypted_content)`.
fn decode_compaction_item(content: &str) -> Option<CompactionBlock> {
    let Ok(Value::Dict(block)) = json::loads(without_affinity_wrap(content)) else {
        return None;
    };
    let block_value = Value::Dict(block);
    let is_compaction =
        matches!(get_or_none(&block_value, "type"), Value::Str(kind) if kind == "compaction");
    match (is_compaction, get_or_none(&block_value, "content")) {
        (true, Value::Str(summary)) if !summary.is_empty() => Some(CompactionBlock {
            summary: summary.clone(),
            wire_block: match block_value {
                Value::Dict(block) => block,
                _ => Vec::new(),
            },
        }),
        _ => None,
    }
}

/// Whether a mapping is a `compaction` item.
pub(crate) fn is_compaction_item(item: &[(Value, Value)]) -> bool {
    matches!(lookup(item, "type"), Some(Value::Str(kind)) if kind == "compaction")
}

/// `compaction_summary(item)`: the summary inside a compaction item's `encrypted_content`.
pub(crate) fn compaction_summary(item: &Value) -> Option<String> {
    compaction_entries_summary(entries(item)?)
}

/// [`compaction_summary`] over a dict's entries.
pub(crate) fn compaction_entries_summary(item: &[(Value, Value)]) -> Option<String> {
    if !is_compaction_item(item) {
        return None;
    }
    match lookup(item, "encrypted_content") {
        Some(Value::Str(content)) => decode_compaction_item(content).map(|block| block.summary),
        _ => None,
    }
}

/// `with_compaction_summary(encrypted_content, summary)`: the same content with the block's summary
/// replaced, keeping any affinity wrap.
pub(crate) fn with_compaction_summary(content: &str, summary: &str) -> String {
    let original = without_affinity_wrap(content);
    let Some(block) = decode_compaction_item(original) else {
        return content.to_owned();
    };
    let mut rewritten = Dict::from_entries(&block.wire_block);
    rewritten.set("content", text(summary));
    let Ok(rewritten) = json::dumps_compact(&rewritten.into_value()) else {
        return content.to_owned();
    };
    format!(
        "{}{rewritten}",
        content.strip_suffix(original).unwrap_or(content)
    )
}

/// `summary_only_compaction_content(summary)`.
pub(crate) fn summary_only_compaction_content(summary: &str) -> String {
    json::dumps_compact(&Value::Dict(vec![
        (text("type"), text("compaction")),
        (text("content"), text(summary)),
    ]))
    .unwrap_or_default()
}
