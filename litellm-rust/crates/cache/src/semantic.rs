//! The embedding and prompt contract every semantic backend shares.
//!
//! Python's semantic caches all read their prompt through `get_str_from_messages_with_tools`, and
//! `RedisSemanticCache._get_prompt_from_kwargs` (inherited by Valkey) adds Responses API
//! `input`. Qdrant reads messages only. Each backend picks one of the two extractors here.

use std::{collections::HashMap, future::Future, io};

use serde::Serialize;
use serde_json::{
    Map, Value,
    ser::{CharEscape, Formatter, Serializer},
};
use sha2::{Digest, Sha256};

use crate::{BaseCache, Error, SemanticCacheContext};

/// Turns a prompt into the vector a semantic backend stores and searches with.
///
/// `metadata` is the request metadata, which a host embedder may route on. Hosts that can only
/// embed asynchronously keep the default `embed`; backends that serve sync calls through a
/// runtime then block on `async_embed` instead.
pub trait Embedder: Send + Sync + 'static {
    fn embed(&self, _prompt: &str, _metadata: Option<&Value>) -> Result<Vec<f32>, Error> {
        Err(Error::UnsupportedOperation)
    }

    fn async_embed(
        &self,
        prompt: &str,
        metadata: Option<&Value>,
    ) -> impl Future<Output = Result<Vec<f32>, Error>> + Send;
}

/// One semantic read: the cached value, if any, and the similarity Python's backend writes to
/// `metadata["semantic-similarity"]`. `similarity` is `None` when the backend reports none.
#[derive(Clone, Debug, PartialEq)]
pub struct SemanticLookup<V> {
    pub value: Option<V>,
    pub similarity: Option<f64>,
}

impl<V> SemanticLookup<V> {
    /// A read that found nothing, with the similarity Python records for it.
    pub fn miss(similarity: Option<f64>) -> Self {
        Self {
            value: None,
            similarity,
        }
    }
}

/// A semantic backend's read that also reports the similarity of the closest cached prompt,
/// the value Python stamps onto the request metadata as `semantic-similarity`.
pub trait SemanticCache: BaseCache {
    fn get_cache_with_similarity(
        &self,
        key: &str,
        context: &Self::Context,
    ) -> Result<SemanticLookup<Self::Value>, Error>;

    fn async_get_cache_with_similarity(
        &self,
        key: &str,
        context: &Self::Context,
    ) -> impl Future<Output = Result<SemanticLookup<Self::Value>, Error>> + Send;
}

/// An embedding computed ahead of time, for callers that already hold the vector.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedEmbedding(pub Vec<f32>);

impl Embedder for PreparedEmbedding {
    fn embed(&self, _prompt: &str, _metadata: Option<&Value>) -> Result<Vec<f32>, Error> {
        Ok(self.0.clone())
    }

    async fn async_embed(
        &self,
        _prompt: &str,
        _metadata: Option<&Value>,
    ) -> Result<Vec<f32>, Error> {
        Ok(self.0.clone())
    }
}

const PLAIN_TYPES: [&str; 4] = ["text", "input_text", "output_text", "message"];
const PLAIN_KEYS: [&str; 5] = ["role", "type", "text", "content", "status"];
const TOOL_ROLES: [&str; 2] = ["tool", "function"];
const IGNORED_KEYS: [&str; 1] = ["cache_control"];
const CALL_ID_KEYS: [&str; 4] = ["id", "call_id", "tool_use_id", "tool_call_id"];
const OPAQUE_KEYS: [&str; 4] = ["data", "file_data", "signature", "encrypted_content"];

/// `get_str_from_messages_with_tools`: plain text parts stay plain text, and every other part is
/// kept as compact JSON with call ids replaced by their position and opaque blobs by a digest.
pub fn str_from_messages(messages: &[Value]) -> String {
    let messages: Vec<&Value> = messages
        .iter()
        .filter(|message| message.is_object())
        .collect();
    let call_ordinals = call_id_ordinals(messages.iter().copied());
    messages
        .into_iter()
        .filter_map(Value::as_object)
        .map(|message| message_prompt(message, &call_ordinals))
        .collect()
}

/// `get_str_from_responses_input`: `str_from_messages` for a Responses API `input`, one stripped
/// line per part. `None` when nothing is left.
fn str_from_responses_input(input: &Value) -> Option<String> {
    let call_ordinals = call_id_ordinals(std::iter::once(input));
    let mut parts = Vec::new();
    push_responses_input_parts(input, &call_ordinals, &mut parts);
    let prompt = python_strip(&parts.join("\n")).to_owned();
    (!prompt.is_empty()).then_some(prompt)
}

fn message_prompt(message: &Map<String, Value>, call_ordinals: &HashMap<&str, usize>) -> String {
    let is_tool = message
        .get("role")
        .and_then(Value::as_str)
        .is_some_and(|role| TOOL_ROLES.contains(&role));
    if is_tool || !is_plain(message) {
        return compact_json(&normalized(message, call_ordinals));
    }
    plain_prompt(message, call_ordinals)
}

fn plain_prompt(value: &Map<String, Value>, call_ordinals: &HashMap<&str, usize>) -> String {
    let mut prompt = value
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    push_content_prompt(&mut prompt, value.get("content"), call_ordinals);
    prompt
}

fn push_content_prompt(
    prompt: &mut String,
    content: Option<&Value>,
    call_ordinals: &HashMap<&str, usize>,
) {
    match content {
        None | Some(Value::Null) => {}
        Some(Value::String(text)) => prompt.push_str(text),
        Some(Value::Array(items)) => {
            for item in items {
                push_content_prompt(prompt, Some(item), call_ordinals);
            }
        }
        Some(Value::Object(map)) if is_plain(map) => {
            prompt.push_str(&plain_prompt(map, call_ordinals));
        }
        Some(Value::Object(map)) => prompt.push_str(&compact_json(&normalized(map, call_ordinals))),
        Some(other) => prompt.push_str(&compact_json(other)),
    }
}

fn push_responses_input_parts(
    value: &Value,
    call_ordinals: &HashMap<&str, usize>,
    parts: &mut Vec<String>,
) {
    match value {
        Value::Null => {}
        Value::String(text) => push_stripped(text, parts),
        Value::Array(items) => {
            for item in items {
                push_responses_input_parts(item, call_ordinals, parts);
            }
        }
        Value::Object(map) if is_plain(map) => {
            push_stripped(
                map.get("text").and_then(Value::as_str).unwrap_or_default(),
                parts,
            );
            if let Some(content) = map.get("content") {
                push_responses_input_parts(content, call_ordinals, parts);
            }
        }
        Value::Object(map) => parts.push(compact_json(&normalized(map, call_ordinals))),
        other => parts.push(compact_json(other)),
    }
}

fn push_stripped(text: &str, parts: &mut Vec<String>) {
    let stripped = python_strip(text);
    if !stripped.is_empty() {
        parts.push(stripped.to_owned());
    }
}

/// `_is_plain`: a text-only part whose every kept key is a plain or call id key.
fn is_plain(value: &Map<String, Value>) -> bool {
    let plain_type = match value.get("type") {
        None | Some(Value::Null) => true,
        Some(Value::String(value_type)) => PLAIN_TYPES.contains(&value_type.as_str()),
        Some(_) => false,
    };
    let plain_text = matches!(
        value.get("text"),
        None | Some(Value::Null | Value::String(_))
    );
    plain_type
        && plain_text
        && kept_entries(value).all(|(key, _)| {
            PLAIN_KEYS.contains(&key.as_str()) || CALL_ID_KEYS.contains(&key.as_str())
        })
}

fn kept_entries(value: &Map<String, Value>) -> impl Iterator<Item = (&String, &Value)> {
    value
        .iter()
        .filter(|(key, item)| !IGNORED_KEYS.contains(&key.as_str()) && !is_empty(item))
}

fn is_empty(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(text) => text.is_empty(),
        Value::Array(items) => items.is_empty(),
        Value::Object(map) => map.is_empty(),
        Value::Bool(_) | Value::Number(_) => false,
    }
}

/// `_normalized`: the part minus ignored and empty entries, with call ids replaced by their
/// position and opaque blobs by `sha256:` plus the first 16 hex digits of their digest.
fn normalized(value: &Map<String, Value>, call_ordinals: &HashMap<&str, usize>) -> Value {
    Value::Object(
        kept_entries(value)
            .map(|(key, item)| {
                (
                    key.clone(),
                    normalized_value(item, call_ordinals, Some(key)),
                )
            })
            .collect(),
    )
}

fn normalized_value(
    value: &Value,
    call_ordinals: &HashMap<&str, usize>,
    key: Option<&str>,
) -> Value {
    match value {
        Value::Object(map) => normalized(map, call_ordinals),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| normalized_value(item, call_ordinals, None))
                .collect(),
        ),
        Value::String(text) if key.is_some_and(|key| CALL_ID_KEYS.contains(&key)) => call_ordinals
            .get(text.as_str())
            .map_or_else(|| value.clone(), |ordinal| Value::from(*ordinal)),
        Value::String(text)
            if key.is_some_and(|key| OPAQUE_KEYS.contains(&key)) || text.starts_with("data:") =>
        {
            Value::String(digest(text))
        }
        other => other.clone(),
    }
}

fn digest(text: &str) -> String {
    let hash = Sha256::digest(text.as_bytes());
    let hex: String = hash[..8].iter().map(|byte| format!("{byte:02x}")).collect();
    format!("sha256:{hex}")
}

/// `_call_id_ordinals`: the 1-based position of each distinct call id string, first seen first.
fn call_id_ordinals<'a>(values: impl Iterator<Item = &'a Value>) -> HashMap<&'a str, usize> {
    let mut ids = Vec::new();
    for value in values {
        collect_call_ids(value, None, &mut ids);
    }
    let mut ordinals = HashMap::new();
    for id in ids {
        let next = ordinals.len() + 1;
        ordinals.entry(id).or_insert(next);
    }
    ordinals
}

fn collect_call_ids<'a>(value: &'a Value, key: Option<&str>, ids: &mut Vec<&'a str>) {
    match value {
        Value::String(text) if key.is_some_and(|key| CALL_ID_KEYS.contains(&key)) => ids.push(text),
        Value::Object(map) => {
            for (child_key, child) in map {
                if !IGNORED_KEYS.contains(&child_key.as_str()) {
                    collect_call_ids(child, Some(child_key), ids);
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_call_ids(item, None, ids);
            }
        }
        _ => {}
    }
}

/// The messages prompt Qdrant embeds: `None` when the request carries no messages.
pub fn prompt_from_messages(context: &SemanticCacheContext) -> Option<String> {
    let messages = context.messages.as_ref()?.as_array()?;
    (!messages.is_empty()).then(|| str_from_messages(messages))
}

/// `RedisSemanticCache._get_prompt_from_kwargs`: chat messages first, then the text parts of a
/// Responses API `input`. `None` when neither yields a prompt.
pub fn prompt_from_context(context: &SemanticCacheContext) -> Option<String> {
    if let Some(messages) = context.messages.as_ref().and_then(Value::as_array)
        && !messages.is_empty()
    {
        return Some(str_from_messages(messages));
    }
    str_from_responses_input(context.input.as_ref()?)
}

/// `str.strip()`: Python's whitespace also covers the ASCII information separators.
fn python_strip(text: &str) -> &str {
    text.trim_matches(|character: char| {
        character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character)
    })
}

/// `json.dumps(value, separators=(",", ":"))`: compact, key insertion order, `ensure_ascii`.
fn compact_json(value: &Value) -> String {
    let mut output = Vec::new();
    // Serializing a `Value` into memory cannot fail.
    let _ = value.serialize(&mut Serializer::with_formatter(&mut output, AsciiFormatter));
    String::from_utf8(output).unwrap_or_default()
}

struct AsciiFormatter;

impl Formatter for AsciiFormatter {
    fn write_string_fragment<W>(&mut self, writer: &mut W, fragment: &str) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        let mut start = 0;
        for (index, character) in fragment.char_indices() {
            if character.is_ascii() && character != '\u{7f}' {
                continue;
            }
            writer.write_all(&fragment.as_bytes()[start..index])?;
            let mut units = [0; 2];
            for unit in character.encode_utf16(&mut units) {
                write!(writer, "\\u{unit:04x}")?;
            }
            start = index + character.len_utf8();
        }
        writer.write_all(&fragment.as_bytes()[start..])
    }

    fn write_f64<W>(&mut self, writer: &mut W, value: f64) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(python_float_repr(value).as_bytes())
    }

    fn write_char_escape<W>(&mut self, writer: &mut W, escape: CharEscape) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        match escape {
            CharEscape::AsciiControl(byte) => write!(writer, "\\u{byte:04x}"),
            escape => serde_json::ser::CompactFormatter.write_char_escape(writer, escape),
        }
    }
}

/// `repr(float)`: the shortest round-trip digits, positional between `1e-4` and `1e16`, and
/// otherwise scientific with a signed exponent of at least two digits.
fn python_float_repr(value: f64) -> String {
    // `{:e}` yields the shortest round-trip digits, e.g. `1.5e-7`.
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let (sign, mantissa) = mantissa
        .strip_prefix('-')
        .map_or(("", mantissa), |rest| ("-", rest));
    let digits = mantissa.replace('.', "");
    if !(-4..16).contains(&exponent) {
        let fraction = &digits[1..];
        let mantissa = if fraction.is_empty() {
            digits[..1].to_owned()
        } else {
            format!("{}.{fraction}", &digits[..1])
        };
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        return format!("{sign}{mantissa}e{exponent_sign}{:02}", exponent.abs());
    }
    let point = exponent + 1;
    let positional = if point <= 0 {
        format!("0.{}{digits}", "0".repeat(point.unsigned_abs() as usize))
    } else if point as usize >= digits.len() {
        format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
    } else {
        let (whole, fraction) = digits.split_at(point as usize);
        format!("{whole}.{fraction}")
    };
    format!("{sign}{positional}")
}
