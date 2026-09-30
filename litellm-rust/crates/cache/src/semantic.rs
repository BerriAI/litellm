//! The embedding and prompt contract every semantic backend shares.
//!
//! Python's semantic caches all read their prompt through `get_str_from_messages`, and
//! `RedisSemanticCache._get_prompt_from_kwargs` (inherited by Valkey) adds Responses API
//! `input`. Qdrant reads messages only. Each backend picks one of the two extractors here.

use std::{future::Future, io};

use serde::Serialize;
use serde_json::{
    Value,
    ser::{CharEscape, Formatter, Serializer},
};

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

/// `get_str_from_messages`: every message's text content followed by its search results.
pub fn str_from_messages(messages: &[Value]) -> String {
    let mut text = String::new();
    for message in messages.iter().filter_map(Value::as_object) {
        match message.get("content") {
            Some(Value::String(content)) => text.push_str(content),
            Some(Value::Array(parts)) => {
                for part in parts {
                    if let Some(part_text) = part.get("text").and_then(Value::as_str) {
                        text.push_str(part_text);
                    }
                }
            }
            _ => {}
        }
        push_search_results_text(&mut text, message.get("search_results"));
    }
    text
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
    let input = context.input.as_ref()?;
    let mut parts = Vec::new();
    collect_input_text(input, &mut parts);
    let prompt = python_strip(&parts.join("\n")).to_owned();
    (!prompt.is_empty()).then_some(prompt)
}

/// `extract_search_results_text`.
fn push_search_results_text(text: &mut String, search_results: Option<&Value>) {
    let Some(Value::Array(results)) = search_results else {
        return;
    };
    for result in results.iter().filter_map(Value::as_object) {
        for key in ["source", "title"] {
            if let Some(value) = result.get(key).and_then(Value::as_str) {
                text.push_str(value);
            }
        }
        if let Some(Value::Array(content)) = result.get("content") {
            for block in content.iter().filter_map(Value::as_object) {
                if let Some(value) = block.get("text").and_then(Value::as_str) {
                    text.push_str(value);
                }
            }
        }
        if let Some(citations) = result.get("citations").filter(|value| !value.is_null()) {
            text.push_str(&compact_json(citations));
        }
    }
}

fn collect_input_text(value: &Value, parts: &mut Vec<String>) {
    match value {
        Value::String(text) => {
            push_trimmed(text, parts);
        }
        Value::Array(items) => {
            for item in items {
                collect_input_text(item, parts);
            }
        }
        Value::Object(map) => {
            if let Some(content) = map.get("content").filter(|content| !content.is_null()) {
                collect_input_text(content, parts);
                return;
            }
            for key in ["text", "output", "input_text", "output_text"] {
                if let Some(Value::String(text)) = map.get(key)
                    && push_trimmed(text, parts)
                {
                    return;
                }
            }
        }
        _ => {}
    }
}

/// Pushes `text` stripped as Python's `str.strip` does, reporting whether anything was left.
fn push_trimmed(text: &str, parts: &mut Vec<String>) -> bool {
    let trimmed = python_strip(text);
    if trimmed.is_empty() {
        return false;
    }
    parts.push(trimmed.to_owned());
    true
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
