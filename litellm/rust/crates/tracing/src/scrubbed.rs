//! A sink adapter that scrubs secrets from every record before the inner sink sees it. A host whose output does no
//! redaction of its own, such as one that prints records to stderr, wraps its sink in it.

use std::sync::Arc;

use litellm_redaction::{REDACTED, SecretRedactor};
use parking_lot::RwLock;
use serde_json::{Map, Value};

use crate::{Metadata, Record, Sink};

pub struct Scrubbed<S> {
    sink: S,
    redactor: Arc<RwLock<SecretRedactor>>,
}

/// Sets the length a `Scrubbed` sink scrubs `sk-` keys at, after the sink is installed. A host that installs its
/// sink before it reads its config starts with the length its process was given and sets the config's here.
#[derive(Clone)]
pub struct KeyLength(Arc<RwLock<SecretRedactor>>);

impl KeyLength {
    pub fn set(&self, minimum_custom_key_length: usize) {
        *self.0.write() = SecretRedactor::new(minimum_custom_key_length);
    }
}

impl<S> Scrubbed<S> {
    pub fn new(sink: S, minimum_custom_key_length: usize) -> Self {
        Self {
            sink,
            redactor: Arc::new(RwLock::new(SecretRedactor::new(minimum_custom_key_length))),
        }
    }

    pub fn key_length(&self) -> KeyLength {
        KeyLength(Arc::clone(&self.redactor))
    }
}

/// One record's scrubbing, with the redactor as it is while that record is emitted.
struct Scrub<'a>(&'a SecretRedactor);

impl Scrub<'_> {
    fn field(&self, name: &str, value: &Value) -> Value {
        match value {
            Value::String(text) => self.text(name, text).into(),
            other => self.document(name, other.clone()),
        }
    }

    /// Text that is a JSON object or array, such as a logged request body, is scrubbed value by value and written
    /// back as JSON. A secret inside a nested string is then seen without the escaping the outer document adds, and
    /// each value is scrubbed under its own key. The `data:` lines of server-sent events are scrubbed the same way.
    /// Any other text is scrubbed as it is.
    fn text(&self, name: &str, text: &str) -> String {
        if let Some(document) = json_document(text) {
            return self.document(name, document).to_string();
        }
        if text.lines().any(|line| event_data(line).is_some()) {
            return self.events(name, text);
        }
        self.leaf(name, text)
    }

    /// A stream chunk, with each event's JSON scrubbed value by value and the lines between them scrubbed as text.
    fn events(&self, name: &str, text: &str) -> String {
        let mut scrubbed = String::with_capacity(text.len());
        let mut plain = String::new();
        for line in text.split_inclusive('\n') {
            match event_data(line) {
                Some(event) => {
                    scrubbed.push_str(&self.leaf(name, &std::mem::take(&mut plain)));
                    scrubbed.push_str(event.prefix);
                    scrubbed.push_str(&self.document(name, event.document).to_string());
                    scrubbed.push_str(event.ending);
                }
                None => plain.push_str(line),
            }
        }
        scrubbed.push_str(&self.leaf(name, &plain));
        scrubbed
    }

    /// A value is scrubbed with its name, so one under a name like `api_key` is hidden whatever it looks like.
    fn leaf(&self, name: &str, text: &str) -> String {
        self.0
            .try_redact_structured(Some(name), text)
            .unwrap_or_else(|_| REDACTED.to_owned())
    }

    /// Every part of a JSON value that can hold text is scrubbed: strings and numbers under their key, and the keys
    /// themselves, since a caller may use a secret as a key.
    fn document(&self, name: &str, value: Value) -> Value {
        match value {
            Value::String(text) => self.leaf(name, &text).into(),
            Value::Number(number) => {
                let text = number.to_string();
                let scrubbed = self.leaf(name, &text);
                if scrubbed == text {
                    Value::Number(number)
                } else {
                    scrubbed.into()
                }
            }
            Value::Array(items) => items
                .into_iter()
                .map(|item| self.document(name, item))
                .collect(),
            Value::Object(entries) => entries
                .into_iter()
                .map(|(key, entry)| {
                    let scrubbed = self.document(&key, entry);
                    (self.0.redact(&key), scrubbed)
                })
                .collect::<Map<String, Value>>()
                .into(),
            other => other,
        }
    }
}

fn json_document(text: &str) -> Option<Value> {
    match serde_json::from_str(text) {
        Ok(document @ (Value::Object(_) | Value::Array(_))) => Some(document),
        _ => None,
    }
}

/// The JSON one `data:` line of a server-sent event carries, and the text around it on that line.
struct EventData<'a> {
    prefix: &'a str,
    document: Value,
    ending: &'a str,
}

fn event_data(line: &str) -> Option<EventData<'_>> {
    let after = line.strip_prefix("data:")?;
    let payload = after.trim_start();
    let json = payload.trim_end();
    Some(EventData {
        prefix: &line[..line.len() - payload.len()],
        document: json_document(json)?,
        ending: &payload[json.len()..],
    })
}

impl<S: Sink> Sink for Scrubbed<S> {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        self.sink.enabled(metadata)
    }

    fn emit(&self, record: &Record) {
        let redactor = self.redactor.read();
        let scrub = Scrub(&redactor);
        self.sink.emit(&Record {
            metadata: record.metadata,
            message: redactor.redact(&record.message),
            fields: record
                .fields
                .iter()
                .map(|(name, value)| (name.clone(), scrub.field(name, value)))
                .collect(),
        });
    }
}
