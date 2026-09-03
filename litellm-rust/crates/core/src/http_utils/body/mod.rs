use std::sync::Arc;

use base64::Engine;
use bytes::Bytes;
use sha2::{Digest, Sha256};

use crate::Error;
use crate::constants::JSON_BODY_CHUNK_BYTES;
mod value;
pub use value::{JsonPayload, SharedText};

#[derive(Clone)]
enum Part {
    Bytes(Bytes),
    Quoted(SharedText),
    Base64(Bytes),
}

#[derive(Clone)]
pub struct PreparedJsonBody {
    parts: Arc<[Part]>,
    length: u64,
    streamed: bool,
}

impl PreparedJsonBody {
    pub fn new(payload: JsonPayload) -> Result<Self, Error> {
        if !payload.contains_media() {
            let bytes = serde_json::to_vec(&payload)
                .map_err(|_| Error::InvalidRequest("invalid JSON request body".into()))?;
            return Ok(Self::buffered(bytes.into()));
        }
        Self::streamed(payload)
    }

    pub fn streamed(payload: JsonPayload) -> Result<Self, Error> {
        let mut parts = Vec::new();
        append_payload(payload, &mut parts)?;
        let mut body = Self {
            parts: parts.into(),
            length: 0,
            streamed: true,
        };
        body.length = body.chunks().try_fold(0_u64, |size, chunk| {
            size.checked_add(chunk.len() as u64).ok_or_else(size_error)
        })?;
        Ok(body)
    }

    pub fn buffered(bytes: Bytes) -> Self {
        Self {
            length: bytes.len() as u64,
            parts: Arc::from([Part::Bytes(bytes)]),
            streamed: false,
        }
    }

    pub fn content_length(&self) -> u64 {
        self.length
    }
    pub fn is_streamed(&self) -> bool {
        self.streamed
    }

    pub fn chunks(&self) -> impl Iterator<Item = Bytes> + Send + 'static {
        BodyChunks {
            parts: self.parts.clone(),
            part: 0,
            offset: 0,
            opened: false,
        }
    }

    pub fn sha256(&self) -> String {
        let mut digest = Sha256::new();
        for chunk in self.chunks() {
            digest.update(chunk);
        }
        format!("{:x}", digest.finalize())
    }

    pub(super) fn buffered_bytes(&self) -> Option<Bytes> {
        match self.parts.first() {
            Some(Part::Bytes(bytes)) if !self.streamed => Some(bytes.clone()),
            _ => None,
        }
    }
}

fn size_error() -> Error {
    Error::InvalidRequest("JSON request body is too large".into())
}

fn append_payload(payload: JsonPayload, parts: &mut Vec<Part>) -> Result<(), Error> {
    match payload {
        JsonPayload::String(text) => parts.push(Part::Quoted(text)),
        JsonPayload::Base64(bytes) => {
            parts.push(Part::Bytes(Bytes::from_static(b"\"")));
            parts.push(Part::Base64(bytes));
            parts.push(Part::Bytes(Bytes::from_static(b"\"")));
        }
        JsonPayload::Array(items) => {
            parts.push(Part::Bytes(Bytes::from_static(b"[")));
            for (index, item) in items.into_iter().enumerate() {
                if index > 0 {
                    parts.push(Part::Bytes(Bytes::from_static(b",")));
                }
                append_payload(item, parts)?;
            }
            parts.push(Part::Bytes(Bytes::from_static(b"]")));
        }
        JsonPayload::Object(fields) => {
            parts.push(Part::Bytes(Bytes::from_static(b"{")));
            for (index, (key, value)) in fields.into_iter().enumerate() {
                if index > 0 {
                    parts.push(Part::Bytes(Bytes::from_static(b",")));
                }
                parts.push(Part::Quoted(key.into()));
                parts.push(Part::Bytes(Bytes::from_static(b":")));
                append_payload(value, parts)?;
            }
            parts.push(Part::Bytes(Bytes::from_static(b"}")));
        }
        scalar => parts.push(Part::Bytes(
            serde_json::to_vec(&scalar)
                .map_err(|_| Error::InvalidRequest("invalid JSON value".into()))?
                .into(),
        )),
    }
    Ok(())
}

fn needs_escape(byte: u8) -> bool {
    byte < 32 || byte == b'"' || byte == b'\\'
}

struct BodyChunks {
    parts: Arc<[Part]>,
    part: usize,
    offset: usize,
    opened: bool,
}

impl Iterator for BodyChunks {
    type Item = Bytes;

    fn next(&mut self) -> Option<Bytes> {
        loop {
            let part = self.parts.get(self.part)?;
            match part {
                Part::Bytes(bytes) if self.offset < bytes.len() => {
                    let end = self
                        .offset
                        .saturating_add(JSON_BODY_CHUNK_BYTES)
                        .min(bytes.len());
                    let chunk = bytes.slice(self.offset..end);
                    self.offset = end;
                    return Some(chunk);
                }
                Part::Base64(bytes) if self.offset < bytes.len() => {
                    let end = self
                        .offset
                        .saturating_add(JSON_BODY_CHUNK_BYTES / 4 * 3)
                        .min(bytes.len());
                    let chunk =
                        base64::engine::general_purpose::STANDARD.encode(&bytes[self.offset..end]);
                    self.offset = end;
                    return Some(chunk.into());
                }
                Part::Quoted(_) if !self.opened => {
                    self.opened = true;
                    return Some(Bytes::from_static(b"\""));
                }
                Part::Quoted(text) if self.offset < text.bytes().len() => {
                    let bytes = text.bytes();
                    let limit = self
                        .offset
                        .saturating_add(JSON_BODY_CHUNK_BYTES)
                        .min(bytes.len());
                    let raw_length = bytes[self.offset..limit]
                        .iter()
                        .position(|byte| needs_escape(*byte))
                        .unwrap_or(limit - self.offset);
                    if raw_length > 0 {
                        let start = self.offset;
                        self.offset += raw_length;
                        return Some(bytes.slice(start..self.offset));
                    }
                    let start = self.offset;
                    let end = (start + (JSON_BODY_CHUNK_BYTES - 2) / 6).min(bytes.len());
                    let count = bytes[start..end]
                        .iter()
                        .take_while(|byte| needs_escape(**byte))
                        .count();
                    self.offset += count;
                    let quoted = serde_json::to_vec(&text.as_str()[start..self.offset])
                        .unwrap_or_else(|_| {
                            unreachable!("serializing a string into Vec cannot fail")
                        });
                    let output = Bytes::from(quoted);
                    return Some(output.slice(1..output.len() - 1));
                }
                Part::Quoted(_) => {
                    self.part += 1;
                    self.offset = 0;
                    self.opened = false;
                    return Some(Bytes::from_static(b"\""));
                }
                _ => {}
            }
            self.part += 1;
            self.offset = 0;
            self.opened = false;
        }
    }
}

#[cfg(test)]
mod tests;
