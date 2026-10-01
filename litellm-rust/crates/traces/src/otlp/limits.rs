use std::fmt;

use prost::encoding::{DecodeContext, WireType, decode_key, decode_varint, skip_field};
use serde::de::{DeserializeSeed, MapAccess, SeqAccess, Visitor};

use crate::{DecodeError, Shared};

pub(super) const MAX_DEPTH: usize = 32;
pub(super) const MAX_NODES: usize = 65_536;
pub(super) const MAX_SPANS: usize = 4_096;
pub(super) const MAX_ATTRIBUTES: usize = 256;
pub(super) const MAX_EVENTS: usize = 256;
pub(super) const MAX_DECODED_SPAN_BYTES: usize = 16 * 1024 * 1024;

pub(super) fn json_preflight(payload: &[u8]) -> Result<(), DecodeError> {
    let mut nodes = 0;
    let mut exceeded = false;
    let mut decoder = serde_json::Deserializer::from_slice(payload);
    let result = JsonBudget {
        nodes: &mut nodes,
        exceeded: &mut exceeded,
        depth: 0,
    }
    .deserialize(&mut decoder)
    .and_then(|()| decoder.end());
    if exceeded {
        return Err(DecodeError::TooLarge);
    }
    result.map_err(|_| DecodeError::InvalidPayload)
}

struct JsonBudget<'a> {
    nodes: &'a mut usize,
    exceeded: &'a mut bool,
    depth: usize,
}

impl<'de> DeserializeSeed<'de> for JsonBudget<'_> {
    type Value = ();

    fn deserialize<D: serde::Deserializer<'de>>(self, decoder: D) -> Result<(), D::Error> {
        *self.nodes += 1;
        if *self.nodes > MAX_NODES || self.depth > MAX_DEPTH {
            *self.exceeded = true;
            return Err(serde::de::Error::custom("OTLP structure exceeds budget"));
        }
        decoder.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for JsonBudget<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("OTLP JSON")
    }
    fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<(), E> {
        Ok(())
    }
    fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<(), E> {
        Ok(())
    }
    fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<(), E> {
        Ok(())
    }
    fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<(), E> {
        Ok(())
    }
    fn visit_str<E: serde::de::Error>(self, _: &str) -> Result<(), E> {
        Ok(())
    }
    fn visit_unit<E: serde::de::Error>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<(), A::Error> {
        while sequence
            .next_element_seed(JsonBudget {
                nodes: self.nodes,
                exceeded: self.exceeded,
                depth: self.depth + 1,
            })?
            .is_some()
        {}
        Ok(())
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        while map
            .next_key_seed(JsonBudget {
                nodes: self.nodes,
                exceeded: self.exceeded,
                depth: self.depth + 1,
            })?
            .is_some()
        {
            map.next_value_seed(JsonBudget {
                nodes: self.nodes,
                exceeded: self.exceeded,
                depth: self.depth + 1,
            })?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum MessageKind {
    Export,
    ResourceSpans,
    Resource,
    ScopeSpans,
    Scope,
    Span,
    Event,
    Link,
    Status,
    KeyValue,
    AnyValue,
    Array,
    KvList,
}

impl MessageKind {
    fn child(self, tag: u32) -> Option<Self> {
        match (self, tag) {
            (Self::Export, 1) => Some(Self::ResourceSpans),
            (Self::ResourceSpans, 1) => Some(Self::Resource),
            (Self::ResourceSpans, 2) => Some(Self::ScopeSpans),
            (Self::Resource, 1)
            | (Self::Scope, 3)
            | (Self::Span, 9)
            | (Self::Event, 3)
            | (Self::Link, 4)
            | (Self::KvList, 1) => Some(Self::KeyValue),
            (Self::ScopeSpans, 1) => Some(Self::Scope),
            (Self::ScopeSpans, 2) => Some(Self::Span),
            (Self::Span, 11) => Some(Self::Event),
            (Self::Span, 13) => Some(Self::Link),
            (Self::Span, 15) => Some(Self::Status),
            (Self::KeyValue, 2) | (Self::Array, 1) => Some(Self::AnyValue),
            (Self::AnyValue, 5) => Some(Self::Array),
            (Self::AnyValue, 6) => Some(Self::KvList),
            _ => None,
        }
    }
}

pub(super) fn protobuf_preflight(payload: &[u8]) -> Result<(), DecodeError> {
    scan_message(payload, MessageKind::Export, 0, &mut 0)
}

fn scan_message(
    mut payload: &[u8],
    kind: MessageKind,
    depth: usize,
    nodes: &mut usize,
) -> Result<(), DecodeError> {
    if depth > MAX_DEPTH {
        return Err(DecodeError::TooLarge);
    }
    while !payload.is_empty() {
        *nodes += 1;
        if *nodes > MAX_NODES {
            return Err(DecodeError::TooLarge);
        }
        let (tag, wire) = decode_key(&mut payload).map_err(|_| DecodeError::InvalidPayload)?;
        if let (WireType::LengthDelimited, Some(child)) = (wire, kind.child(tag)) {
            let length = decode_varint(&mut payload).map_err(|_| DecodeError::InvalidPayload)?;
            let length = usize::try_from(length).map_err(|_| DecodeError::InvalidPayload)?;
            let (message, rest) = payload
                .split_at_checked(length)
                .ok_or(DecodeError::InvalidPayload)?;
            scan_message(message, child, depth + 1, nodes)?;
            payload = rest;
        } else {
            skip_field(wire, tag, &mut payload, DecodeContext::default())
                .map_err(|_| DecodeError::InvalidPayload)?;
        }
    }
    Ok(())
}

pub(super) struct Budget {
    remaining: usize,
}

impl Budget {
    pub(super) fn new(remaining: usize) -> Self {
        Self { remaining }
    }

    pub(super) fn clone_shared<T: Clone>(
        &mut self,
        value: &Shared<T>,
        allocated_bytes: impl FnOnce(&T) -> usize,
    ) -> Result<Shared<T>, DecodeError> {
        let cloned = value.clone();
        if !value.shares_storage_with(&cloned) {
            self.consume(allocated_bytes(value))?;
        }
        Ok(cloned)
    }

    pub(super) fn consume(&mut self, bytes: usize) -> Result<(), DecodeError> {
        self.remaining = self
            .remaining
            .checked_sub(bytes)
            .ok_or(DecodeError::TooLarge)?;
        Ok(())
    }
}
