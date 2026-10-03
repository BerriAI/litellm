use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

use crate::{Binding, Error};

/// The pinned traversal a page belongs to. Every page of one traversal shares `id`, sees the
/// data as published at `published_at`, and can be continued until `expires_at`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Traversal {
    pub id: String,
    pub published_at: String,
    pub expires_at: String,
}

impl Traversal {
    pub fn new(binding: &Binding, published_ms: u64, expires_at: OffsetDateTime) -> Self {
        let digest = Sha256::new()
            .chain_update(b"litellm-traversal\0")
            .chain_update(binding.fingerprint())
            .chain_update(published_ms.to_be_bytes())
            .finalize();
        Self {
            id: format!("{digest:x}")[..32].to_owned(),
            published_at: rfc3339_ms(published_ms),
            expires_at: rfc3339(expires_at),
        }
    }
}

/// One page of an ordered traversal: a complete ordered prefix of what remains, and either the
/// opaque cursor for the rest or `None` when the traversal is exhausted.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(rename = "{T}Page"))]
pub struct Page<T> {
    pub items: Vec<T>,
    pub next_cursor: Option<String>,
    pub traversal: Traversal,
}

impl<T: Serialize> Page<T> {
    /// Shrinks the page to the longest prefix whose JSON fits in `max_bytes`, re-issuing the
    /// continuation cursor after the last kept item. Page size stays an upper bound and the
    /// shortened page remains a complete ordered prefix.
    pub fn bounded(
        mut self,
        max_bytes: usize,
        continue_after: impl Fn(&T) -> Result<String, Error>,
    ) -> Result<Self, Error> {
        while serde_json::to_vec(&self)?.len() > max_bytes {
            if self.items.len() <= 1 {
                return Err(Error::ResourceTooLarge);
            }
            self.items.truncate(self.items.len() / 2);
            self.next_cursor = self.items.last().map(&continue_after).transpose()?;
        }
        Ok(self)
    }
}

pub fn rfc3339(instant: OffsetDateTime) -> String {
    instant
        .replace_nanosecond(instant.millisecond() as u32 * 1_000_000)
        .unwrap_or(instant)
        .format(&Rfc3339)
        .unwrap_or_default()
}

pub fn rfc3339_ms(unix_ms: u64) -> String {
    OffsetDateTime::from_unix_timestamp_nanos(i128::from(unix_ms) * 1_000_000)
        .map(rfc3339)
        .unwrap_or_default()
}
