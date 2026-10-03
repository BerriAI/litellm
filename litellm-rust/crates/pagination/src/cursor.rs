use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::value::RawValue;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use time::OffsetDateTime;

use crate::Error;

const FORMAT_VERSION: u8 = 1;
const SIGNATURE_DOMAIN: &[u8] = b"litellm-pagination-cursor-v1\0";

struct Key {
    id: String,
    secret: Vec<u8>,
}

/// The HMAC keys cursors are signed with. The first key signs; every key verifies, so a rotated
/// key keeps earlier cursors valid until they expire.
pub struct KeyRing {
    keys: Vec<Key>,
}

impl KeyRing {
    pub fn new<S: AsRef<[u8]>>(secrets: impl IntoIterator<Item = S>) -> Result<Self, Error> {
        let keys = secrets
            .into_iter()
            .map(|secret| {
                let secret = secret.as_ref();
                if secret.is_empty() {
                    return Err(Error::InvalidKeys);
                }
                let digest = Sha256::new()
                    .chain_update(b"litellm-pagination-key\0")
                    .chain_update(secret)
                    .finalize();
                Ok(Key {
                    id: format!("{digest:x}")[..8].to_owned(),
                    secret: secret.to_vec(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if keys.is_empty() {
            return Err(Error::InvalidKeys);
        }
        Ok(Self { keys })
    }

    pub fn signing_key_id(&self) -> &str {
        &self.keys[0].id
    }

    fn sign(key: &Key, binding: &Binding, payload: &[u8]) -> Vec<u8> {
        let mut mac =
            Hmac::<Sha256>::new_from_slice(&key.secret).expect("HMAC accepts keys of any length");
        mac.update(SIGNATURE_DOMAIN);
        mac.update(&binding.fingerprint());
        mac.update(payload);
        mac.finalize().into_bytes().to_vec()
    }

    /// Encodes `cursor` as an opaque token bound to `binding`.
    pub fn encode<P: Serialize>(
        &self,
        binding: &Binding,
        cursor: &Cursor<P>,
    ) -> Result<String, Error> {
        let payload = serde_json::to_vec(&Envelope {
            version: FORMAT_VERSION,
            key_id: &self.keys[0].id,
            expires_unix: cursor.expires_at.unix_timestamp(),
            published_ms: cursor.published_ms,
            revision: &cursor.revision,
            position: &cursor.position,
        })?;
        let tag = Self::sign(&self.keys[0], binding, &payload);
        Ok(format!(
            "{}.{}",
            URL_SAFE_NO_PAD.encode(payload),
            URL_SAFE_NO_PAD.encode(tag)
        ))
    }

    /// Verifies `token` against `binding` and `now`, returning the typed cursor it carries.
    pub fn decode<P: DeserializeOwned>(
        &self,
        binding: &Binding,
        token: &str,
        now: OffsetDateTime,
    ) -> Result<Cursor<P>, Error> {
        let (payload, tag) = token.split_once('.').ok_or(Error::InvalidCursor)?;
        let payload = URL_SAFE_NO_PAD
            .decode(payload)
            .map_err(|_| Error::InvalidCursor)?;
        let tag = URL_SAFE_NO_PAD
            .decode(tag)
            .map_err(|_| Error::InvalidCursor)?;
        let envelope: Envelope<'_, Box<RawValue>> =
            serde_json::from_slice(&payload).map_err(|_| Error::InvalidCursor)?;
        let key = self
            .keys
            .iter()
            .find(|key| key.id == envelope.key_id)
            .ok_or(Error::InvalidCursor)?;
        if envelope.version != FORMAT_VERSION
            || !bool::from(Self::sign(key, binding, &payload).ct_eq(&tag))
        {
            return Err(Error::InvalidCursor);
        }
        if envelope.expires_unix <= now.unix_timestamp() {
            return Err(Error::TraversalExpired);
        }
        Ok(Cursor {
            position: serde_json::from_str(envelope.position.get())
                .map_err(|_| Error::InvalidCursor)?,
            revision: envelope.revision.to_owned(),
            published_ms: envelope.published_ms,
            expires_at: OffsetDateTime::from_unix_timestamp(envelope.expires_unix)
                .map_err(|_| Error::InvalidCursor)?,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct Envelope<'a, P> {
    #[serde(rename = "v")]
    version: u8,
    #[serde(rename = "k", borrow)]
    key_id: &'a str,
    #[serde(rename = "e")]
    expires_unix: i64,
    #[serde(rename = "p")]
    published_ms: u64,
    #[serde(rename = "r", borrow)]
    revision: &'a str,
    #[serde(rename = "c")]
    position: P,
}

/// What a cursor is valid for: one resource, one authorization scope and one query. Any of them
/// changing makes earlier cursors invalid instead of silently continuing a different traversal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Binding {
    resource: String,
    scope: String,
    query: String,
}

impl Binding {
    pub fn new(
        resource: &str,
        scope: &impl Serialize,
        query: &impl Serialize,
    ) -> Result<Self, Error> {
        Ok(Self {
            resource: resource.to_owned(),
            scope: digest(scope)?,
            query: digest(query)?,
        })
    }

    pub(crate) fn fingerprint(&self) -> Vec<u8> {
        Sha256::new()
            .chain_update(self.resource.as_bytes())
            .chain_update(b"\0")
            .chain_update(self.scope.as_bytes())
            .chain_update(b"\0")
            .chain_update(self.query.as_bytes())
            .finalize()
            .to_vec()
    }
}

fn digest(value: &impl Serialize) -> Result<String, Error> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(value)?)))
}

/// A typed continuation: where the next page starts, which revision of the data it continues,
/// when that data was published and until when the traversal can be continued.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Cursor<P> {
    pub position: P,
    pub revision: String,
    pub published_ms: u64,
    pub expires_at: OffsetDateTime,
}

impl<P> Cursor<P> {
    /// Fails with `traversal_changed` when the data behind the traversal no longer matches the
    /// revision the cursor was issued for.
    pub fn require_revision(&self, current: &str) -> Result<(), Error> {
        if self.revision.as_bytes().ct_eq(current.as_bytes()).into() {
            Ok(())
        } else {
            Err(Error::TraversalChanged)
        }
    }

    pub fn advance<Q>(&self, position: Q) -> Cursor<Q> {
        Cursor {
            position,
            revision: self.revision.clone(),
            published_ms: self.published_ms,
            expires_at: self.expires_at,
        }
    }
}
