mod claim;
mod keys;
mod lua;
mod types;

use std::{
    sync::{Arc, LazyLock},
    time::Duration,
};

use litellm_python_compat::{Value, json};
use litellm_redis_client::{Script, commands};

use crate::{
    DecodeError, Error, Pending, Send, codec,
    dual_cache::{DualCache, MemorySettings},
    memory::{Clock, system_clock},
    remote::{Remote, RemoteSlot},
    request::ready,
};

static READ_PIN: LazyLock<Script> = LazyLock::new(|| Script::new(lua::READ_PIN));
static CLAIM_PIN: LazyLock<Script> = LazyLock::new(|| Script::new(lua::CLAIM_PIN));

pub use keys::PinKey;
pub use types::Eligible;

pub struct RouterSessionPins {
    pins: DualCache<Value>,
}

impl RouterSessionPins {
    pub fn new(remote: Option<Remote>, default_in_memory_ttl: Option<Duration>) -> Self {
        Self::with_clock(remote, default_in_memory_ttl, system_clock())
    }

    pub fn with_clock(
        remote: Option<Remote>,
        default_in_memory_ttl: Option<Duration>,
        now: Clock,
    ) -> Self {
        Self {
            pins: DualCache::new(
                Arc::new(RemoteSlot::new(remote)),
                MemorySettings {
                    entries: None,
                    ttl: default_in_memory_ttl,
                },
                now,
            ),
        }
    }

    pub fn attach(&self, remote: Remote) -> bool {
        self.pins.redis().attach(remote)
    }

    pub fn has_redis(&self) -> bool {
        self.pins.redis().get().is_some()
    }

    pub async fn get(&self, key: &PinKey) -> Result<Option<Value>, Error> {
        if let Some(local) = self.get_local(key)? {
            return Ok(Some(local));
        }
        self.pins.get(None, &key.spelling()?, None).await
    }

    /// `in_memory_cache.get_cache`, which hands a stored string back as `json.loads` reads it.
    pub fn get_local(&self, key: &PinKey) -> Result<Option<Value>, Error> {
        Ok(self
            .pins
            .memory()
            .get(&key.spelling()?)?
            .map(|value| match value {
                Value::Str(text) => json::loads(&text).unwrap_or(Value::Str(text)),
                other => other,
            }))
    }

    /// `set_local_affinity_pin`: the entry is replaced, since `InMemoryCache.set_cache` keeps a live
    /// key's expiry.
    pub fn set_local(&self, key: &PinKey, value: Value, ttl: Duration) -> Result<(), Error> {
        self.replace_local(&key.spelling()?, value, ttl)
    }

    pub fn delete_local(&self, key: &PinKey) -> Result<(), Error> {
        Ok(self.pins.memory().delete(&key.spelling()?)?)
    }

    pub async fn set(&self, key: &PinKey, value: Value, ttl: Duration) -> Result<(), Error> {
        self.pins.set(&key.spelling()?, value, Some(ttl)).await
    }

    /// Unlike `get`, a Redis failure reaches the caller, which then keeps its memory pin.
    pub async fn get_shared(&self, key: &PinKey) -> Result<Option<Value>, Error> {
        let Some(client) = self.pins.redis().client() else {
            return Ok(None);
        };
        let raw = client
            .send(commands::get::<Vec<u8>>(&key.spelling()?))
            .await?;
        Ok(raw.as_deref().map(codec::decode).transpose()?)
    }

    /// `read_affinity_pin`'s Redis half: the read script refreshes the pin's TTL on the batch
    /// `send` picks, then memory takes what Redis holds, or drops the key on a miss. A Redis failure
    /// is the caller's, which keeps its memory pin. Without Redis, memory answers.
    pub fn read(
        self: &Arc<Self>,
        send: Send<'_>,
        key: &PinKey,
        ttl: Duration,
    ) -> Result<Pending<Option<Value>>, Error> {
        let Some(batch) = self.pins.redis().batch(send) else {
            return Ok(ready(self.get_local(key)?));
        };
        let key = key.spelling()?;
        let declared = batch.add(commands::eval::<Option<Vec<u8>>>(
            &READ_PIN,
            &[key.as_str()],
            vec![ttl.as_secs().to_string()],
        ));
        let store = Arc::clone(self);
        Ok(Box::pin(async move {
            let pin = decode_pin(declared.resolve().await?)?;
            match &pin {
                None => store.pins.memory().delete(&key)?,
                Some(value) => store.replace_local(&key, value.clone(), ttl)?,
            }
            Ok(pin)
        }))
    }

    /// `claim_affinity_pin`'s Redis half: the claim script keeps the first writer unless it is no
    /// longer eligible, and memory takes the winner. `None` when the script answered something other
    /// than a pin, or without Redis. A Redis failure is the caller's, which settles the claim with
    /// `claim_local`.
    pub async fn claim(
        &self,
        key: &PinKey,
        pin: &Value,
        ttl: Duration,
        eligible: &Eligible,
    ) -> Result<Option<Value>, Error> {
        let Some(client) = self.pins.redis().client() else {
            return Ok(None);
        };
        let key = key.spelling()?;
        let reply = client
            .send(commands::eval::<Option<Vec<u8>>>(
                &CLAIM_PIN,
                &[key.as_str()],
                claim_args(pin, ttl, eligible)?,
            ))
            .await?;
        let Some(winner) = decode_pin(reply)? else {
            return Ok(None);
        };
        self.replace_local(&key, winner.clone(), ttl)?;
        Ok(Some(winner))
    }

    /// `claim` against this router's memory alone, for a router without Redis or one whose Redis
    /// claim failed.
    pub fn claim_local(
        &self,
        key: &PinKey,
        pin: &Value,
        ttl: Duration,
        eligible: &Eligible,
    ) -> Result<Value, Error> {
        let settled = claim::local(self.get_local(key)?, pin, eligible);
        if let Some(stored) = settled.stored {
            self.set_local(key, stored, ttl)?;
        }
        Ok(settled.winner)
    }

    fn replace_local(&self, key: &str, value: Value, ttl: Duration) -> Result<(), Error> {
        Ok(self.pins.memory().refresh(key, value, ttl)?)
    }
}

/// `json.dumps(pin)`, the TTL, then `json.dumps(eligible)` and `"1"` for legacy ids, each only when
/// given, as `claim_affinity_pin` builds `ARGV`.
fn claim_args(pin: &Value, ttl: Duration, eligible: &Eligible) -> Result<Vec<String>, Error> {
    let eligible_args = match &eligible.values {
        None => Vec::new(),
        Some(values) => std::iter::once(json::dumps(&Value::List(values.clone()))?)
            .chain(eligible.preserve_legacy_model_ids.then(|| "1".to_owned()))
            .collect(),
    };
    Ok([json::dumps(pin)?, ttl.as_secs().to_string()]
        .into_iter()
        .chain(eligible_args)
        .collect())
}

/// `_decode_pin` over a script's reply: JSON, else the text as written. A reply that is not UTF-8
/// fails, as its `decode("utf-8")` raises there.
fn decode_pin(raw: Option<Vec<u8>>) -> Result<Option<Value>, Error> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let text = String::from_utf8(raw).map_err(|_| DecodeError::NotUtf8)?;
    Ok(Some(json::loads(&text).unwrap_or(Value::Str(text))))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::json(br#"{"model_id": "w"}"#.as_slice(), json::loads(r#"{"model_id": "w"}"#).unwrap())]
    #[case::json_string(br#""w""#.as_slice(), Value::Str("w".into()))]
    #[case::bare_text(b"w".as_slice(), Value::Str("w".into()))]
    fn a_script_reply_decodes_as_json_else_as_its_text(
        #[case] raw: &[u8],
        #[case] expected: Value,
    ) {
        assert_eq!(decode_pin(Some(raw.to_vec())).unwrap(), Some(expected));
    }

    #[test]
    fn a_script_reply_that_is_not_utf8_fails() {
        assert!(decode_pin(Some(vec![0xff])).is_err());
    }
}
