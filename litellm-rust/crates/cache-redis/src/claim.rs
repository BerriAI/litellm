use litellm_cache::{CacheCodec, ClaimCache, Error, ExactCacheContext};
use redis::Commands;

use crate::{cache::RedisCache, connection::ConnectionRef};

const CLAIM_SCRIPT: &str = concat!(
    "local current = redis.call('GET', KEYS[1]); ",
    "if ARGV[1] == '' then if current ~= false and current ~= '' then return 0; end; ",
    "elseif current ~= ARGV[1] then return 0; end; ",
    "if ARGV[3] ~= '' then redis.call('SET', KEYS[1], ARGV[3], 'EX', ARGV[2]); ",
    "elseif ARGV[4] == '1' then redis.call('EXPIRE', KEYS[1], ARGV[2]); end; return 1"
);
const CLAIM_ATTEMPTS: usize = 8;

fn stored_bytes(value: redis::Value) -> Result<Option<Vec<u8>>, Error> {
    match value {
        redis::Value::Nil => Ok(None),
        redis::Value::BulkString(bytes) => Ok(Some(bytes)),
        redis::Value::SimpleString(text) => Ok(Some(text.into_bytes())),
        _ => Err(Error::InvalidEntry),
    }
}

/// Eligibility is decided on decoded values, so a pin written by another encoder (Python's
/// `json.dumps` spacing or key order) still matches. The write is a compare-and-set on the
/// bytes that decision was made on, retried when another claimant wins the race.
fn claim<S: CacheCodec>(
    connection: &mut ConnectionRef<'_>,
    codec: &S,
    key: &str,
    candidate: S::Value,
    eligible: &[S::Value],
    ttl: u64,
) -> Result<S::Value, Error>
where
    S::Value: PartialEq,
{
    let payload = codec.encode(&candidate)?;
    if payload.is_empty() {
        return Err(Error::InvalidEntry);
    }
    for _ in 0..CLAIM_ATTEMPTS {
        let current = stored_bytes(
            connection
                .get::<_, redis::Value>(key)
                .map_err(|_| Error::Unavailable)?,
        )?
        .filter(|bytes| !bytes.is_empty());
        let existing = current
            .as_deref()
            .and_then(|bytes| codec.decode(bytes).ok())
            .filter(|existing| eligible.is_empty() || eligible.contains(existing));
        let refresh = existing
            .as_ref()
            .is_some_and(|existing| !eligible.is_empty() || *existing == candidate);
        let write: &[u8] = if existing.is_some() { b"" } else { &payload };
        let applied = redis::cmd("EVAL")
            .arg(CLAIM_SCRIPT)
            .arg(1)
            .arg(key)
            .arg(current.as_deref().unwrap_or_default())
            .arg(ttl)
            .arg(write)
            .arg(u8::from(refresh))
            .query::<bool>(connection)
            .map_err(|_| Error::Unavailable)?;
        if applied {
            return Ok(existing.unwrap_or(candidate));
        }
    }
    Err(Error::Unavailable)
}

impl<S, C> ClaimCache for RedisCache<S, C>
where
    S: CacheCodec + Clone + 'static,
    S::Value: PartialEq,
    C: redis::ConnectionLike + Send + 'static,
{
    fn claim_cache(
        &self,
        key: &str,
        candidate: S::Value,
        eligible: &[S::Value],
        context: ExactCacheContext,
    ) -> Result<S::Value, Error> {
        let key = self.namespaced_key(key);
        let ttl = self.ttl_or_default(context.ttl);
        self.execute(|connection| claim(connection, &self.codec, &key, candidate, eligible, ttl))
    }

    async fn async_claim_cache(
        &self,
        key: &str,
        candidate: S::Value,
        eligible: Vec<S::Value>,
        context: ExactCacheContext,
    ) -> Result<S::Value, Error> {
        let key = self.namespaced_key(key);
        let ttl = self.ttl_or_default(context.ttl);
        let codec = self.codec.clone();
        self.run(move |connection| claim(connection, &codec, &key, candidate, &eligible, ttl))
            .await
    }
}
