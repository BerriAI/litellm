use std::{sync::LazyLock, time::Duration};

use litellm_spend::{Batch, BatchId, Buffer, Claimed, Key, Store, Tally};
use redis::{Script, aio::ConnectionLike};
use uuid::Uuid;

use crate::{BatchCodec, Error};

static CLAIM: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r"
local time = redis.call('TIME')
local now = tonumber(time[1]) * 1000 + math.floor(tonumber(time[2]) / 1000)
local deadline = now + tonumber(ARGV[2])
local due = redis.call('ZRANGEBYSCORE', KEYS[3], '-inf', now, 'LIMIT', 0, 1)
if #due > 0 then
    local record = redis.call('HGET', KEYS[2], due[1])
    if record then
        redis.call('ZADD', KEYS[3], deadline, due[1])
        return {due[1], record, deadline}
    end
    redis.call('ZREM', KEYS[3], due[1])
end
local blobs = redis.call('LPOP', KEYS[1], ARGV[3])
if not blobs then
    return false
end
local record = cjson.encode(blobs)
redis.call('HSET', KEYS[2], ARGV[1], record)
redis.call('ZADD', KEYS[3], deadline, ARGV[1])
return {ARGV[1], record, deadline}
",
    )
});

static NARROW: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r"
local deadline = redis.call('ZSCORE', KEYS[3], ARGV[1])
if not deadline or tonumber(deadline) ~= tonumber(ARGV[2]) then
    return 0
end
redis.call('HSET', KEYS[2], ARGV[1], ARGV[3])
redis.call('LPUSH', KEYS[1], ARGV[4])
return 1
",
    )
});

static ACK: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r"
redis.call('HDEL', KEYS[1], ARGV[1])
return redis.call('ZREM', KEYS[2], ARGV[1])
",
    )
});

static COMMIT: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r"
if redis.call('SET', KEYS[2], 1, 'NX', 'PX', ARGV[2]) then
    redis.call('RPUSH', KEYS[1], ARGV[1])
end
return 1
",
    )
});

#[derive(Clone, Copy, Debug)]
pub struct BufferSettings {
    pub claim_ttl: Duration,
    pub remember_applied_for: Duration,
    pub blobs_per_claim: usize,
}

#[derive(Clone)]
pub struct RedisBuffer<C, D> {
    connection: C,
    codec: D,
    settings: BufferSettings,
}

struct Keys {
    list: String,
    claims: String,
    deadlines: String,
}

impl<C, D> RedisBuffer<C, D> {
    pub fn new(connection: C, codec: D, settings: BufferSettings) -> Self {
        Self {
            connection,
            codec,
            settings,
        }
    }
}

impl<C, D> RedisBuffer<C, D>
where
    C: ConnectionLike + Clone + Send + Sync + 'static,
{
    fn keys<K, V>(&self) -> Keys
    where
        D: BatchCodec<K, V>,
    {
        let list = self.codec.list();
        Keys {
            list: list.to_owned(),
            claims: format!("{{{list}}}:claims"),
            deadlines: format!("{{{list}}}:claim_deadlines"),
        }
    }

    fn decode<K: Key, V: Tally>(&self, record: &str) -> Result<Batch<K, V>, Error<D::Error>>
    where
        D: BatchCodec<K, V>,
    {
        let blobs: Vec<String> = serde_json::from_str(record).map_err(Error::ClaimRecord)?;
        blobs.iter().try_fold(Batch::default(), |merged, blob| {
            Ok(merged.merge(self.codec.decode(blob).map_err(Error::Codec)?))
        })
    }

    async fn narrow<K: Key, V: Tally>(
        &self,
        keys: &Keys,
        id: &str,
        deadline: i64,
        head: &Batch<K, V>,
        tail: &Batch<K, V>,
    ) -> Result<bool, Error<D::Error>>
    where
        D: BatchCodec<K, V>,
    {
        let head = self.codec.encode(head).map_err(Error::Codec)?;
        let tail = self.codec.encode(tail).map_err(Error::Codec)?;
        let record = serde_json::to_string(&[head]).map_err(Error::ClaimRecord)?;
        let mut connection = self.connection.clone();
        let narrowed: i64 = NARROW
            .key(&keys.list)
            .key(&keys.claims)
            .key(&keys.deadlines)
            .arg(id)
            .arg(deadline)
            .arg(record)
            .arg(tail)
            .invoke_async(&mut connection)
            .await?;
        Ok(narrowed == 1)
    }
}

fn batch_id<E: std::error::Error>(id: &str) -> Result<BatchId, Error<E>> {
    Ok(BatchId::from_uuid(
        Uuid::parse_str(id).map_err(Error::ClaimId)?,
    ))
}

impl<K, V, C, D> Buffer<K, V> for RedisBuffer<C, D>
where
    K: Key,
    V: Tally,
    C: ConnectionLike + Clone + Send + Sync + 'static,
    D: BatchCodec<K, V>,
{
    type Error = Error<D::Error>;

    async fn push(&self, batch: Batch<K, V>) -> Result<(), Self::Error> {
        if batch.is_empty() {
            return Ok(());
        }
        let blob = self.codec.encode(&batch).map_err(Error::Codec)?;
        let mut connection = self.connection.clone();
        let _: i64 = redis::cmd("RPUSH")
            .arg(self.codec.list())
            .arg(blob)
            .query_async(&mut connection)
            .await?;
        Ok(())
    }

    async fn claim(&self, max: usize) -> Result<Option<Claimed<K, V>>, Self::Error> {
        if max == 0 {
            return Ok(None);
        }
        let keys = self.keys();
        let mut connection = self.connection.clone();
        let claimed: Option<(String, String, i64)> = CLAIM
            .key(&keys.list)
            .key(&keys.claims)
            .key(&keys.deadlines)
            .arg(BatchId::random().as_uuid().to_string())
            .arg(self.settings.claim_ttl.as_millis().max(1) as u64)
            .arg(self.settings.blobs_per_claim.clamp(1, max))
            .invoke_async(&mut connection)
            .await?;
        let Some((id, record, deadline)) = claimed else {
            return Ok(None);
        };
        let batch_id = batch_id(&id)?;
        let batch: Batch<K, V> = self.decode(&record)?;
        if batch.len() <= max {
            return Ok(Some(Claimed::from_buffer(batch_id, batch)));
        }
        let (head, tail) = batch.split_at(max);
        if self.narrow(&keys, &id, deadline, &head, &tail).await? {
            return Ok(Some(Claimed::from_buffer(batch_id, head)));
        }
        Ok(Some(Claimed::from_buffer(batch_id, head.merge(tail))))
    }

    async fn ack(&self, claimed: Claimed<K, V>) -> Result<(), Self::Error> {
        let keys = self.keys();
        let mut connection = self.connection.clone();
        let _: i64 = ACK
            .key(&keys.claims)
            .key(&keys.deadlines)
            .arg(claimed.id().as_uuid().to_string())
            .invoke_async(&mut connection)
            .await?;
        Ok(())
    }

    async fn release(&self, claimed: Claimed<K, V>) -> Result<(), Self::Error> {
        let keys = self.keys();
        let mut connection = self.connection.clone();
        let _: i64 = redis::cmd("ZADD")
            .arg(&keys.deadlines)
            .arg("XX")
            .arg(0)
            .arg(claimed.id().as_uuid().to_string())
            .query_async(&mut connection)
            .await?;
        Ok(())
    }
}

impl<K, V, C, D> Store<K, V> for RedisBuffer<C, D>
where
    K: Key,
    V: Tally,
    C: ConnectionLike + Clone + Send + Sync + 'static,
    D: BatchCodec<K, V>,
{
    type Error = Error<D::Error>;

    async fn commit(&self, claimed: &Claimed<K, V>) -> Result<(), Self::Error> {
        if claimed.batch().is_empty() {
            return Ok(());
        }
        let blob = self.codec.encode(claimed.batch()).map_err(Error::Codec)?;
        let list = self.codec.list();
        let mut connection = self.connection.clone();
        let _: i64 = COMMIT
            .key(list)
            .key(format!("{{{list}}}:applied:{}", claimed.id().as_uuid()))
            .arg(blob)
            .arg(self.settings.remember_applied_for.as_millis().max(1) as u64)
            .invoke_async(&mut connection)
            .await?;
        Ok(())
    }
}
