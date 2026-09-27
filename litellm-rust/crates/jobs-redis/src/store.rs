use std::{sync::LazyLock, time::Duration};

use litellm_jobs::{Acquire, HolderId, JobName, LeaseStore, Renewal};
use redis::{Script, aio::ConnectionLike};

use crate::{Error, LeaseNaming};

static ACQUIRE: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r"
local owner = redis.call('GET', KEYS[1])
if owner and owner ~= ARGV[1] then
    return 0
end
redis.call('SET', KEYS[1], ARGV[1], 'PX', ARGV[2])
return 1
",
    )
});

static RENEW: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r"
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
    return 0
end
return redis.call('PEXPIRE', KEYS[1], ARGV[2])
",
    )
});

static RELEASE: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r"
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
",
    )
});

#[derive(Clone)]
pub struct RedisLeaseStore<C, N> {
    connection: C,
    naming: N,
}

impl<C, N> RedisLeaseStore<C, N>
where
    C: ConnectionLike + Clone + Send + Sync + 'static,
    N: LeaseNaming,
{
    pub fn new(connection: C, naming: N) -> Self {
        Self { connection, naming }
    }

    async fn invoke(
        &self,
        script: &Script,
        job: &JobName,
        holder: &HolderId,
        ttl: Option<Duration>,
    ) -> Result<bool, Error> {
        let mut invocation = script.key(self.naming.key(job));
        invocation.arg(self.naming.owner(holder));
        if let Some(ttl) = ttl {
            invocation.arg(ttl.as_millis().max(1) as u64);
        }
        let mut connection = self.connection.clone();
        Ok(invocation.invoke_async::<i64>(&mut connection).await? == 1)
    }
}

impl<C, N> LeaseStore for RedisLeaseStore<C, N>
where
    C: ConnectionLike + Clone + Send + Sync + 'static,
    N: LeaseNaming,
{
    type Error = Error;

    async fn try_acquire(
        &self,
        job: &JobName,
        holder: &HolderId,
        ttl: Duration,
    ) -> Result<Acquire, Self::Error> {
        Ok(match self.invoke(&ACQUIRE, job, holder, Some(ttl)).await? {
            true => Acquire::Held,
            false => Acquire::Busy,
        })
    }

    async fn renew(
        &self,
        job: &JobName,
        holder: &HolderId,
        ttl: Duration,
    ) -> Result<Renewal, Self::Error> {
        Ok(match self.invoke(&RENEW, job, holder, Some(ttl)).await? {
            true => Renewal::Extended,
            false => Renewal::Lost,
        })
    }

    async fn release(&self, job: &JobName, holder: &HolderId) -> Result<(), Self::Error> {
        self.invoke(&RELEASE, job, holder, None).await?;
        Ok(())
    }
}
