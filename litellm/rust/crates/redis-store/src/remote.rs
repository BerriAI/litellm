use std::{sync::OnceLock, time::Duration};

use litellm_redis_client::{RedisBatch, RedisClient};

use crate::Send;

/// The Redis a store shares with every pod, with the `RedisCache.default_ttl` a write takes when it
/// names none.
#[derive(Clone)]
pub struct Remote {
    pub client: RedisClient,
    pub default_ttl: Duration,
}

/// A store's Redis, absent until the host attaches one, as `DualCache.attach_redis_cache` leaves
/// memory in place and only adds the second tier. It is attached at most once.
pub(crate) struct RemoteSlot(OnceLock<Remote>);

impl RemoteSlot {
    pub(crate) fn new(remote: Option<Remote>) -> Self {
        Self(remote.map(OnceLock::from).unwrap_or_default())
    }

    /// `false` when a Redis was already attached, which keeps it.
    pub(crate) fn attach(&self, remote: Remote) -> bool {
        self.0.set(remote).is_ok()
    }

    pub(crate) fn get(&self) -> Option<&Remote> {
        self.0.get()
    }

    pub(crate) fn client(&self) -> Option<&RedisClient> {
        self.get().map(|remote| &remote.client)
    }

    pub(crate) fn batch(&self, send: Send<'_>) -> Option<RedisBatch> {
        Some(send.batch(self.client()?))
    }
}
