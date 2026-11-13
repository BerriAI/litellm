mod keys;
mod types;

use std::{sync::Arc, time::Duration};

use litellm_redis_client::commands;

pub use self::types::{HealthState, HealthStates};
use crate::{
    Error, codec,
    dual_cache::{DualCache, MemorySettings},
    memory::{Clock, system_clock},
    remote::{Remote, RemoteSlot},
};

/// The background health checks' per-deployment states, kept by `DeploymentHealthCache` in the
/// router's `DualCache`: memory per worker, Redis shared by every pod. The writer's merge and the
/// readers' staleness rules stay with the host.
pub struct DeploymentHealth {
    states: DualCache<HealthStates>,
}

impl DeploymentHealth {
    pub fn new(remote: Option<Remote>, default_in_memory_ttl: Option<Duration>) -> Self {
        Self::with_clock(remote, default_in_memory_ttl, system_clock())
    }

    pub fn with_clock(
        remote: Option<Remote>,
        default_in_memory_ttl: Option<Duration>,
        now: Clock,
    ) -> Self {
        Self {
            states: DualCache::new(
                Arc::new(RemoteSlot::new(remote)),
                MemorySettings {
                    entries: None,
                    ttl: default_in_memory_ttl,
                },
                now,
            ),
        }
    }

    /// `DualCache.attach_redis_cache`: memory stays, and `false` means a Redis was already attached.
    pub fn attach(&self, remote: Remote) -> bool {
        self.states.redis().attach(remote)
    }

    pub fn has_redis(&self) -> bool {
        self.states.redis().get().is_some()
    }

    /// `DualCache.get_cache`: memory, else Redis written back to memory.
    pub async fn get(&self) -> Result<Option<HealthStates>, Error> {
        self.states.get(None, keys::STATES, None).await
    }

    /// `redis_cache.get_cache`, what every pod merged so far. No Redis, a failed read and an open
    /// breaker all read as `None`.
    pub async fn get_shared(&self) -> Option<HealthStates> {
        let client = self.states.redis().client()?;
        client
            .send(commands::get::<Vec<u8>>(keys::STATES))
            .await
            .map_err(Error::from)
            .and_then(|raw| Ok(raw.as_deref().map(codec::decode).transpose()?))
            .unwrap_or_else(|error| {
                crate::failure_log::redis_failure!(&error, "LiteLLM Cache: exception in get_cache");
                None
            })
    }

    /// `DualCache.set_cache`, whose sync `RedisCache.set_cache` writes `str(value)`.
    pub async fn set(&self, states: HealthStates, ttl: Duration) -> Result<(), Error> {
        self.states.set_str(keys::STATES, states, Some(ttl)).await
    }
}
