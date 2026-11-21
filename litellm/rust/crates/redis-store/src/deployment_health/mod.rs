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

    /// `DualCache.set_cache`, whose sync `RedisCache.set_cache` writes `str(value)`. The background
    /// health checks write every cycle, so each write restarts the deadline in memory as it does in
    /// Redis. Left on its first deadline, the entry would expire `ttl` after the first write and the
    /// router would see no health states, so every deployment would read as healthy, until the
    /// next write.
    pub async fn set(&self, states: HealthStates, ttl: Duration) -> Result<(), Error> {
        self.states.refresh_str(keys::STATES, states, ttl).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use rstest::rstest;

    use super::*;
    use crate::Number;

    const TTL: Duration = Duration::from_secs(270);

    fn states() -> HealthStates {
        HealthStates(vec![(
            "dep-a".to_owned(),
            HealthState {
                is_healthy: false,
                timestamp: Number::Float(1.5),
                reason: "background_health_check_failed".to_owned(),
            },
        )])
    }

    /// A store with no Redis whose clock reads whatever `at` last held.
    fn store_at(at: &Arc<Mutex<Duration>>) -> DeploymentHealth {
        let now = Arc::clone(at);
        DeploymentHealth::with_clock(None, None, Arc::new(move || *now.lock().unwrap()))
    }

    fn secs(seconds: u64) -> Duration {
        Duration::from_secs(seconds)
    }

    /// The health checks rewrite the states every cycle, so a write to a key that is still live
    /// has to push its deadline out. Otherwise the states vanish `ttl` after the first write, and
    /// until the next cycle the router sees every deployment as healthy, dead ones included.
    #[rstest]
    #[case::rewritten_before_the_first_deadline(&[0, 200], 300, true)]
    #[case::read_in_the_gap_before_the_next_cycle(&[0, 70, 140, 210], 275, true)]
    #[case::never_rewritten(&[0], 300, false)]
    #[case::rewritten_but_the_new_deadline_passed_too(&[0, 200], 500, false)]
    #[tokio::test]
    async fn a_rewrite_moves_the_deadline_of_the_states(
        #[case] writes_at: &[u64],
        #[case] read_at: u64,
        #[case] held: bool,
    ) {
        let at = Arc::new(Mutex::new(Duration::ZERO));
        let health = store_at(&at);

        for write_at in writes_at {
            *at.lock().unwrap() = secs(*write_at);
            health.set(states(), TTL).await.unwrap();
        }
        *at.lock().unwrap() = secs(read_at);

        assert_eq!(health.get().await.unwrap(), held.then(states));
    }
}
