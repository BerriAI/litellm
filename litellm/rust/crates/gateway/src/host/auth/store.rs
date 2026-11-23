//! A virtual key read through the auth cache, as the proxy's `get_key_object` reads it: the
//! cache, then the database on a miss, which caches the key as `cache_key_object` does, for every
//! pod when the cache has the shared Redis attached. Nothing guards that write against an eviction
//! that lands during the database read, as nothing guards Python's.
//!
//! A key's team is read the same way, as `get_team_object` reads it, except that a team read from
//! the database is kept in this process's memory alone. The entry a Python pod wrote is read from
//! the shared Redis, but none is written there until the bytes this host would write are pinned
//! against Python's on both sides, as a key's are.

use std::{sync::Arc, time::Duration};

use litellm_auth_store::{Error, KeyDatabase, TeamDatabase};
use litellm_gateway_auth_core::KeyHash;
use litellm_redis_store::{
    FromValue, IntoValue,
    auth_cache::{AuthCache, AuthObject, CachedKey, CachedTeam},
};
use litellm_repositories::Side;

/// How long Python keeps a key or a team it cached, its `DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL`, and
/// the `default_in_memory_ttl` its `user_api_key_cache` keeps a Redis hit for.
pub(crate) const KEY_TTL: Duration = Duration::from_secs(60);

pub struct AuthStore {
    cache: Arc<AuthCache>,
    keys: KeyDatabase,
    teams: TeamDatabase,
}

impl AuthStore {
    pub fn new(cache: Arc<AuthCache>, keys: KeyDatabase, teams: TeamDatabase) -> Self {
        Self { cache, keys, teams }
    }

    /// What the cache holds for `object`, decoded. A cache that fails and an entry that does not
    /// decode are both a miss, as they are to `UserApiKeyCache`.
    async fn held<T: FromValue>(&self, object: &AuthObject) -> Option<T> {
        let value = self
            .cache
            .get(None, object)
            .await
            .inspect_err(|error| {
                tracing::warn!(
                    error = %litellm_error::Report::new(error),
                    "an auth object was not read from the auth cache"
                );
            })
            .ok()
            .flatten()?;
        T::from_value(&value)
            .inspect_err(|error| {
                tracing::warn!(
                    error = %litellm_error::Report::new(error),
                    "a cached auth object does not decode and is read as a miss"
                );
            })
            .ok()
    }

    /// The team `team_id`: the cache's, else the database's, which this process then keeps for
    /// as long as Python keeps a team. `None` when no row has that id.
    pub async fn team(&self, team_id: &str) -> Result<Option<CachedTeam>, Error> {
        let object = AuthObject::Team(team_id.to_owned());
        if let Some(team) = self.held(&object).await {
            return Ok(Some(team));
        }
        let Some(team) = Box::pin(self.teams.team(team_id, Side::Read)).await? else {
            return Ok(None);
        };
        let kept = self
            .cache
            .set_local(&object, team.clone().into_value(), Some(KEY_TTL));
        if let Err(error) = kept {
            tracing::warn!(
                error = %litellm_error::Report::new(&error),
                "a team was not cached"
            );
        }
        Ok(Some(team))
    }

    /// The key the cache holds for `hash`. A cache that fails and an entry that does not decode
    /// are both a miss, as they are to `UserApiKeyCache`.
    pub async fn cached(&self, hash: &KeyHash) -> Option<CachedKey> {
        self.held(&object(hash)).await
    }

    /// The key the database holds for `hash`, cached. A key the cache did not take
    /// is returned all the same.
    pub async fn loaded(&self, hash: &KeyHash) -> Result<Option<CachedKey>, Error> {
        let Some(key) = Box::pin(self.keys.read(hash, Side::Read)).await? else {
            return Ok(None);
        };
        let value = key.clone().into_value();
        if let Err(error) = self
            .cache
            .set(None, &object(hash), value, Some(KEY_TTL))
            .await
        {
            tracing::warn!(
                error = %litellm_error::Report::new(&error),
                "a key was not cached"
            );
        }
        Ok(Some(key))
    }
}

fn object(hash: &KeyHash) -> AuthObject {
    AuthObject::Key(hash.as_str().to_owned())
}
