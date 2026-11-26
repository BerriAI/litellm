//! A virtual key read through the auth cache, as the proxy's `get_key_object` reads it: the
//! cache, then the database on a miss, which caches the key as `cache_key_object` does, for every
//! pod when the cache has the shared Redis attached. Nothing guards that write against an eviction
//! that lands during the database read, as nothing guards Python's.

use std::sync::Arc;

use litellm_auth_store::{Error, KeyDatabase};
use litellm_gateway_auth_core::KeyHash;
use litellm_redis_store::{
    FromValue, IntoValue,
    auth_cache::{AuthCache, AuthObject, CachedKey, Settings},
};
use litellm_repositories::Side;

pub struct AuthStore {
    cache: Arc<AuthCache>,
    keys: KeyDatabase,
}

impl AuthStore {
    pub fn new(cache: Arc<AuthCache>, keys: KeyDatabase) -> Self {
        Self { cache, keys }
    }

    /// The key the cache holds for `hash`. A cache that fails and an entry that does not decode
    /// are both a miss, as they are to `UserApiKeyCache`.
    pub async fn cached(&self, hash: &KeyHash) -> Option<CachedKey> {
        let value = self
            .cache
            .get(None, &object(hash))
            .await
            .inspect_err(|error| {
                tracing::warn!(
                    error = %litellm_error::Report::new(error),
                    "a key was not read from the auth cache"
                );
            })
            .ok()
            .flatten()?;
        CachedKey::from_value(&value)
            .inspect_err(|error| {
                tracing::warn!(
                    error = %litellm_error::Report::new(error),
                    "a cached key does not decode and is read as a miss"
                );
            })
            .ok()
    }

    /// The key the database holds for `hash`, cached for the cache's own TTL, as
    /// `get_management_object_ttl` names `default_in_memory_ttl`. A key the cache did not take is
    /// returned all the same.
    pub async fn loaded(&self, hash: &KeyHash) -> Result<Option<CachedKey>, Error> {
        let Some(key) = Box::pin(self.keys.read(hash, Side::Read)).await? else {
            return Ok(None);
        };
        let value = key.clone().into_value();
        if let Err(error) = self.cache.set(None, &object(hash), value, None).await {
            tracing::warn!(
                error = %litellm_error::Report::new(&error),
                "a key was not cached"
            );
        }
        Ok(Some(key))
    }

    /// `_resize_the_auth_cache` on a reload: `entries` in each partition from the next write on,
    /// with the TTL startup set kept.
    pub(crate) fn resize(&self, entries: usize) {
        self.cache.reconfigure(Settings {
            memory_entries: entries,
            ..self.cache.settings()
        });
    }
}

fn object(hash: &KeyHash) -> AuthObject {
    AuthObject::Key(hash.as_str().to_owned())
}
