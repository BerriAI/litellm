use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};
use veil::Redact;

use crate::Credentials;

const CREDENTIAL_FETCH_LOCK_STRIPES: usize = 64;

struct CredentialCache {
    entries: HashMap<CredentialScope, (Credentials, Duration)>,
    expirations: BinaryHeap<Reverse<(Duration, CredentialScope)>>,
    max_entries: usize,
}

impl Default for CredentialCache {
    fn default() -> Self {
        Self::new(200)
    }
}

impl CredentialCache {
    fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            expirations: BinaryHeap::new(),
            max_entries: max_entries.max(1),
        }
    }

    fn get(&mut self, key: &CredentialScope, now: Duration) -> Option<Credentials> {
        let (credentials, expiration) = self.entries.get(key)?;
        if *expiration > now {
            return Some(credentials.clone());
        }
        self.entries.remove(key);
        None
    }

    fn insert(
        &mut self,
        key: CredentialScope,
        credentials: Credentials,
        ttl: Duration,
        now: Duration,
    ) {
        while let Some(Reverse((expiration, key))) = self.expirations.peek().cloned() {
            if self.entries.get(&key).map(|(_, current)| *current) != Some(expiration) {
                self.expirations.pop();
            } else if expiration <= now || self.entries.len() >= self.max_entries {
                self.expirations.pop();
                self.entries.remove(&key);
            } else {
                break;
            }
        }
        let expiration = now + ttl;
        self.entries.insert(key.clone(), (credentials, expiration));
        self.expirations.push(Reverse((expiration, key)));
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Redact)]
pub struct CredentialScope(#[redact(fixed = 8)] [u8; 32]);

impl CredentialScope {
    pub fn from_optional_values<'a>(
        namespace: &str,
        values: impl IntoIterator<Item = Option<&'a str>>,
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(namespace.len().to_le_bytes());
        hasher.update(namespace.as_bytes());
        for value in values {
            match value {
                Some(value) => {
                    hasher.update([1]);
                    hasher.update(value.len().to_le_bytes());
                    hasher.update(value.as_bytes());
                }
                None => hasher.update([0]),
            }
        }
        Self(hasher.finalize().into())
    }
}

pub trait Clock: Send + Sync {
    fn now(&self) -> Duration;
}

impl<T> Clock for Arc<T>
where
    T: Clock + ?Sized,
{
    fn now(&self) -> Duration {
        (**self).now()
    }
}

#[derive(Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Duration {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
    }
}

pub struct CredentialState<R, C = SystemClock> {
    runtime: R,
    clock: C,
    cache: Mutex<CredentialCache>,
    fetch_locks: Box<[tokio::sync::Mutex<()>]>,
}

impl<R> CredentialState<R, SystemClock> {
    pub fn new(runtime: R, max_entries: usize) -> Self {
        Self::with_clock(runtime, max_entries, SystemClock)
    }
}

impl<R, C> CredentialState<R, C>
where
    C: Clock,
{
    pub fn with_clock(runtime: R, max_entries: usize, clock: C) -> Self {
        Self {
            runtime,
            clock,
            cache: Mutex::new(CredentialCache::new(max_entries)),
            fetch_locks: (0..CREDENTIAL_FETCH_LOCK_STRIPES)
                .map(|_| tokio::sync::Mutex::new(()))
                .collect(),
        }
    }

    pub fn runtime(&self) -> &R {
        &self.runtime
    }

    pub async fn get_or_acquire<E, F, Fut>(
        &self,
        scope: CredentialScope,
        ttl: Duration,
        acquire: F,
    ) -> Result<Credentials, E>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Credentials, E>>,
    {
        let mut stripe_hasher = std::collections::hash_map::DefaultHasher::new();
        scope.hash(&mut stripe_hasher);
        let stripe = stripe_hasher.finish() as usize % self.fetch_locks.len();
        let _fetch_guard = self.fetch_locks[stripe].lock().await;

        if let Some(credentials) = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&scope, self.clock.now())
        {
            return Ok(credentials);
        }

        let credentials = acquire().await?;
        self.cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(scope, credentials.clone(), ttl, self.clock.now());
        Ok(credentials)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    use super::*;
    use crate::static_credentials;

    #[test]
    fn cache_expiry_and_bounds_are_clock_driven() {
        let mut cache = CredentialCache::new(1);
        let first = CredentialScope::from_optional_values("test", [Some("first")]);
        let second = CredentialScope::from_optional_values("test", [Some("second")]);
        cache.insert(
            first.clone(),
            static_credentials("ak1", "sk1"),
            Duration::from_secs(10),
            Duration::ZERO,
        );
        assert_eq!(
            cache
                .get(&first, Duration::from_secs(9))
                .unwrap()
                .access_key_id(),
            "ak1"
        );
        cache.insert(
            second.clone(),
            static_credentials("ak2", "sk2"),
            Duration::from_secs(10),
            Duration::ZERO,
        );
        assert!(cache.get(&first, Duration::ZERO).is_none());
        assert!(cache.get(&second, Duration::from_secs(11)).is_none());
    }

    #[tokio::test]
    async fn credential_state_coordinates_concurrent_misses() {
        let state = CredentialState::new((), 1);
        let acquisitions = AtomicUsize::new(0);
        let scope = CredentialScope::from_optional_values("test", [Some("identity")]);
        let acquire = || async {
            acquisitions.fetch_add(1, Ordering::SeqCst);
            tokio::task::yield_now().await;
            Ok::<_, ()>(static_credentials("ak", "sk"))
        };
        let (first, second) = tokio::join!(
            state.get_or_acquire(scope.clone(), Duration::from_secs(10), acquire),
            state.get_or_acquire(scope, Duration::from_secs(10), acquire),
        );

        assert_eq!(first.unwrap(), second.unwrap());
        assert_eq!(acquisitions.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn credential_state_reacquires_at_the_injected_expiry_boundary() {
        struct TestClock(Arc<AtomicU64>);

        impl Clock for TestClock {
            fn now(&self) -> Duration {
                Duration::from_secs(self.0.load(Ordering::SeqCst))
            }
        }

        let now = Arc::new(AtomicU64::new(0));
        let state = CredentialState::with_clock((), 1, TestClock(now.clone()));
        let acquisitions = AtomicUsize::new(0);
        let scope = CredentialScope::from_optional_values("test", [Some("identity")]);
        let acquire = || async {
            acquisitions.fetch_add(1, Ordering::SeqCst);
            Ok::<_, ()>(static_credentials("ak", "sk"))
        };

        state
            .get_or_acquire(scope.clone(), Duration::from_secs(10), acquire)
            .await
            .unwrap();
        now.store(9, Ordering::SeqCst);
        state
            .get_or_acquire(scope.clone(), Duration::from_secs(10), acquire)
            .await
            .unwrap();
        now.store(10, Ordering::SeqCst);
        state
            .get_or_acquire(scope, Duration::from_secs(10), acquire)
            .await
            .unwrap();

        assert_eq!(acquisitions.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn credential_scope_does_not_expose_key_material() {
        let scope = CredentialScope::from_optional_values(
            "test",
            [Some("visible-id"), Some("never-print-secret"), None],
        );
        let debug = format!("{scope:?}");
        assert!(!debug.contains("visible-id"));
        assert!(!debug.contains("never-print-secret"));
    }
}
