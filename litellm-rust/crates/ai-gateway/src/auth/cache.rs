//! A bounded, in-memory cache of verified keys — **off by default (strict).**
//!
//! The control-plane verifier is where per-request budget, block, and rate-limit
//! checks run, so caching its result lets a caller reuse a cached identity for the
//! cache window without those checks — a budget/rate-limit bypass on billable
//! connections. The current route (realtime) authenticates **once per long-lived
//! connection**, not per request, so re-verifying every connection is cheap (one
//! call per voice session, nowhere near a per-request 10k-RPS load). So the default
//! is strict: every connection re-verifies.
//!
//! Caching is **opt-in** via `LITELLM_AUTH_CACHE_TTL_SECS > 0`, for future
//! high-RPS *per-request* routes where the verify endpoint would otherwise be a
//! bottleneck. Enabling it trades budget/rate-limit freshness (bounded by the TTL)
//! for fewer control-plane calls — like the proxy's own ~60s auth cache.
//!
//! Bounds when enabled: at most [`MAX_ENTRIES`] entries (FIFO-evicted) plus TTL
//! expiry. Std-only: a `Mutex<HashMap>` for lookup plus a `VecDeque` for FIFO order.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::auth::UserApiKeyAuth;

/// Hard cap on cached entries; the 201st insert evicts the oldest. Keeps memory
/// bounded regardless of how many distinct keys hit the gateway.
const MAX_ENTRIES: usize = 200;
/// Default TTL when `LITELLM_AUTH_CACHE_TTL_SECS` is unset/invalid. **0 = caching
/// off** → every connection re-verifies (budget/block/rate-limit enforced each
/// time). Opt into caching by setting this `> 0` for high-RPS per-request routes.
const DEFAULT_TTL_SECS: u64 = 0;
/// Env var overriding the entry TTL (seconds). Set `> 0` to enable caching.
const TTL_ENV: &str = "LITELLM_AUTH_CACHE_TTL_SECS";

struct Inner {
    /// hash → (identity, insertion time). Insertion time drives TTL expiry.
    entries: HashMap<[u8; 32], (UserApiKeyAuth, Instant)>,
    /// Insertion order, oldest first, for FIFO eviction.
    order: VecDeque<[u8; 32]>,
}

/// Bounded TTL cache keyed by the SHA-256 of the API key.
pub struct KeyCache {
    inner: Mutex<Inner>,
    ttl: Duration,
}

impl KeyCache {
    /// Build with the TTL from `LITELLM_AUTH_CACHE_TTL_SECS` (falling back to
    /// [`DEFAULT_TTL_SECS`] when unset or unparseable).
    pub fn new() -> Self {
        let ttl_secs = std::env::var(TTL_ENV)
            .ok()
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_TTL_SECS);
        Self::with_ttl(Duration::from_secs(ttl_secs))
    }

    /// Build with an explicit TTL (used by tests for tiny windows).
    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            inner: Mutex::new(Inner {
                entries: HashMap::new(),
                order: VecDeque::new(),
            }),
            ttl,
        }
    }

    /// Return the cached identity for `hash`, or `None`. A present-but-expired
    /// entry is evicted and treated as a miss, so the caller re-verifies.
    pub fn get(&self, hash: &[u8; 32]) -> Option<UserApiKeyAuth> {
        // TTL 0 ⇒ caching disabled: always a miss, so every connection re-verifies.
        if self.ttl.is_zero() {
            return None;
        }
        let mut inner = self.inner.lock().expect("auth cache mutex poisoned");
        let expired = match inner.entries.get(hash) {
            Some((_, inserted)) => inserted.elapsed() >= self.ttl,
            None => return None,
        };
        if expired {
            // Drop the entry but leave its (now-stale) `order` slot — purging it
            // here would be an O(n) scan on the hot `get` path under the lock.
            // The stale slot is reconciled on the next `insert` (see below).
            inner.entries.remove(hash);
            return None;
        }
        inner.entries.get(hash).map(|(auth, _)| auth.clone())
    }

    /// Insert (or refresh) an entry, evicting the oldest if at capacity.
    pub fn insert(&self, hash: [u8; 32], auth: UserApiKeyAuth) {
        // TTL 0 ⇒ caching disabled: don't store anything.
        if self.ttl.is_zero() {
            return;
        }
        let mut inner = self.inner.lock().expect("auth cache mutex poisoned");

        // Reconcile `order` so it tracks exactly the live entries: drop this key's
        // prior slot (refresh) AND any stale slots left by expiry in `get`. This is
        // the O(n) pass — kept off the hot `get` path and onto the cache-miss path.
        // Take `order` out so the closure can read `entries` without aliasing.
        let mut order = std::mem::take(&mut inner.order);
        order.retain(|h| h != &hash && inner.entries.contains_key(h));
        inner.order = order;

        while inner.entries.len() >= MAX_ENTRIES {
            match inner.order.pop_front() {
                Some(oldest) => {
                    inner.entries.remove(&oldest);
                }
                None => break,
            }
        }

        inner.entries.insert(hash, (auth, Instant::now()));
        inner.order.push_back(hash);
    }
}

impl Default for KeyCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    #[test]
    fn get_returns_inserted_entry() {
        let cache = KeyCache::with_ttl(Duration::from_secs(60));
        let auth = UserApiKeyAuth {
            user_id: Some("u".to_string()),
            ..UserApiKeyAuth::default()
        };
        cache.insert(hash(1), auth);

        let got = cache.get(&hash(1)).expect("entry present");
        assert_eq!(got.user_id.as_deref(), Some("u"));
        assert!(cache.get(&hash(2)).is_none());
    }

    #[test]
    fn evicts_oldest_at_capacity() {
        let cache = KeyCache::with_ttl(Duration::from_secs(60));
        // Fill exactly to capacity.
        for i in 0..MAX_ENTRIES {
            cache.insert(hash(i as u8), UserApiKeyAuth::default());
        }
        assert!(cache.get(&hash(0)).is_some());

        // The 201st distinct insert evicts the oldest (hash(0)).
        cache.insert(hash(255), UserApiKeyAuth::default());
        assert!(cache.get(&hash(0)).is_none());
        assert!(cache.get(&hash(255)).is_some());
        // A middle entry survives.
        assert!(cache.get(&hash(1)).is_some());
    }

    #[test]
    fn get_returns_none_after_ttl() {
        let cache = KeyCache::with_ttl(Duration::from_millis(10));
        cache.insert(hash(1), UserApiKeyAuth::default());
        std::thread::sleep(Duration::from_millis(25));
        assert!(cache.get(&hash(1)).is_none());
    }

    #[test]
    fn ttl_zero_disables_caching() {
        // The default: every lookup is a miss, so callers always re-verify.
        let cache = KeyCache::with_ttl(Duration::from_secs(0));
        cache.insert(hash(1), UserApiKeyAuth::default());
        assert!(cache.get(&hash(1)).is_none());
    }
}
