use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use super::hash::HashedToken;
use super::types::KeyObject;

struct CacheEntry {
    key_object: Arc<KeyObject>,
    inserted_at: Instant,
}

/// In-memory cache for API key auth objects with TTL expiry.
///
/// Thread-safe via `RwLock`. Uses lazy eviction: expired entries are reaped
/// on access and when the cache is at capacity.
///
/// Keyed by raw `[u8; 32]` digest (from `HashedToken`) to avoid String allocation.
/// Values are `Arc<KeyObject>` so `get()` returns a cheap Arc clone instead of
/// deep-cloning the entire KeyObject (which contains ~10 String fields and 2 HashSets).
pub struct KeyCache {
    entries: RwLock<HashMap<[u8; 32], CacheEntry>>,
    ttl: Duration,
    max_size: usize,
}

impl KeyCache {
    pub fn new(ttl: Duration, max_size: usize) -> Self {
        Self {
            entries: RwLock::new(HashMap::with_capacity(max_size.min(1024))),
            ttl,
            max_size,
        }
    }

    /// Look up a key by its hashed token. Returns `Arc<KeyObject>` (cheap clone).
    pub fn get(&self, hashed: &HashedToken) -> Option<Arc<KeyObject>> {
        let entries = self.entries.read().ok()?;
        let entry = entries.get(hashed.as_raw())?;

        if entry.inserted_at.elapsed() > self.ttl {
            return None;
        }

        Some(Arc::clone(&entry.key_object))
    }

    /// Store a key object. Takes `Arc<KeyObject>` to avoid an extra allocation
    /// when the caller already has one (e.g., from a DB lookup wrapped in Arc).
    pub fn set(&self, hashed: HashedToken, key_object: Arc<KeyObject>) {
        let mut entries = match self.entries.write() {
            Ok(entries) => entries,
            Err(_) => return,
        };

        if entries.len() >= self.max_size {
            self.evict_expired(&mut entries);
        }

        if entries.len() >= self.max_size {
            self.evict_oldest(&mut entries);
        }

        entries.insert(
            *hashed.as_raw(),
            CacheEntry {
                key_object,
                inserted_at: Instant::now(),
            },
        );
    }

    pub fn remove(&self, hashed: &HashedToken) {
        if let Ok(mut entries) = self.entries.write() {
            entries.remove(hashed.as_raw());
        }
    }

    pub fn len(&self) -> usize {
        self.entries.read().map(|e| e.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn evict_expired(&self, entries: &mut HashMap<[u8; 32], CacheEntry>) {
        entries.retain(|_, entry| entry.inserted_at.elapsed() <= self.ttl);
    }

    fn evict_oldest(&self, entries: &mut HashMap<[u8; 32], CacheEntry>) {
        if let Some(oldest_key) = entries
            .iter()
            .min_by_key(|(_, entry)| entry.inserted_at)
            .map(|(key, _)| *key)
        {
            entries.remove(&oldest_key);
        }
    }
}

impl Default for KeyCache {
    fn default() -> Self {
        Self::new(Duration::from_secs(600), 10_000)
    }
}
