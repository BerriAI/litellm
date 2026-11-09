use std::time::Duration;

use litellm_python_compat::Value;
use moka::sync::Cache;

use super::{Settings, is_key_object};

const MAX_AUTH_TTL: Duration = Duration::from_secs(60);

#[derive(Clone)]
struct Entry {
    value: Value,
    expires: Duration,
}

pub(super) struct Local {
    objects: Cache<String, Entry>,
    keys: Cache<String, Entry>,
    pub generation: u64,
    pub healthy: bool,
}

impl Local {
    pub fn new(settings: Settings) -> Self {
        let partition = || {
            Cache::builder()
                .max_capacity(settings.memory_entries as u64)
                .build()
        };
        Self {
            objects: partition(),
            keys: partition(),
            generation: 0,
            healthy: true,
        }
    }

    fn partition(&self, key: &str) -> &Cache<String, Entry> {
        if is_key_object(key) {
            &self.keys
        } else {
            &self.objects
        }
    }

    pub fn get(&self, key: &str, now: Duration) -> Option<Value> {
        if !self.healthy {
            return None;
        }
        let entry = self.partition(key).get(key)?;
        if now >= entry.expires {
            self.partition(key).invalidate(key);
            return None;
        }
        Some(entry.value)
    }

    pub fn set(&self, key: &str, value: Value, ttl: Option<Duration>, now: Duration) {
        if !self.healthy || key.starts_with("sk-") {
            return;
        }
        let ttl = ttl.unwrap_or(Duration::from_secs(600));
        let ttl = if is_key_object(key) {
            ttl.min(MAX_AUTH_TTL)
        } else {
            ttl
        };
        self.partition(key).insert(
            key.to_owned(),
            Entry {
                value,
                expires: now.saturating_add(ttl),
            },
        );
    }

    pub fn delete(&mut self, key: &str) {
        self.generation = self.generation.wrapping_add(1);
        self.partition(key).invalidate(key);
    }

    pub fn flush(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.keys.invalidate_all();
        self.objects.invalidate_all();
    }

    pub fn reconfigure(&mut self, settings: Settings) {
        let replacement = Self::new(settings);
        for (key, value) in &self.keys {
            replacement.keys.insert((*key).clone(), value);
        }
        for (key, value) in &self.objects {
            replacement.objects.insert((*key).clone(), value);
        }
        self.keys = replacement.keys;
        self.objects = replacement.objects;
    }

    pub fn len(&self) -> (usize, usize) {
        self.objects.run_pending_tasks();
        self.keys.run_pending_tasks();
        (
            self.objects.entry_count() as usize,
            self.keys.entry_count() as usize,
        )
    }
}
