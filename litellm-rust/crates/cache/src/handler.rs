use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::{CacheBackend, CacheControls, CacheEntry, CacheKwargs};

#[derive(Clone)]
pub struct LLMCachingHandler {
    pub backend: CacheBackend,
    pub controls: CacheControls,
    pub key: String,
    pub kwargs: CacheKwargs,
    pub max_age: Option<Duration>,
}

impl LLMCachingHandler {
    pub async fn get(&self) -> Option<Value> {
        if !self.controls.reads() {
            return None;
        }
        let entry = self
            .backend
            .async_get_cache(&self.key, &self.kwargs)
            .await
            .ok()??;
        entry.fresh(now(), self.max_age).then_some(entry.response)
    }

    pub async fn set(&self, response: Value) {
        if !self.controls.writes() {
            return;
        }
        let entry = CacheEntry {
            timestamp: now().as_secs_f64(),
            response,
        };
        let _ = self
            .backend
            .async_set_cache(&self.key, entry, self.kwargs.clone())
            .await;
    }
}

fn now() -> Duration {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
}
