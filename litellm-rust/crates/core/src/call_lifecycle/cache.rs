use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use litellm_cache::{BaseCache, CacheControls, CacheEntry, CacheKwargs};
use serde_json::Value;

use super::admission::AdmissionDecline;

#[derive(Clone, Default)]
pub struct ResponseCachePlan {
    pub controls: CacheControls,
    pub key: String,
    pub backend: Option<Arc<dyn BaseCache<Value = CacheEntry>>>,
    pub ttl: Option<Duration>,
    pub max_age: Option<Duration>,
}

impl ResponseCachePlan {
    pub fn admit(&self) -> Result<(), AdmissionDecline> {
        if !self.controls.native_backend && (self.controls.reads() || self.controls.writes()) {
            return Err(AdmissionDecline::Feature(
                "response cache backend is not implemented in Rust",
            ));
        }
        Ok(())
    }
}

pub async fn lookup(plan: &ResponseCachePlan) -> Option<Value> {
    let backend = plan.backend.as_ref().filter(|_| plan.controls.reads())?;
    let entry = backend
        .async_get_cache(&plan.key, &CacheKwargs::default())
        .await
        .ok()??;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    if !entry.fresh(now, plan.max_age) {
        return None;
    }
    match entry.response {
        Value::String(text) => serde_json::from_str(&text).ok(),
        value => Some(value),
    }
}

pub async fn store(plan: &ResponseCachePlan, value: Option<Value>) {
    let (Some(value), Some(backend)) = (
        value,
        plan.backend.as_ref().filter(|_| plan.controls.writes()),
    ) else {
        return;
    };
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();
    let _ = backend
        .async_set_cache(
            &plan.key,
            CacheEntry {
                timestamp,
                response: value,
            },
            CacheKwargs {
                ttl: plan.ttl,
                ..Default::default()
            },
        )
        .await;
}
