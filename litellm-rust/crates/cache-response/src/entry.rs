use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CacheEntry {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<f64>,
    pub response: Value,
}

impl CacheEntry {
    pub fn produced_at(response: Value, produced_at: Duration) -> Self {
        Self {
            timestamp: Some(produced_at.as_secs_f64()),
            response,
        }
    }

    pub fn fresh(&self, now: Duration, max_age: Option<Duration>) -> bool {
        self.timestamp.is_none_or(|timestamp| {
            timestamp.is_finite()
                && max_age.is_none_or(|age| now.as_secs_f64() - timestamp <= age.as_secs_f64())
        })
    }

    pub fn into_fresh(self, now: Duration, max_age: Option<Duration>) -> Option<Value> {
        self.fresh(now, max_age).then_some(self.response)
    }
}
