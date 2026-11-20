use std::fmt;

use litellm_secret_value::SecretValue;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::Spelled;

/// The Redis a router's `redis_*` and `cache_kwargs` settings name for its routing state.
#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub struct RouterRedis {
    pub redis_url: Option<SecretValue>,
    pub redis_host: Option<String>,
    pub redis_port: Option<Spelled<i64>>,
    pub redis_password: Option<SecretValue>,
    pub redis_db: Option<Spelled<i64>>,
    pub cache_kwargs: Option<Map<String, Value>>,
}

impl fmt::Debug for RouterRedis {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RouterRedis")
            .field("redis_host", &self.redis_host)
            .field("redis_port", &self.redis_port)
            .field("redis_db", &self.redis_db)
            .finish_non_exhaustive()
    }
}
