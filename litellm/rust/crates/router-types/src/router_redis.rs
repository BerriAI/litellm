use litellm_secret_value::SecretValue;
use serde::Deserialize;
use serde_json::{Map, Value};
use veil::Redact;

use crate::Spelled;

/// The Redis a router's `redis_*` and `cache_kwargs` settings name for its routing state.
#[derive(Clone, Default, Deserialize, Redact)]
#[serde(default)]
pub struct RouterRedis {
    pub redis_url: Option<SecretValue>,
    pub redis_host: Option<String>,
    pub redis_port: Option<Spelled<i64>>,
    pub redis_password: Option<SecretValue>,
    pub redis_db: Option<Spelled<i64>>,
    #[redact(with = "[REDACTED]")]
    pub cache_kwargs: Option<Map<String, Value>>,
}
