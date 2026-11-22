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

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::RouterRedis;

    #[rstest]
    fn debug_shows_where_the_redis_is_and_none_of_its_credentials() {
        let redis: RouterRedis = serde_json::from_value(json!({
            "redis_url": "redis://user:url-credential-value@router:6390",
            "redis_host": "router",
            "redis_password": "password-credential-value",
            "cache_kwargs": {"sentinel_password": "kwarg-credential-value"},
        }))
        .unwrap();

        let shown = format!("{redis:?}");

        assert!(!shown.contains("credential-value"), "{shown}");
        assert!(shown.contains("[REDACTED]"), "{shown}");
        assert!(shown.contains("router"), "{shown}");
    }
}
