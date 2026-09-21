use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::CacheContext;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SemanticCacheScope {
    #[default]
    Key,
    EndUser,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SemanticCacheContext {
    pub input: Option<String>,
    pub messages: Vec<Value>,
    pub metadata: Map<String, Value>,
    pub scope: SemanticCacheScope,
    pub ttl: Option<Duration>,
}

impl CacheContext for SemanticCacheContext {
    fn ttl(&self) -> Option<Duration> {
        self.ttl
    }

    fn with_ttl(&self, ttl: Option<Duration>) -> Self {
        Self {
            ttl,
            ..self.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_ttl_keeps_request_fields() {
        let context = SemanticCacheContext {
            input: Some("query".to_owned()),
            messages: vec![serde_json::json!({"role": "user", "content": "hi"})],
            metadata: Map::from_iter([("user".to_owned(), Value::from("u1"))]),
            scope: SemanticCacheScope::EndUser,
            ttl: None,
        };

        let updated = context.with_ttl(Some(Duration::from_secs(5)));

        assert_eq!(updated.ttl(), Some(Duration::from_secs(5)));
        assert_eq!(updated.input, context.input);
        assert_eq!(updated.messages, context.messages);
        assert_eq!(updated.metadata, context.metadata);
        assert_eq!(updated.scope, SemanticCacheScope::EndUser);
    }

    #[test]
    fn scope_serializes_like_python_cache_scope() {
        assert_eq!(
            serde_json::to_value(SemanticCacheScope::EndUser).unwrap(),
            Value::from("end_user")
        );
        assert_eq!(
            serde_json::from_value::<SemanticCacheScope>(Value::from("key")).unwrap(),
            SemanticCacheScope::Key
        );
    }
}
