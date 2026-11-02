use litellm_redis_types::ClusterNode as RedisNode;
use litellm_secret_value::SecretValue;

use super::{
    Expected, Kwargs, Lookup, RedisError, as_text, configured, invalid, keys, node_address, nodes,
    text_or_fallback,
};
use crate::Value;

pub(super) struct SentinelTarget {
    pub(super) sentinels: Vec<RedisNode>,
    pub(super) service_name: String,
    pub(super) sentinel_password: Option<SecretValue>,
}

pub(super) fn sentinel_target(
    merged: &Kwargs,
    lookup: &dyn Lookup,
) -> Result<Option<SentinelTarget>, RedisError> {
    let (nodes, source) = match (
        configured(merged, "sentinel_nodes"),
        keys::fallback("sentinel_nodes", lookup),
        merged
            .get("sentinel_nodes")
            .filter(|entry| !entry.value.is_null()),
    ) {
        (Some(entry), _, _) => (sentinel_nodes(&entry.value), entry.source.as_str()),
        (None, Some((source, json)), _) => (sentinel_nodes(&Value::String(json)), source),
        (None, None, Some(empty)) => (None, empty.source.as_str()),
        (None, None, None) => return Ok(None),
    };
    let nodes = nodes.ok_or_else(|| invalid(source, Expected::SentinelNodes))?;
    let service_name =
        text_or_fallback(merged, "service_name", lookup).ok_or(RedisError::NoSentinelService)?;
    Ok(Some(SentinelTarget {
        sentinels: nodes,
        service_name,
        sentinel_password: text_or_fallback(merged, "sentinel_password", lookup)
            .map(SecretValue::new),
    }))
}

fn sentinel_nodes(value: &Value) -> Option<Vec<RedisNode>> {
    let nodes = nodes(value, |pair| match pair {
        Value::Sequence(pair) if pair.len() == 2 => {
            node_address(as_text(&pair[0]), as_text(&pair[1]))
        }
        _ => None,
    })?;
    (!nodes.is_empty()).then_some(nodes)
}
