use litellm_core_utils::settings::Lookup;

use super::{ConfiguredRedis, Nulls, keys, options::Kwargs};
use crate::Value;

#[derive(Clone)]
pub(super) struct Entry {
    pub(super) value: Value,
    pub(super) source: String,
}

const NAMED_BY_CACHE: &[&str] = &[
    "host",
    "port",
    "password",
    "socket_timeout",
    "startup_nodes",
    "namespace",
    "gcp_service_account",
    "gcp_ssl_ca_certs",
];

/// The configured settings over their `REDIS_*` values, `{**environment_kwargs, **env_overrides}`
/// in `_get_redis_client_logic`. A null left in the result is an explicit `None` that hid its
/// environment value.
pub(super) fn merged(configured: &ConfiguredRedis, lookup: &dyn Lookup) -> Kwargs {
    let block: Kwargs = configured
        .fields
        .iter()
        .map(|(key, value)| (key, environment_reference(value, lookup)))
        .filter(|(key, value)| !value.is_null() || keeps_null(configured.nulls, key))
        .map(|(key, value)| {
            (
                key.clone(),
                Entry {
                    value,
                    source: key.clone(),
                },
            )
        })
        .collect();
    let names_a_target = ["host", "startup_nodes", "sentinel_nodes"]
        .iter()
        .any(|key| is_set(&block, key));
    let drop_environment_url = names_a_target && !is_set(&block, "url");
    environment(lookup)
        .filter(|(key, _)| !(key == "url" && drop_environment_url))
        .chain(block)
        .collect()
}

fn keeps_null(nulls: Nulls, key: &str) -> bool {
    match nulls {
        Nulls::Dropped => false,
        Nulls::KeptUnlessNamedByCache => !NAMED_BY_CACHE.contains(&key),
        Nulls::Kept => true,
    }
}

fn environment(lookup: &dyn Lookup) -> impl Iterator<Item = (String, Entry)> + '_ {
    keys::SETTINGS.iter().filter_map(|setting| {
        let source = setting.environment_name()?;
        let value = lookup.get(&source)?;
        Some((
            setting.kwarg.to_owned(),
            Entry {
                value: Value::String(value),
                source,
            },
        ))
    })
}

// load_config resolves os.environ/ to None when unset, so a missing reference is a null like any other
fn environment_reference(value: &Value, lookup: &dyn Lookup) -> Value {
    match value {
        Value::String(text) => match text.strip_prefix("os.environ/") {
            Some(name) => lookup.get(name).map_or(Value::Null, Value::String),
            None => value.clone(),
        },
        _ => value.clone(),
    }
}

fn is_set(kwargs: &Kwargs, key: &str) -> bool {
    kwargs.get(key).is_some_and(|entry| !entry.value.is_null())
}
