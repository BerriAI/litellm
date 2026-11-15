use std::collections::{BTreeMap, BTreeSet};

use litellm_core_utils::settings::Lookup;
use litellm_secret_value::SecretValue;

use litellm_config::Value;
use litellm_redis_resolve::SETTINGS;

use super::configured::ConfiguredRedis;

#[derive(Clone, Debug, Default)]
pub(crate) struct SecretSnapshot(BTreeMap<String, Option<SecretValue>>);

impl SecretSnapshot {
    pub(crate) fn read(names: BTreeSet<String>, lookup: &dyn Lookup) -> Self {
        Self(
            names
                .into_iter()
                .map(|name| {
                    let value = lookup.get(&name).map(SecretValue::new);
                    (name, value)
                })
                .collect(),
        )
    }
}

impl Lookup for SecretSnapshot {
    fn get(&self, name: &str) -> Option<String> {
        match self.0.get(name) {
            Some(value) => value.as_ref().map(|value| value.expose().to_owned()),
            None => {
                debug_assert!(
                    false,
                    "{name} is missing from ConfiguredRedis::secret_names"
                );
                tracing::error!(
                    name,
                    "a Redis secret was read without being in the snapshot"
                );
                None
            }
        }
    }
}

impl ConfiguredRedis {
    pub(crate) fn secret_names(&self) -> BTreeSet<String> {
        let references = self.fields.values().filter_map(|value| match value {
            Value::String(text) => text.strip_prefix("os.environ/").map(str::to_owned),
            _ => None,
        });
        let declared = SETTINGS.iter().flat_map(|setting| {
            setting
                .environment_name()
                .into_iter()
                .chain(setting.fallback.iter().map(|name| (*name).to_owned()))
        });
        declared.chain(references).collect()
    }
}
