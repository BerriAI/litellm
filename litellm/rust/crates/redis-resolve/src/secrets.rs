use std::collections::{BTreeMap, BTreeSet};

use futures_util::future::try_join_all;
use litellm_config::Value;
use litellm_secrets::{Secret, SecretResolver};
use strum::IntoEnumIterator;

use crate::{configured::ConfiguredRedis, keys::Kwarg};

/// What `get_secret` returns for every name the merge reads, fetched once: OIDC, the secret manager, then
/// the environment.
#[derive(Clone, Debug, Default)]
pub(crate) struct SecretSnapshot(BTreeMap<String, Option<Secret>>);

impl SecretSnapshot {
    pub(crate) async fn read(
        names: BTreeSet<String>,
        secrets: &SecretResolver,
    ) -> Result<Self, litellm_secrets::Error> {
        let values = try_join_all(names.into_iter().map(|name| async move {
            let value = secrets.get_secret(&name, None).await?;
            Ok::<_, litellm_secrets::Error>((name, value))
        }))
        .await?;
        Ok(Self(values.into_iter().collect()))
    }

    /// The values `get_secret` returns with no secret manager: a text that spells a boolean is one.
    #[cfg(test)]
    pub(crate) fn from_environment(
        names: BTreeSet<String>,
        lookup: &dyn litellm_env::Lookup,
    ) -> Self {
        use litellm_python_compat::flags::str_to_bool;
        use litellm_secret_value::SecretValue;

        Self(
            names
                .into_iter()
                .map(|name| {
                    let value = lookup.get(&name).map(|text| {
                        str_to_bool(&text)
                            .map_or_else(|| Secret::String(SecretValue::new(text)), Secret::Bool)
                    });
                    (name, value)
                })
                .collect(),
        )
    }

    pub(crate) fn get(&self, name: &str) -> Option<serde_json::Value> {
        let Some(value) = self.0.get(name) else {
            debug_assert!(
                false,
                "{name} is missing from ConfiguredRedis::secret_names"
            );
            tracing::error!(
                name,
                "a Redis secret was read without being in the snapshot"
            );
            return None;
        };
        value.as_ref().map(|secret| match secret {
            Secret::String(text) => serde_json::Value::String(text.expose().to_owned()),
            Secret::Bool(flag) => serde_json::Value::Bool(*flag),
            Secret::Json(value) => value.clone(),
        })
    }
}

impl ConfiguredRedis {
    pub(crate) fn secret_names(&self) -> BTreeSet<String> {
        let references = self.fields.values().filter_map(|value| match value {
            Value::String(text) => text.strip_prefix("os.environ/").map(str::to_owned),
            _ => None,
        });
        let declared = Kwarg::iter().flat_map(Kwarg::source_names);
        declared.chain(references).collect()
    }
}
