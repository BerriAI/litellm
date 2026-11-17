use std::{collections::BTreeMap, sync::Arc};

use litellm_config::{Object, Value};
use litellm_env::Lookup;
use litellm_secret_value::SecretValue;

use crate::Error;

/// A config's `environment_variables` in front of another environment, the way the proxy
/// exports them into `os.environ`.
pub struct ConfigEnvironment {
    values: BTreeMap<String, SecretValue>,
    fallback: Arc<dyn Lookup + Send + Sync>,
}

impl ConfigEnvironment {
    pub fn new(
        values: BTreeMap<String, SecretValue>,
        fallback: Arc<dyn Lookup + Send + Sync>,
    ) -> Self {
        Self { values, fallback }
    }
}

impl Lookup for ConfigEnvironment {
    fn get(&self, name: &str) -> Option<String> {
        self.values
            .get(name)
            .map(|value| value.expose().to_owned())
            .or_else(|| self.fallback.get(name))
    }
}

/// The text each `environment_variables` entry exports. An `os.environ/` reference that named
/// an unset variable is null by now and exports nothing.
pub fn environment_values(values: &Object) -> Result<BTreeMap<String, SecretValue>, Error> {
    values
        .iter()
        .filter_map(|(key, value)| {
            let text = match value {
                Value::Null => return None,
                Value::String(value) => value.clone(),
                Value::Number(value) => value.to_string(),
                Value::Bool(value) => value.to_string(),
                _ => return Some(Err(Error::Environment { name: key.clone() })),
            };
            Some(Ok((key.clone(), SecretValue::new(text))))
        })
        .collect()
}
