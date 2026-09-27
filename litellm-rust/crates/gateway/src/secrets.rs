use std::{collections::BTreeMap, sync::Arc};

use futures_util::future::BoxFuture;
use litellm_auth_types::SecretValue;
use litellm_config::{Object, Value};
use litellm_secrets::source::SecretSource;

use crate::Error;

pub(super) struct ConfigSecrets {
    values: BTreeMap<String, SecretValue>,
    fallback: Arc<dyn SecretSource>,
}

impl ConfigSecrets {
    pub fn new(values: BTreeMap<String, SecretValue>, fallback: Arc<dyn SecretSource>) -> Self {
        Self { values, fallback }
    }
}

impl SecretSource for ConfigSecrets {
    fn get_secret_str<'a>(
        &'a self,
        name: &'a str,
    ) -> BoxFuture<'a, Result<Option<SecretValue>, litellm_secrets::Error>> {
        Box::pin(async move {
            match self.values.get(name) {
                Some(value) => Ok(Some(value.clone())),
                None => self.fallback.get_secret_str(name).await,
            }
        })
    }
}

pub(super) fn environment_values(values: &Object) -> Result<BTreeMap<String, SecretValue>, Error> {
    values
        .iter()
        .map(|(key, value)| {
            let text = match value {
                Value::String(value) => value.clone(),
                Value::Number(value) => value.to_string(),
                Value::Bool(value) => value.to_string(),
                _ => return Err(Error::Environment),
            };
            Ok((key.clone(), SecretValue::new(text)))
        })
        .collect()
}
