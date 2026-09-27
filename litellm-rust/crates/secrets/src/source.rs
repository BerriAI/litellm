use std::{collections::HashMap, sync::Arc};

use futures_util::future::{BoxFuture, try_join_all};
use litellm_core_utils::settings::Lookup;

use crate::{Error, SecretResolver, SecretValue};

pub type Secrets = Arc<dyn Lookup + Send + Sync>;

pub trait SecretSource: Send + Sync {
    fn get_secret_str<'a>(
        &'a self,
        name: &'a str,
    ) -> BoxFuture<'a, Result<Option<SecretValue>, Error>>;

    fn resolve<'a>(&'a self, names: &'a [&str]) -> BoxFuture<'a, Result<Secrets, Error>> {
        Box::pin(async move {
            let values = try_join_all(names.iter().map(|name| async move {
                self.get_secret_str(name)
                    .await
                    .map(|value| ((*name).to_owned(), value))
            }))
            .await?
            .into_iter()
            .collect();
            Ok(Arc::new(SecretSnapshot { values }) as Secrets)
        })
    }
}

impl SecretSource for SecretResolver {
    fn get_secret_str<'a>(
        &'a self,
        name: &'a str,
    ) -> BoxFuture<'a, Result<Option<SecretValue>, Error>> {
        Box::pin(SecretResolver::get_secret_str(self, name, None))
    }
}

pub struct EnvironmentSecrets(SecretResolver);

impl EnvironmentSecrets {
    pub fn python_compatible(client: litellm_http::Client) -> Self {
        Self(SecretResolver::new_python_compatible(
            Arc::new(crate::SecretManagerState::default()),
            Arc::new(litellm_core_utils::settings::ProcessEnvironment),
            crate::OidcResolver::new(client),
        ))
    }
}

impl SecretSource for EnvironmentSecrets {
    fn get_secret_str<'a>(
        &'a self,
        name: &'a str,
    ) -> BoxFuture<'a, Result<Option<SecretValue>, Error>> {
        Box::pin(self.0.get_secret_str(name, None))
    }
}

struct SecretSnapshot {
    values: HashMap<String, Option<SecretValue>>,
}

impl Lookup for SecretSnapshot {
    fn get(&self, name: &str) -> Option<String> {
        self.values
            .get(name)
            .and_then(Option::as_ref)
            .map(|value| value.expose().to_owned())
    }
}
