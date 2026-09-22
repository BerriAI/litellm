use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_core_utils::settings::{Lookup, ProcessEnvironment};
use litellm_secrets::Error;

pub type Secrets = Arc<dyn Lookup + Send + Sync>;

pub trait SecretSource: Send + Sync {
    fn resolve<'a>(&'a self, names: &'a [&'static str]) -> BoxFuture<'a, Result<Secrets, Error>>;
}

pub struct EnvironmentSecrets;

impl SecretSource for EnvironmentSecrets {
    fn resolve<'a>(&'a self, _names: &'a [&'static str]) -> BoxFuture<'a, Result<Secrets, Error>> {
        Box::pin(async { Ok(Arc::new(ProcessEnvironment) as Secrets) })
    }
}
