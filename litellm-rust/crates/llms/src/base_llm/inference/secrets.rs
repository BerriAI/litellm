use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_core_utils::settings::{Lookup, ProcessEnvironment};

pub type Secrets = Arc<dyn Lookup + Send + Sync>;

pub trait SecretSource: Send + Sync {
    fn resolve<'a>(&'a self, names: &'a [&'static str]) -> BoxFuture<'a, Secrets>;
}

pub struct EnvironmentSecrets;

impl SecretSource for EnvironmentSecrets {
    fn resolve<'a>(&'a self, _names: &'a [&'static str]) -> BoxFuture<'a, Secrets> {
        Box::pin(async { Arc::new(ProcessEnvironment) as Secrets })
    }
}
