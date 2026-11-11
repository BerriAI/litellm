use std::sync::Arc;

use litellm_host_python::{PythonContext, attach_blocking};
use litellm_redis_connect::{CredentialSource, CredentialStage, Error, Expiring, SourceFuture};
use litellm_secret_value::SecretValue;
use pyo3::prelude::*;

const MODULE: &str = "litellm.rust_bridge.host.redis_credentials";

/// A redis-py `CredentialProvider` the SDK caller passed in `cache_kwargs`, asked through the host
/// module the way redis-py's async connection asks it.
pub(crate) struct PythonCredentialSource {
    provider: Arc<Py<PyAny>>,
    context: PythonContext,
}

impl PythonCredentialSource {
    pub(crate) fn new(provider: Py<PyAny>, context: PythonContext) -> Self {
        Self {
            provider: Arc::new(provider),
            context,
        }
    }
}

impl CredentialSource for PythonCredentialSource {
    fn fetch(&self) -> SourceFuture<'_> {
        let provider = Arc::clone(&self.provider);
        let context = self.context.clone();
        Box::pin(async move {
            let (username, password) = attach_blocking(context, move |py| {
                py.import(MODULE)?
                    .getattr("credentials")?
                    .call1((provider.bind(py),))?
                    .extract::<(Option<String>, String)>()
            })
            .await
            .and_then(|credentials| credentials)
            .map_err(Error::credential(CredentialStage::Provider))?;
            Ok(Expiring {
                username,
                password: SecretValue::new(password),
                expires_at: None,
            })
        })
    }
}
