use std::sync::Arc;
use std::time::Duration;

use litellm_core::auth::{
    AuthBinding, AuthFuture, AuthHttpClient, AuthRuntime, CredentialResolver, CredentialSpec,
    StaticHeaderAuthorizer, SystemClock,
};

use crate::constants::{AUTH_CONNECT_TIMEOUT_SECS, AUTH_REQUEST_TIMEOUT_SECS};

pub fn runtime() -> Result<Arc<AuthRuntime>, litellm_core::Error> {
    Ok(Arc::new(AuthRuntime {
        http: AuthHttpClient::new(
            reqwest::Client::builder(),
            Duration::from_secs(AUTH_CONNECT_TIMEOUT_SECS),
            Duration::from_secs(AUTH_REQUEST_TIMEOUT_SECS),
        )?,
        clock: Arc::new(SystemClock),
        credentials: Arc::new(NativeCredentials),
    }))
}

struct NativeCredentials;

impl CredentialResolver for NativeCredentials {
    fn resolve(
        &self,
        credential: CredentialSpec,
        _runtime: Arc<AuthRuntime>,
    ) -> AuthFuture<'_, AuthBinding> {
        Box::pin(async move {
            match credential {
                CredentialSpec::Header { name, value } => Ok(AuthBinding::new(Arc::new(
                    StaticHeaderAuthorizer::new(name, value, Vec::new()),
                ))),
            }
        })
    }
}
