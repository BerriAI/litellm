use std::sync::Arc;

use litellm_core_utils::settings::{Lookup, ProcessEnvironment};

use crate::state::{LookupTarget, normalize_secret_name};
use crate::{Error, OidcResolver, Secret, SecretManagerState, SecretValue};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FailurePolicy {
    #[default]
    Propagate,
    EnvironmentFallback,
}

pub struct SecretResolver {
    state: Arc<SecretManagerState>,
    environment: Arc<dyn Lookup + Send + Sync>,
    oidc: OidcResolver,
    failure_policy: FailurePolicy,
}

impl Default for SecretResolver {
    fn default() -> Self {
        Self::new(
            Arc::new(SecretManagerState::default()),
            Arc::new(ProcessEnvironment),
            OidcResolver::default(),
        )
    }
}

impl SecretResolver {
    pub fn new(
        state: Arc<SecretManagerState>,
        environment: Arc<dyn Lookup + Send + Sync>,
        oidc: OidcResolver,
    ) -> Self {
        Self {
            state,
            environment,
            oidc,
            failure_policy: FailurePolicy::default(),
        }
    }

    pub fn with_failure_policy(self, failure_policy: FailurePolicy) -> Self {
        Self {
            failure_policy,
            ..self
        }
    }

    pub async fn get_secret(
        &self,
        name: &str,
        default_value: Option<Secret>,
    ) -> Result<Option<Secret>, Error> {
        let name = normalize_secret_name(name);
        if name.starts_with("oidc/") {
            return self
                .oidc
                .resolve(name, self.environment.as_ref())
                .await
                .map(|value| value.map(Secret::String).or(default_value));
        }
        let LookupTarget::Manager { backend, settings } = self.state.lookup_target(name) else {
            return Ok(self.environment_secret(name).or(default_value));
        };
        match crate::get_secret_from_manager(backend, name, settings, self.environment.as_ref())
            .await
        {
            Ok(value) => Ok(value
                .or_else(|| self.environment_secret(name))
                .or(default_value)),
            Err(error) => match self.failure_policy {
                FailurePolicy::Propagate => Err(error),
                FailurePolicy::EnvironmentFallback => self
                    .environment_secret(name)
                    .or(default_value)
                    .map(Some)
                    .ok_or(error),
            },
        }
    }

    fn environment_secret(&self, name: &str) -> Option<Secret> {
        self.environment
            .get(name)
            .map(SecretValue::new)
            .map(Secret::String)
    }

    pub async fn get_secret_str(
        &self,
        name: &str,
        default_value: Option<SecretValue>,
    ) -> Result<Option<SecretValue>, Error> {
        match self
            .get_secret(name, default_value.map(Secret::String))
            .await?
        {
            Some(Secret::String(value)) => Ok(Some(value)),
            None => Ok(None),
            Some(Secret::Bool(_) | Secret::Json(_)) => {
                Err(Error::TypeMismatch { expected: "string" })
            }
        }
    }

    pub async fn get_secret_bool(
        &self,
        name: &str,
        default_value: Option<bool>,
    ) -> Result<Option<bool>, Error> {
        match self
            .get_secret(name, default_value.map(Secret::Bool))
            .await?
        {
            Some(Secret::Bool(value)) => Ok(Some(value)),
            Some(Secret::String(value)) => {
                match value.expose().trim().to_ascii_lowercase().as_str() {
                    "true" => Ok(Some(true)),
                    "false" => Ok(Some(false)),
                    _ => Err(Error::TypeMismatch {
                        expected: "boolean",
                    }),
                }
            }
            Some(Secret::Json(_)) => Err(Error::TypeMismatch {
                expected: "boolean",
            }),
            None => Ok(None),
        }
    }
}
