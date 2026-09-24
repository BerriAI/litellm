use std::sync::Arc;

use crate::compatibility::python_manager_string;
use litellm_core_utils::{
    serde_compat::parse_str_bool,
    settings::{Lookup, ProcessEnvironment},
};

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
    python_compatible: bool,
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
            python_compatible: false,
        }
    }

    pub fn new_python_compatible(
        state: Arc<SecretManagerState>,
        environment: Arc<dyn Lookup + Send + Sync>,
        oidc: OidcResolver,
    ) -> Self {
        Self {
            python_compatible: true,
            failure_policy: FailurePolicy::EnvironmentFallback,
            ..Self::new(state, environment, oidc)
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
        let value = self.read(name, default_value.clone()).await?;
        Ok(if self.python_compatible {
            value
        } else {
            value.or(default_value)
        })
    }

    async fn read(
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
                .map(|value| value.map(Secret::String));
        }
        let LookupTarget::Manager { backend, settings } = self.state.lookup_target(name) else {
            return Ok(self.environment_value(name));
        };
        let result = if self.python_compatible {
            crate::get_secret_from_python_manager(
                backend,
                name,
                settings,
                self.environment.as_ref(),
            )
            .await
        } else {
            crate::get_secret_from_manager(backend, name, settings, self.environment.as_ref()).await
        };
        match result {
            Ok(value) => Ok(value.and_then(|value| self.manager_value(value))),
            Err(error @ Error::ExternalManager(_)) => Err(error),
            Err(error) => match self.failure_policy {
                FailurePolicy::Propagate if self.python_compatible => {
                    default_value.map(Some).ok_or(error)
                }
                FailurePolicy::Propagate => Err(error),
                FailurePolicy::EnvironmentFallback => Ok(self
                    .environment
                    .get(name)
                    .and_then(|value| self.manager_value(Secret::String(SecretValue::new(value))))),
            },
        }
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
            Some(Secret::Bool(_) | Secret::Json(_)) if self.python_compatible => Ok(None),
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
            Some(Secret::String(value)) => match parse_str_bool(value.expose()) {
                Some(value) => Ok(Some(value)),
                None if self.python_compatible => Ok(None),
                None => Err(Error::TypeMismatch {
                    expected: "boolean",
                }),
            },
            Some(Secret::Json(_)) if self.python_compatible => Ok(None),
            Some(Secret::Json(_)) => Err(Error::TypeMismatch {
                expected: "boolean",
            }),
            None => Ok(None),
        }
    }

    fn environment_value(&self, name: &str) -> Option<Secret> {
        let value = self.environment.get(name)?;
        if !self.python_compatible {
            return Some(Secret::String(SecretValue::new(value)));
        }
        if self
            .state
            .settings()
            .is_some_and(|settings| settings.access_mode.readable())
        {
            return Some(python_manager_string(SecretValue::new(value)));
        }
        Some(
            parse_str_bool(&value)
                .map_or_else(|| Secret::String(SecretValue::new(value)), Secret::Bool),
        )
    }

    fn manager_value(&self, secret: Secret) -> Option<Secret> {
        if !self.python_compatible {
            return Some(secret);
        }

        let Secret::String(value) = secret else {
            return None;
        };
        Some(python_manager_string(value))
    }
}

pub fn normalize_nonempty_secret_str(value: Option<&str>) -> Option<&str> {
    value
        .map(|value| {
            value.trim_matches(|character: char| {
                character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
            })
        })
        .filter(|value| !value.is_empty())
}
