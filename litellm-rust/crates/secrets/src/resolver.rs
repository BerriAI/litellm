use std::sync::Arc;

use litellm_core_utils::settings::{Lookup, ProcessEnvironment};

use crate::{Error, OidcResolver, Secret, SecretManagerState, SecretValue};

use crate::state::{LookupTarget, normalize_secret_name};

pub struct SecretResolver {
    state: Arc<SecretManagerState>,
    environment: Arc<dyn Lookup + Send + Sync>,
    oidc: OidcResolver,
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
        }
    }

    pub async fn get_secret(
        &self,
        name: &str,
        _default_value: Option<Secret>,
    ) -> Result<Option<Secret>, Error> {
        let name = normalize_secret_name(name);
        if name.starts_with("oidc/") {
            return self
                .oidc
                .resolve(name, self.environment.as_ref())
                .await
                .map(|value| value.map(Secret::String));
        }
        if !self.state.readable() {
            return Ok(self
                .environment
                .get(name)
                .map(|value| match str_to_bool(&value) {
                    Some(value) => Secret::Bool(value),
                    None => Secret::String(SecretValue::new(value)),
                }));
        }
        let result = match self.state.lookup_target(name) {
            LookupTarget::Environment => Ok(self.environment_secret(name)),
            LookupTarget::Manager { backend, settings } => {
                crate::get_secret_from_manager(backend, name, settings, self.environment.as_ref())
                    .await
            }
        };
        let value = match result {
            Ok(value) => value,
            Err(_) => {
                tracing::error!("secret manager lookup failed; falling back to environment");
                self.environment_secret(name)
            }
        };
        Ok(value.and_then(managed_secret))
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
        default_value: Option<Secret>,
    ) -> Result<Option<SecretValue>, Error> {
        Ok(match self.get_secret(name, default_value).await? {
            Some(Secret::String(value)) => Some(value),
            Some(Secret::Bool(_) | Secret::Json(_)) | None => None,
        })
    }

    pub async fn get_secret_bool(
        &self,
        name: &str,
        default_value: Option<bool>,
    ) -> Result<Option<bool>, Error> {
        Ok(
            match self
                .get_secret(name, default_value.map(Secret::Bool))
                .await?
            {
                Some(Secret::Bool(value)) => Some(value),
                Some(Secret::String(value)) => str_to_bool(value.expose()),
                Some(Secret::Json(_)) | None => None,
            },
        )
    }
}

fn str_to_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn literal_bool(value: &str) -> Option<bool> {
    use rustpython_parser::{Parse, ast};
    match ast::Expr::parse(value.trim_start_matches([' ', '\t']), "<secret>").ok()? {
        ast::Expr::Constant(node) => match node.value {
            ast::Constant::Bool(value) => Some(value),
            _ => None,
        },
        _ => None,
    }
}

fn managed_secret(value: Secret) -> Option<Secret> {
    match value {
        Secret::String(value) => Some(match literal_bool(value.expose()) {
            Some(boolean) => Secret::Bool(boolean),
            None => Secret::String(value),
        }),
        Secret::Bool(_) | Secret::Json(_) => None,
    }
}
