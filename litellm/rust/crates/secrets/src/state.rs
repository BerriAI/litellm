use crate::{Error, KeyManagementSettings, KeyManagementSystem, OidcReference, SecretManager};

const ENVIRONMENT_PREFIX: &str = "os.environ/";
const OIDC_PREFIX: &str = "oidc/";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SecretReference<'a> {
    Oidc(OidcReference<'a>),
    Named(&'a str),
}

impl<'a> TryFrom<&'a str> for SecretReference<'a> {
    type Error = Error;

    fn try_from(text: &'a str) -> Result<Self, Error> {
        let name = text.strip_prefix(ENVIRONMENT_PREFIX).unwrap_or(text);
        if name.starts_with(OIDC_PREFIX) {
            return name.try_into().map(Self::Oidc);
        }
        Ok(Self::Named(name))
    }
}

pub(crate) enum Route<'a> {
    Oidc(OidcReference<'a>),
    Manager {
        backend: &'a SecretManager,
        settings: &'a KeyManagementSettings,
        name: &'a str,
    },
    Environment(&'a str),
}

#[derive(Clone, Default)]
pub struct SecretManagerState {
    manager: Option<(SecretManager, KeyManagementSettings)>,
}

impl SecretManagerState {
    pub fn new(backend: SecretManager, settings: KeyManagementSettings) -> Self {
        Self {
            manager: Some((backend, settings)),
        }
    }

    pub fn system(&self) -> Option<KeyManagementSystem> {
        self.backend().map(SecretManager::system)
    }

    pub fn settings(&self) -> Option<&KeyManagementSettings> {
        self.manager.as_ref().map(|(_, settings)| settings)
    }

    pub fn backend(&self) -> Option<&SecretManager> {
        self.manager.as_ref().map(|(backend, _)| backend)
    }

    pub(crate) fn route<'a>(&'a self, reference: SecretReference<'a>) -> Route<'a> {
        let name = match reference {
            SecretReference::Oidc(reference) => return Route::Oidc(reference),
            SecretReference::Named(name) => name,
        };
        match &self.manager {
            Some((backend, settings))
                if backend.system() != KeyManagementSystem::Local
                    && settings.access_mode.readable()
                    && settings
                        .hosted_keys
                        .as_ref()
                        .is_none_or(|keys| keys.iter().any(|key| key == name)) =>
            {
                Route::Manager {
                    backend,
                    settings,
                    name,
                }
            }
            _ => Route::Environment(name),
        }
    }
}

pub fn secret_manager_would_be_consulted(state: &SecretManagerState, name: &str) -> bool {
    SecretReference::try_from(name)
        .is_ok_and(|reference| matches!(state.route(reference), Route::Manager { .. }))
}
