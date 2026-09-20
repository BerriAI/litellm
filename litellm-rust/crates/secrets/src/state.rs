use crate::{Error, KeyManagementSettings, KeyManagementSystem, SecretManager};

pub(crate) enum LookupTarget<'a> {
    Environment,
    Manager {
        backend: &'a SecretManager,
        settings: &'a KeyManagementSettings,
    },
}

pub(crate) fn normalize_secret_name(name: &str) -> &str {
    name.strip_prefix("os.environ/").unwrap_or(name)
}

#[derive(Clone, Default)]
pub struct SecretManagerState {
    system: Option<KeyManagementSystem>,
    settings: Option<KeyManagementSettings>,
    backend: Option<SecretManager>,
}

impl SecretManagerState {
    pub fn new(
        system: Option<KeyManagementSystem>,
        settings: Option<KeyManagementSettings>,
        backend: Option<SecretManager>,
    ) -> Result<Self, Error> {
        if let Some(system) = system {
            let available = match system {
                KeyManagementSystem::Local => true,
                KeyManagementSystem::AwsKms | KeyManagementSystem::AwsSecretManager => {
                    cfg!(feature = "aws")
                }
                KeyManagementSystem::GoogleKms | KeyManagementSystem::GoogleSecretManager => {
                    cfg!(feature = "google")
                }
                KeyManagementSystem::AzureKeyVault
                | KeyManagementSystem::HashicorpVault
                | KeyManagementSystem::Cyberark
                | KeyManagementSystem::Custom => false,
            };
            if !available {
                return Err(Error::UnsupportedBackend(system));
            }
            if let Some(backend) = &backend
                && system != backend.system()
            {
                return Err(Error::BackendMismatch);
            }
        }
        Ok(Self {
            system,
            settings,
            backend,
        })
    }

    pub fn system(&self) -> Option<KeyManagementSystem> {
        self.system
    }
    pub fn settings(&self) -> Option<&KeyManagementSettings> {
        self.settings.as_ref()
    }
    pub fn backend(&self) -> Option<&SecretManager> {
        self.backend.as_ref()
    }

    pub(crate) fn readable(&self) -> bool {
        self.backend.is_some()
            && self
                .settings
                .as_ref()
                .is_some_and(|settings| settings.access_mode.readable())
    }

    pub(crate) fn lookup_target(&self, name: &str) -> LookupTarget<'_> {
        match (&self.backend, &self.settings) {
            (Some(backend), Some(settings))
                if settings.access_mode.readable()
                    && hosts_secret(settings, name)
                    && self
                        .system
                        .is_some_and(|system| system != KeyManagementSystem::Local) =>
            {
                LookupTarget::Manager { backend, settings }
            }
            _ => LookupTarget::Environment,
        }
    }
}

pub fn secret_manager_would_be_consulted(state: &SecretManagerState, name: &str) -> bool {
    state.readable()
        && state
            .settings
            .as_ref()
            .is_some_and(|settings| hosts_secret(settings, normalize_secret_name(name)))
}

fn hosts_secret(settings: &KeyManagementSettings, name: &str) -> bool {
    settings
        .hosted_keys
        .as_ref()
        .is_none_or(|keys| keys.iter().any(|key| key == name))
}
