use crate::{KeyManagementSettings, KeyManagementSystem, SecretManager};

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

    pub(crate) fn lookup_target(&self, name: &str) -> LookupTarget<'_> {
        match &self.manager {
            Some((backend, settings))
                if backend.system() != KeyManagementSystem::Local
                    && settings.access_mode.readable()
                    && settings
                        .hosted_keys
                        .as_ref()
                        .is_none_or(|keys| keys.iter().any(|key| key == name)) =>
            {
                LookupTarget::Manager { backend, settings }
            }
            _ => LookupTarget::Environment,
        }
    }
}

pub fn secret_manager_would_be_consulted(state: &SecretManagerState, name: &str) -> bool {
    let name = normalize_secret_name(name);
    !name.starts_with("oidc/") && matches!(state.lookup_target(name), LookupTarget::Manager { .. })
}
