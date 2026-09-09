use crate::CallerTokenProvider;

#[derive(Clone, Default)]
pub struct CredentialInputs {
    pub azure: AzureCredentialInputs,
}

#[derive(Clone, Default)]
pub struct AzureCredentialInputs {
    pub token: Option<String>,
    pub has_token_provider: bool,
    pub tenant_id: Option<String>,
    pub client_id: Option<String>,
    pub has_client_secret: bool,
    pub has_username: bool,
    pub has_password: bool,
    pub refresh: bool,
}

#[derive(Clone, Copy)]
pub enum CallerCredential {
    AzureAdToken,
}

pub trait AuthServices: Send + Sync {
    fn token_provider(&self, credential: CallerCredential) -> Option<&dyn CallerTokenProvider>;
}

impl AuthServices for () {
    fn token_provider(&self, _: CallerCredential) -> Option<&dyn CallerTokenProvider> {
        None
    }
}
