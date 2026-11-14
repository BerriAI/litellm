use litellm_secret_value::SecretValue;

#[derive(Clone, Debug, PartialEq)]
pub enum Auth {
    Static(StaticAuth),
    /// Credentials come from a host object, like a redis-py `credential_provider`.
    Provider(ProviderHandle),
    GcpIam {
        service_account: String,
    },
    AzureAd(AzureAdAuth),
    AwsIam(AwsIamAuth),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StaticAuth {
    pub username: Option<SecretValue>,
    pub password: Option<SecretValue>,
}

/// Names one credential provider the host holds. Two resolutions with the same handle use the
/// same provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProviderHandle(u64);

impl ProviderHandle {
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    pub const fn id(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AzureAdAuth {
    pub client_id: Option<String>,
    pub tenant_id: Option<String>,
    pub client_secret: Option<SecretValue>,
    pub username: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AwsIamAuth {
    pub user_name: String,
    pub cache_name: String,
    pub region: String,
    pub serverless: bool,
}
