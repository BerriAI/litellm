use std::{collections::BTreeMap, time::Duration};

use crate::{Error, KeyManagementSystem, SecretValue};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum SecretOperationContext {
    #[default]
    Default,
    Aws(AwsOperationContext),
    Azure(AzureOperationContext),
    Google(GoogleOperationContext),
    Hashicorp(HashicorpOperationContext),
    Cyberark(CyberarkOperationContext),
}

impl SecretOperationContext {
    pub fn validate_for(&self, system: KeyManagementSystem) -> Result<(), Error> {
        let compatible = match self {
            Self::Default => true,
            Self::Aws(_) => system == KeyManagementSystem::AwsSecretManager,
            Self::Azure(_) => system == KeyManagementSystem::AzureKeyVault,
            Self::Google(_) => system == KeyManagementSystem::GoogleSecretManager,
            Self::Hashicorp(_) => system == KeyManagementSystem::HashicorpVault,
            Self::Cyberark(_) => system == KeyManagementSystem::Cyberark,
        };
        if compatible {
            Ok(())
        } else {
            Err(Error::InvalidOperationContext)
        }
    }

    pub fn timeout(&self) -> Option<Duration> {
        match self {
            Self::Default => None,
            Self::Aws(context) => context.timeout,
            Self::Azure(context) => context.timeout,
            Self::Google(context) => context.timeout,
            Self::Hashicorp(context) => context.timeout,
            Self::Cyberark(context) => context.timeout,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AwsOperationContext {
    pub access_key_id: Option<SecretValue>,
    pub secret_access_key: Option<SecretValue>,
    pub session_token: Option<SecretValue>,
    pub timeout: Option<Duration>,
    pub region_name: Option<String>,
    pub role_name: Option<String>,
    pub session_name: Option<String>,
    pub external_id: Option<SecretValue>,
    pub profile_name: Option<String>,
    pub web_identity_token: Option<SecretValue>,
    pub sts_endpoint: Option<String>,
    pub bedrock_runtime_endpoint: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HashicorpOperationContext {
    pub namespace: Option<String>,
    pub timeout: Option<Duration>,
    pub mount: Option<String>,
    pub path_prefix: Option<String>,
    pub data_key: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CyberarkOperationContext {
    pub timeout: Option<Duration>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SecretWriteContext<C = SecretOperationContext> {
    pub description: Option<String>,
    pub tags: BTreeMap<String, String>,
    pub operation: C,
}

impl<C> SecretWriteContext<C> {
    pub fn rotated_from(current_name: &str, operation: C) -> Self {
        Self {
            description: Some(format!("Rotated from {current_name}")),
            tags: BTreeMap::new(),
            operation,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AzureOperationContext {
    pub timeout: Option<Duration>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GoogleOperationContext {
    pub timeout: Option<Duration>,
}
