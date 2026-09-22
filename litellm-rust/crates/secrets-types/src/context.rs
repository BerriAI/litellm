use std::{collections::BTreeMap, time::Duration};

use crate::SecretValue;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum SecretOperationContext {
    #[default]
    Default,
    Aws(AwsOperationContext),
    Hashicorp(HashicorpOperationContext),
    Cyberark(CyberarkOperationContext),
}

impl SecretOperationContext {
    pub fn timeout(&self) -> Option<Duration> {
        match self {
            Self::Default => None,
            Self::Aws(context) => context.timeout,
            Self::Hashicorp(context) => context.timeout,
            Self::Cyberark(context) => context.timeout,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AwsOperationContext {
    pub timeout: Option<Duration>,
    pub region_name: Option<String>,
    pub role_name: Option<String>,
    pub session_name: Option<String>,
    pub external_id: Option<SecretValue>,
    pub profile_name: Option<String>,
    pub web_identity_token: Option<SecretValue>,
    pub sts_endpoint: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HashicorpOperationContext {
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
pub struct SecretWriteContext {
    pub description: Option<String>,
    pub tags: BTreeMap<String, String>,
    pub operation: SecretOperationContext,
}

impl SecretWriteContext {
    pub fn rotated_from(current_name: &str, operation: SecretOperationContext) -> Self {
        Self {
            description: Some(format!("Rotated from {current_name}")),
            tags: BTreeMap::new(),
            operation,
        }
    }
}
