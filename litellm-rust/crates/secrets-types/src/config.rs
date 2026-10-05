use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use strum::IntoStaticStr;

use crate::SecretValue;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, IntoStaticStr, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum KeyManagementSystem {
    GoogleKms,
    AzureKeyVault,
    AwsSecretManager,
    GoogleSecretManager,
    HashicorpVault,
    Cyberark,
    Local,
    AwsKms,
    Custom,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AccessMode {
    #[default]
    ReadOnly,
    WriteOnly,
    ReadAndWrite,
}

impl AccessMode {
    pub fn readable(self) -> bool {
        matches!(self, Self::ReadOnly | Self::ReadAndWrite)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(default)]
pub struct KeyManagementSettings {
    pub hosted_keys: Option<Vec<String>>,
    pub store_virtual_keys: Option<bool>,
    pub prefix_for_stored_virtual_keys: String,
    pub access_mode: AccessMode,
    pub primary_secret_name: Option<String>,
    pub description: Option<String>,
    pub tags: Option<BTreeMap<String, String>>,
    pub kms_key_id: Option<String>,
    pub custom_secret_manager: Option<String>,
    pub aws_region_name: Option<String>,
    pub aws_role_name: Option<String>,
    pub aws_session_name: Option<String>,
    #[serde(serialize_with = "serialize_secret")]
    pub aws_external_id: Option<SecretValue>,
    pub aws_profile_name: Option<String>,
    #[serde(serialize_with = "serialize_secret")]
    pub aws_web_identity_token: Option<SecretValue>,
    pub aws_sts_endpoint: Option<String>,
    pub replica_regions: Option<Vec<String>>,
}

impl Default for KeyManagementSettings {
    fn default() -> Self {
        Self {
            hosted_keys: None,
            store_virtual_keys: Some(false),
            prefix_for_stored_virtual_keys: "litellm/".into(),
            access_mode: AccessMode::ReadOnly,
            primary_secret_name: None,
            description: None,
            tags: None,
            kms_key_id: None,
            custom_secret_manager: None,
            aws_region_name: None,
            aws_role_name: None,
            aws_session_name: None,
            aws_external_id: None,
            aws_profile_name: None,
            aws_web_identity_token: None,
            aws_sts_endpoint: None,
            replica_regions: None,
        }
    }
}

fn serialize_secret<S: serde::Serializer>(
    value: &Option<SecretValue>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    value
        .as_ref()
        .map(SecretValue::expose)
        .serialize(serializer)
}
