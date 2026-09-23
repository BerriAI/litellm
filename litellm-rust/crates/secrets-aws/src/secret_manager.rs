mod client;
mod read;
mod write;

pub use read::is_bootstrap_key;

use litellm_auth_aws::constants::AWS_BEDROCK_RUNTIME_ENDPOINT;
use std::{collections::BTreeMap, sync::Arc};

use aws_sdk_secretsmanager::{
    Client,
    config::{BehaviorVersion, Region},
    operation::{
        create_secret::CreateSecretOutput, delete_secret::DeleteSecretOutput,
        put_secret_value::PutSecretValueOutput,
        replicate_secret_to_regions::ReplicateSecretToRegionsOutput,
    },
    types::{ReplicaRegionType, Tag},
};
use litellm_auth_aws::constants::{
    AWS_ACCESS_KEY_ID, AWS_REGION, AWS_REGION_NAME, AWS_SECRET_ACCESS_KEY,
};
use litellm_core_utils::settings::Lookup;
use litellm_secrets_types::{
    AwsOperationContext, BaseSecretManager, KeyManagementSettings, RotationError, Secret,
    SecretDeleter, SecretRotator, SecretValue, SecretWriteContext, SecretWriter,
    async_rotate_secret,
};
use serde_json::Value;

use crate::{Error, auth};

#[derive(Clone)]
pub struct AwsSecretsManagerV2 {
    client: Client,
    context_client_factory: Option<Box<ContextClientFactory>>,
    write_settings: AwsSecretWriteSettings,
}

#[derive(Clone)]
struct ContextClientFactory {
    settings: KeyManagementSettings,
    environment: Arc<dyn Lookup + Send + Sync>,
    endpoint_url: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct AwsSecretWriteSettings {
    pub kms_key_id: Option<String>,
    pub tags: Option<BTreeMap<String, String>>,
    pub replica_regions: Option<Vec<String>>,
}

impl From<&KeyManagementSettings> for AwsSecretWriteSettings {
    fn from(settings: &KeyManagementSettings) -> Self {
        Self {
            kms_key_id: settings.kms_key_id.clone(),
            tags: settings.tags.clone(),
            replica_regions: settings.replica_regions.clone(),
        }
    }
}

#[derive(Debug)]
pub enum RotationResponse {
    Created(CreateSecretOutput),
    Updated(PutSecretValueOutput),
}

impl AwsSecretsManagerV2 {
    pub fn new(client: Client, write_settings: AwsSecretWriteSettings) -> Self {
        Self {
            client,
            context_client_factory: None,
            write_settings,
        }
    }
}
