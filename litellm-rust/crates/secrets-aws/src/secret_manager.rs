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
    BaseSecretManager, KeyManagementSettings, Secret, SecretValue, async_rotate_secret,
};
use serde_json::Value;

use crate::{Error, auth};

#[derive(Clone)]
pub struct AwsSecretsManagerV2 {
    client: Client,
    write_settings: AwsSecretWriteSettings,
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
            write_settings,
        }
    }

    pub fn load_aws_secret_manager(
        use_aws_secret_manager: Option<bool>,
        settings: KeyManagementSettings,
        environment: Arc<dyn Lookup + Send + Sync>,
    ) -> Result<Option<Self>, Error> {
        if use_aws_secret_manager != Some(true) {
            return Ok(None);
        }
        let builder = aws_sdk_secretsmanager::Config::builder()
            .behavior_version(BehaviorVersion::latest())
            .region(Region::new(auth::region(&settings, environment.as_ref())?))
            .credentials_provider(auth::Credentials::new(&settings, environment.clone()));
        let config = match environment.get(AWS_BEDROCK_RUNTIME_ENDPOINT) {
            Some(url) => builder
                .endpoint_url(url.replace("bedrock-runtime", "secretsmanager"))
                .build(),
            None => builder.build(),
        };
        Ok(Some(Self::new(
            Client::from_conf(config),
            (&settings).into(),
        )))
    }

    pub async fn read_secret_for_resolver(
        &self,
        name: &str,
        primary_name: Option<&str>,
        environment: &(dyn Lookup + Sync),
    ) -> Result<Option<Secret>, Error> {
        if bootstrap_key(name) {
            return Ok(environment
                .get(name)
                .map(SecretValue::new)
                .map(Secret::String));
        }
        match primary_name.filter(|name| !name.is_empty()) {
            None => self
                .async_read_secret(name)
                .await
                .map(|value| value.map(Secret::String)),
            Some(primary) => {
                let value = if bootstrap_key(primary) {
                    environment.get(primary).map(SecretValue::new)
                } else {
                    self.async_read_secret(primary).await?
                };
                let Some(value) = value else {
                    return Ok(None);
                };
                let object: Value =
                    serde_json::from_str(value.expose()).map_err(|_| Error::PrimarySecret)?;
                let object = object.as_object().ok_or(Error::PrimarySecret)?;
                Ok(object.get(name).cloned().map(Secret::from_json))
            }
        }
    }

    pub async fn async_read_secret(&self, name: &str) -> Result<Option<SecretValue>, Error> {
        match self.client.get_secret_value().secret_id(name).send().await {
            Ok(response) => response
                .secret_string
                .map(SecretValue::new)
                .map(Some)
                .ok_or(Error::MissingString),
            Err(error)
                if matches!(
                    &error,
                    aws_sdk_secretsmanager::error::SdkError::TimeoutError(_)
                ) || matches!(&error, aws_sdk_secretsmanager::error::SdkError::DispatchFailure(failure) if failure.is_timeout()) =>
            {
                Err(Error::Timeout)
            }
            Err(error)
                if error
                    .as_service_error()
                    .is_some_and(|error| error.is_resource_not_found_exception()) =>
            {
                Ok(None)
            }
            Err(error) => Err(Error::Read(Box::new(error))),
        }
    }

    pub async fn async_write_secret(
        &self,
        name: &str,
        value: &SecretValue,
        description: Option<&str>,
    ) -> Result<CreateSecretOutput, Error> {
        let response = self
            .client
            .create_secret()
            .name(name)
            .secret_string(value.expose())
            .set_description(description.filter(|v| !v.is_empty()).map(str::to_owned))
            .set_kms_key_id(
                self.write_settings
                    .kms_key_id
                    .clone()
                    .filter(|v| !v.is_empty()),
            )
            .set_tags(self.write_settings.tags.as_ref().map(|tags| {
                tags.iter()
                    .map(|(key, value)| Tag::builder().key(key).value(value).build())
                    .collect()
            }))
            .send()
            .await
            .map_err(|error| Error::Create(Box::new(error)))?;
        if let Some(regions) = &self.write_settings.replica_regions
            && !regions.is_empty()
            && self.async_replicate_secret(name, regions).await.is_err()
        {
            tracing::warn!("secret created but replication failed");
        }
        Ok(response)
    }

    pub async fn async_replicate_secret(
        &self,
        name: &str,
        regions: &[String],
    ) -> Result<Option<ReplicateSecretToRegionsOutput>, Error> {
        if regions.is_empty() {
            return Ok(None);
        }
        self.client
            .replicate_secret_to_regions()
            .secret_id(name)
            .set_add_replica_regions(Some(
                regions
                    .iter()
                    .map(|region| ReplicaRegionType::builder().region(region).build())
                    .collect(),
            ))
            .send()
            .await
            .map(Some)
            .map_err(|error| Error::Replicate(Box::new(error)))
    }

    pub async fn async_put_secret_value(
        &self,
        name: &str,
        value: &SecretValue,
    ) -> Result<PutSecretValueOutput, Error> {
        self.client
            .put_secret_value()
            .secret_id(name)
            .secret_string(value.expose())
            .send()
            .await
            .map_err(|error| Error::Put(Box::new(error)))
    }

    pub async fn async_delete_secret(
        &self,
        name: &str,
        recovery_window_in_days: i64,
    ) -> Result<DeleteSecretOutput, Error> {
        self.client
            .delete_secret()
            .secret_id(name)
            .recovery_window_in_days(recovery_window_in_days)
            .send()
            .await
            .map_err(|error| Error::Delete(Box::new(error)))
    }

    pub async fn async_rotate_secret(
        &self,
        current_name: &str,
        new_name: &str,
        value: &SecretValue,
    ) -> Result<RotationResponse, Error> {
        if current_name == new_name {
            return self
                .async_put_secret_value(current_name, value)
                .await
                .map(RotationResponse::Updated);
        }
        async_rotate_secret(self, current_name, new_name, value)
            .await
            .map(RotationResponse::Created)
    }
}

impl BaseSecretManager for AwsSecretsManagerV2 {
    type Error = Error;
    type WriteResponse = CreateSecretOutput;
    type DeleteResponse = DeleteSecretOutput;

    async fn async_read_secret(&self, name: &str) -> Result<Option<SecretValue>, Error> {
        self.async_read_secret(name).await
    }

    async fn async_write_secret(
        &self,
        name: &str,
        value: &SecretValue,
        description: Option<&str>,
    ) -> Result<CreateSecretOutput, Error> {
        self.async_write_secret(name, value, description).await
    }

    async fn async_delete_secret(
        &self,
        name: &str,
        recovery_window_in_days: i64,
    ) -> Result<DeleteSecretOutput, Error> {
        self.async_delete_secret(name, recovery_window_in_days)
            .await
    }
}

fn bootstrap_key(name: &str) -> bool {
    matches!(
        name,
        AWS_ACCESS_KEY_ID
            | AWS_SECRET_ACCESS_KEY
            | AWS_REGION_NAME
            | AWS_REGION
            | AWS_BEDROCK_RUNTIME_ENDPOINT
    )
}
