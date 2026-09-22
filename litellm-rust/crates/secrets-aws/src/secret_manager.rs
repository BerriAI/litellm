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
    AwsOperationContext, BaseSecretManager, KeyManagementSettings, Secret, SecretOperationContext,
    SecretValue, SecretWriteContext, async_rotate_secret,
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

    fn with_context_client_factory(
        client: Client,
        write_settings: AwsSecretWriteSettings,
        context_client_factory: ContextClientFactory,
    ) -> Self {
        Self {
            client,
            context_client_factory: Some(Box::new(context_client_factory)),
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
        let context_client_factory = ContextClientFactory {
            settings: settings.clone(),
            environment: environment.clone(),
            endpoint_url: environment
                .get(AWS_BEDROCK_RUNTIME_ENDPOINT)
                .map(|url| url.replace("bedrock-runtime", "secretsmanager")),
        };
        let client = context_client_factory.client(&AwsOperationContext::default())?;
        Ok(Some(Self::with_context_client_factory(
            client,
            (&settings).into(),
            context_client_factory,
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
        Self::async_read_secret_with_client(&self.client, name).await
    }

    async fn async_read_secret_with_client(
        client: &Client,
        name: &str,
    ) -> Result<Option<SecretValue>, Error> {
        match client.get_secret_value().secret_id(name).send().await {
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
        self.async_write_secret_with_client_and_tags(&self.client, name, value, description, None)
            .await
    }

    async fn async_write_secret_with_client_and_tags(
        &self,
        client: &Client,
        name: &str,
        value: &SecretValue,
        description: Option<&str>,
        tags: Option<&BTreeMap<String, String>>,
    ) -> Result<CreateSecretOutput, Error> {
        let response = client
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
            .set_tags(tags.or(self.write_settings.tags.as_ref()).map(|tags| {
                tags.iter()
                    .map(|(key, value)| Tag::builder().key(key).value(value).build())
                    .collect()
            }))
            .send()
            .await
            .map_err(|error| Error::Create(Box::new(error)))?;
        if let Some(regions) = &self.write_settings.replica_regions
            && !regions.is_empty()
            && self
                .async_replicate_secret_with_client(client, name, regions)
                .await
                .is_err()
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
        self.async_replicate_secret_with_client(&self.client, name, regions)
            .await
    }

    async fn async_replicate_secret_with_client(
        &self,
        client: &Client,
        name: &str,
        regions: &[String],
    ) -> Result<Option<ReplicateSecretToRegionsOutput>, Error> {
        if regions.is_empty() {
            return Ok(None);
        }
        client
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
        self.async_put_secret_value_with_client(&self.client, name, value)
            .await
    }

    async fn async_put_secret_value_with_client(
        &self,
        client: &Client,
        name: &str,
        value: &SecretValue,
    ) -> Result<PutSecretValueOutput, Error> {
        client
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
        recovery_window_in_days: Option<u32>,
    ) -> Result<DeleteSecretOutput, Error> {
        self.async_delete_secret_with_client(&self.client, name, recovery_window_in_days)
            .await
    }

    async fn async_delete_secret_with_client(
        &self,
        client: &Client,
        name: &str,
        recovery_window_in_days: Option<u32>,
    ) -> Result<DeleteSecretOutput, Error> {
        client
            .delete_secret()
            .secret_id(name)
            .set_recovery_window_in_days(recovery_window_in_days.map(i64::from))
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
        self.async_rotate_secret_with_context(
            current_name,
            new_name,
            value,
            &SecretOperationContext::default(),
        )
        .await
    }

    pub async fn async_rotate_secret_with_context(
        &self,
        current_name: &str,
        new_name: &str,
        value: &SecretValue,
        context: &SecretOperationContext,
    ) -> Result<RotationResponse, Error> {
        if current_name == new_name {
            let client = self.client_for_context(context)?;
            return self
                .async_put_secret_value_with_client(&client, current_name, value)
                .await
                .map(RotationResponse::Updated);
        }
        async_rotate_secret(self, current_name, new_name, value, context)
            .await
            .map(RotationResponse::Created)
    }

    fn client_for_context(&self, context: &SecretOperationContext) -> Result<Client, Error> {
        match context {
            SecretOperationContext::Default => Ok(self.client.clone()),
            SecretOperationContext::Aws(context) if context == &AwsOperationContext::default() => {
                Ok(self.client.clone())
            }
            SecretOperationContext::Aws(context) => self
                .context_client_factory
                .as_ref()
                .ok_or(Error::OperationContextUnavailable)?
                .client(context),
            _ => Err(Error::InvalidOperationContext),
        }
    }
}

impl ContextClientFactory {
    fn client(&self, context: &AwsOperationContext) -> Result<Client, Error> {
        let settings = KeyManagementSettings {
            aws_region_name: context
                .region_name
                .clone()
                .or_else(|| self.settings.aws_region_name.clone()),
            aws_role_name: context
                .role_name
                .clone()
                .or_else(|| self.settings.aws_role_name.clone()),
            aws_session_name: context
                .session_name
                .clone()
                .or_else(|| self.settings.aws_session_name.clone()),
            aws_external_id: context
                .external_id
                .clone()
                .or_else(|| self.settings.aws_external_id.clone()),
            aws_profile_name: context
                .profile_name
                .clone()
                .or_else(|| self.settings.aws_profile_name.clone()),
            aws_web_identity_token: context
                .web_identity_token
                .clone()
                .or_else(|| self.settings.aws_web_identity_token.clone()),
            aws_sts_endpoint: context
                .sts_endpoint
                .clone()
                .or_else(|| self.settings.aws_sts_endpoint.clone()),
            ..self.settings.clone()
        };
        let builder = aws_sdk_secretsmanager::Config::builder()
            .behavior_version(BehaviorVersion::latest())
            .region(Region::new(auth::region(
                &settings,
                self.environment.as_ref(),
            )?))
            .credentials_provider(auth::Credentials::new(&settings, self.environment.clone()));
        let builder = match context.timeout {
            Some(timeout) => builder.timeout_config(
                aws_sdk_secretsmanager::config::timeout::TimeoutConfig::builder()
                    .operation_timeout(timeout)
                    .build(),
            ),
            None => builder,
        };
        let config = match &self.endpoint_url {
            Some(endpoint_url) => builder.endpoint_url(endpoint_url.clone()).build(),
            None => builder.build(),
        };
        Ok(Client::from_conf(config))
    }
}

impl BaseSecretManager for AwsSecretsManagerV2 {
    type Error = Error;
    type WriteResponse = CreateSecretOutput;
    type DeleteResponse = DeleteSecretOutput;

    async fn async_read_secret(
        &self,
        name: &str,
        context: &SecretOperationContext,
    ) -> Result<Option<SecretValue>, Error> {
        let client = self.client_for_context(context)?;
        Self::async_read_secret_with_client(&client, name).await
    }

    async fn async_write_secret(
        &self,
        name: &str,
        value: &SecretValue,
        context: &SecretWriteContext,
    ) -> Result<CreateSecretOutput, Error> {
        let client = self.client_for_context(&context.operation)?;
        self.async_write_secret_with_client_and_tags(
            &client,
            name,
            value,
            context.description.as_deref(),
            (!context.tags.is_empty()).then_some(&context.tags),
        )
        .await
    }

    async fn async_delete_secret(
        &self,
        name: &str,
        recovery_window_in_days: Option<u32>,
        context: &SecretOperationContext,
    ) -> Result<DeleteSecretOutput, Error> {
        let client = self.client_for_context(context)?;
        self.async_delete_secret_with_client(&client, name, recovery_window_in_days)
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
