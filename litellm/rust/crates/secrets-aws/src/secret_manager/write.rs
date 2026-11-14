use super::*;

impl AwsSecretsManagerV2 {
    pub async fn async_write_secret(
        &self,
        name: &str,
        value: &SecretValue,
        description: Option<&str>,
    ) -> Result<CreateSecretOutput, Error> {
        self.async_write_secret_with_client_and_tags(&self.client, name, value, description, None)
            .await
    }

    pub(super) async fn async_write_secret_with_client_and_tags(
        &self,
        client: &Client,
        name: &str,
        value: &SecretValue,
        description: Option<&str>,
        tags: Option<&BTreeMap<String, String>>,
    ) -> Result<CreateSecretOutput, Error> {
        let tags = self.write_tags(tags);
        let request = client
            .create_secret()
            .name(name)
            .secret_string(value.expose())
            .set_description(description.filter(|v| !v.is_empty()).map(str::to_owned))
            .set_kms_key_id(self.write_kms_key_id())
            .set_tags(tags.clone());
        let response = match request.send().await {
            Ok(response) => response,
            Err(error) => self
                .restore_and_update_secret(client, name, value, description, tags)
                .await?
                .ok_or_else(|| Error::Create(Box::new(error)))?,
        };
        if let Some(regions) = &self.write_settings.replica_regions
            && !regions.is_empty()
            && self
                .async_replicate_secret_with_client(client, name, regions)
                .await
                .is_err()
        {
            litellm_tracing::warn!("secret created but replication failed");
        }
        Ok(response)
    }

    async fn restore_and_update_secret(
        &self,
        client: &Client,
        name: &str,
        value: &SecretValue,
        description: Option<&str>,
        tags: Option<Vec<Tag>>,
    ) -> Result<Option<CreateSecretOutput>, Error> {
        let scheduled = client
            .describe_secret()
            .secret_id(name)
            .send()
            .await
            .is_ok_and(|response| response.deleted_date().is_some());
        if !scheduled {
            return Ok(None);
        }
        client
            .restore_secret()
            .secret_id(name)
            .send()
            .await
            .map_err(|error| Error::Restore(Box::new(error)))?;
        match self
            .update_restored_secret(client, name, value, description, tags)
            .await
        {
            Ok(response) => Ok(Some(response)),
            Err(error) => {
                self.async_delete_secret_with_client(client, name, Some(7))
                    .await?;
                Err(error)
            }
        }
    }

    fn write_kms_key_id(&self) -> Option<String> {
        self.write_settings
            .kms_key_id
            .clone()
            .filter(|value| !value.is_empty())
    }

    fn write_tags(&self, tags: Option<&BTreeMap<String, String>>) -> Option<Vec<Tag>> {
        tags.or(self.write_settings.tags.as_ref()).map(|tags| {
            tags.iter()
                .map(|(key, value)| Tag::builder().key(key).value(value).build())
                .collect()
        })
    }

    async fn update_restored_secret(
        &self,
        client: &Client,
        name: &str,
        value: &SecretValue,
        description: Option<&str>,
        tags: Option<Vec<Tag>>,
    ) -> Result<CreateSecretOutput, Error> {
        let response = client
            .update_secret()
            .secret_id(name)
            .secret_string(value.expose())
            .set_description(
                description
                    .filter(|value| !value.is_empty())
                    .map(str::to_owned),
            )
            .set_kms_key_id(self.write_kms_key_id())
            .send()
            .await
            .map_err(|error| Error::Update(Box::new(error)))?;
        if let Some(tags) = tags {
            client
                .tag_resource()
                .secret_id(name)
                .set_tags(Some(tags))
                .send()
                .await
                .map_err(|error| Error::Tag(Box::new(error)))?;
        }
        Ok(CreateSecretOutput::builder()
            .set_arn(response.arn)
            .set_name(response.name)
            .set_version_id(response.version_id)
            .build())
    }

    pub async fn async_replicate_secret(
        &self,
        name: &str,
        regions: &[String],
    ) -> Result<Option<ReplicateSecretToRegionsOutput>, Error> {
        self.async_replicate_secret_with_client(&self.client, name, regions)
            .await
    }

    pub(super) async fn async_replicate_secret_with_client(
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

    pub(super) async fn async_put_secret_value_with_client(
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

    pub async fn async_delete_secret_with_context(
        &self,
        name: &str,
        recovery_window_in_days: Option<u32>,
        context: &AwsOperationContext,
    ) -> Result<DeleteSecretOutput, Error> {
        let client = self.client_for_context(context)?;
        self.async_delete_secret_with_client(&client, name, recovery_window_in_days)
            .await
    }

    pub(super) async fn async_delete_secret_with_client(
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
    ) -> Result<RotationResponse, RotationError<RotationResponse, Error>> {
        self.async_rotate_secret_with_context(
            current_name,
            new_name,
            value,
            &AwsOperationContext::default(),
        )
        .await
    }

    pub async fn async_rotate_secret_with_context(
        &self,
        current_name: &str,
        new_name: &str,
        value: &SecretValue,
        context: &AwsOperationContext,
    ) -> Result<RotationResponse, RotationError<RotationResponse, Error>> {
        if current_name == new_name {
            return self
                .async_write_replacement(current_name, new_name, value, context)
                .await
                .map_err(RotationError::Write);
        }
        async_rotate_secret(self, current_name, new_name, value, context).await
    }
}

impl SecretWriter for AwsSecretsManagerV2 {
    type WriteResponse = CreateSecretOutput;

    async fn async_write_secret(
        &self,
        name: &str,
        value: &SecretValue,
        context: &SecretWriteContext<Self::Context>,
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
}

impl SecretDeleter for AwsSecretsManagerV2 {
    type DeleteResponse = DeleteSecretOutput;

    async fn async_delete_secret(
        &self,
        name: &str,
        context: &Self::Context,
    ) -> Result<DeleteSecretOutput, Error> {
        self.async_delete_secret_with_context(name, Some(7), context)
            .await
    }
}

impl SecretRotator for AwsSecretsManagerV2 {
    type RotationResponse = RotationResponse;

    async fn async_read_secret_fresh(
        &self,
        name: &str,
        context: &Self::Context,
    ) -> Result<Option<SecretValue>, Error> {
        BaseSecretManager::async_read_secret(self, name, context).await
    }

    async fn async_write_replacement(
        &self,
        current_name: &str,
        new_name: &str,
        value: &SecretValue,
        context: &Self::Context,
    ) -> Result<RotationResponse, Error> {
        if current_name == new_name {
            let client = self.client_for_context(context)?;
            return self
                .async_put_secret_value_with_client(&client, new_name, value)
                .await
                .map(RotationResponse::Updated);
        }
        SecretWriter::async_write_secret(
            self,
            new_name,
            value,
            &SecretWriteContext::rotated_from(current_name, context.clone()),
        )
        .await
        .map(RotationResponse::Created)
    }
}
