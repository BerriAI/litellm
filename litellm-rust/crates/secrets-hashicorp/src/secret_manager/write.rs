use super::*;

impl HashicorpVault {
    pub async fn async_write_secret(
        &self,
        secret_name: &str,
        value: SecretValue,
        description: Option<&str>,
    ) -> Result<Value, Error> {
        self.async_write_secret_with_context(
            secret_name,
            &value,
            &SecretWriteContext {
                description: description.map(str::to_owned),
                ..SecretWriteContext::default()
            },
        )
        .await
    }

    pub async fn async_write_secret_with_context(
        &self,
        secret_name: &str,
        value: &SecretValue,
        context: &SecretWriteContext<HashicorpOperationContext>,
    ) -> Result<Value, Error> {
        let location: SecretLocation =
            self.secret_location_with_context(secret_name, &context.operation)?;
        let data = write_data(value, context)?;
        let metadata = with_timeout(&context.operation, async {
            let client = self.client_for_location(&location).await?;
            match kv2::set(client.as_ref(), &location.mount, &location.path, &data).await {
                Ok(metadata) => Ok(metadata),
                Err(error) if api_status(&error) == Some(400) => {
                    let version =
                        match kv2::read_metadata(client.as_ref(), &location.mount, &location.path)
                            .await
                        {
                            Ok(metadata) => u32::try_from(metadata.current_version)
                                .map_err(|_| Error::CasVersionOverflow)?,
                            Err(error) if api_status(&error) == Some(404) => 0,
                            Err(_) => return Err(map_api_error(error, ErrorContext::Secret)),
                        };
                    kv2::set_with_options(
                        client.as_ref(),
                        &location.mount,
                        &location.path,
                        &data,
                        SetSecretRequestOptions { cas: version },
                    )
                    .await
                    .map_err(|error| map_api_error(error, ErrorContext::Secret))
                }
                Err(error) => Err(map_api_error(error, ErrorContext::Secret)),
            }
        })
        .await?;
        self.cache
            .invalidate_where(move |key| key.location == location);
        serde_json::to_value(metadata)
            .map_err(|source| Error::Client(ClientError::JsonParseError { source }))
    }

    pub async fn async_delete_secret(&self, secret_name: &str) -> Result<(), Error> {
        self.async_delete_secret_with_context(secret_name, &HashicorpOperationContext::default())
            .await
    }

    pub async fn async_delete_secret_with_context(
        &self,
        secret_name: &str,
        context: &HashicorpOperationContext,
    ) -> Result<(), Error> {
        let location: SecretLocation = self.secret_location_with_context(secret_name, context)?;
        with_timeout(context, async {
            let client = self.client_for_location(&location).await?;
            kv2::delete_latest(client.as_ref(), &location.mount, &location.path)
                .await
                .map_err(|error| map_api_error(error, ErrorContext::Secret))
        })
        .await?;
        self.cache
            .invalidate_where(move |key| key.location == location);
        Ok(())
    }

    pub async fn async_rotate_secret(
        &self,
        current_name: &str,
        new_name: &str,
        value: &SecretValue,
    ) -> Result<Value, RotationError<Value, Error>> {
        self.async_rotate_secret_with_context(
            current_name,
            new_name,
            value,
            &HashicorpOperationContext::default(),
        )
        .await
    }

    pub async fn async_rotate_secret_with_context(
        &self,
        current_name: &str,
        new_name: &str,
        value: &SecretValue,
        context: &HashicorpOperationContext,
    ) -> Result<Value, RotationError<Value, Error>> {
        async_rotate_secret(self, current_name, new_name, value, context).await
    }
}

impl SecretWriter for HashicorpVault {
    type WriteResponse = Value;

    async fn async_write_secret(
        &self,
        name: &str,
        value: &SecretValue,
        context: &SecretWriteContext<Self::Context>,
    ) -> Result<Value, Error> {
        HashicorpVault::async_write_secret_with_context(self, name, value, context).await
    }
}

impl SecretDeleter for HashicorpVault {
    type DeleteResponse = ();

    async fn async_delete_secret(&self, name: &str, context: &Self::Context) -> Result<(), Error> {
        HashicorpVault::async_delete_secret_with_context(self, name, context).await
    }
}

impl SecretRotator for HashicorpVault {
    type RotationResponse = Value;

    async fn async_read_secret_fresh(
        &self,
        name: &str,
        context: &Self::Context,
    ) -> Result<Option<SecretValue>, Error> {
        let location = self.secret_location_with_context(name, context)?;
        let data_key = data_key(context);
        let key = CacheKey {
            location: location.clone(),
            data_key: data_key.clone(),
        };
        with_timeout(
            context,
            self.cache
                .refresh(key, self.read_uncached(&location, &data_key)),
        )
        .await
    }

    async fn async_write_replacement(
        &self,
        current_name: &str,
        new_name: &str,
        value: &SecretValue,
        context: &Self::Context,
    ) -> Result<Value, Error> {
        SecretWriter::async_write_secret(
            self,
            new_name,
            value,
            &SecretWriteContext::rotated_from(current_name, context.clone()),
        )
        .await
    }
}

pub(super) fn write_data(
    value: &SecretValue,
    context: &SecretWriteContext<HashicorpOperationContext>,
) -> Result<Value, Error> {
    let data_key = data_key(&context.operation);
    if context.description.is_some() && data_key == "description" {
        return Err(Error::DataKeyConflictsWithDescription);
    }
    let data = std::iter::once((data_key, Value::String(value.expose().to_owned())))
        .chain(
            context
                .description
                .as_ref()
                .map(|description| ("description".to_owned(), Value::String(description.clone()))),
        )
        .collect();
    Ok(Value::Object(data))
}
