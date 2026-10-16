use super::*;

impl HashicorpVault {
    pub async fn async_read_secret(&self, secret_name: &str) -> Result<Option<SecretValue>, Error> {
        self.async_read_secret_with_context(secret_name, &HashicorpOperationContext::default())
            .await
    }

    pub async fn async_read_secret_with_context(
        &self,
        secret_name: &str,
        context: &HashicorpOperationContext,
    ) -> Result<Option<SecretValue>, Error> {
        let location: SecretLocation = self.secret_location_with_context(secret_name, context)?;
        let data_key: String = data_key(context);
        let cache_key = CacheKey {
            location: location.clone(),
            data_key: data_key.clone(),
        };
        with_timeout(
            context,
            self.cache
                .read(cache_key, self.read_uncached(&location, &data_key)),
        )
        .await
    }

    pub(super) async fn read_uncached(
        &self,
        location: &SecretLocation,
        data_key: &str,
    ) -> Result<Option<SecretValue>, Error> {
        let client = self.client_for_location(location).await?;
        let data: Option<HashMap<String, Value>> =
            match kv2::read(client.as_ref(), &location.mount, &location.path).await {
                Ok(data) => Some(data),
                Err(error) if api_status(&error) == Some(404) => None,
                Err(error) => return Err(map_api_error(error, ErrorContext::Read)),
            };
        let Some(data) = data else {
            return Ok(None);
        };
        let Some(value) = data.get(data_key) else {
            return Ok(None);
        };
        let value: &str = value.as_str().ok_or(Error::NonStringValue)?;
        let value: SecretValue = SecretValue::new(value);
        Ok(Some(value))
    }
}

impl BaseSecretManager for HashicorpVault {
    type Error = Error;
    type Context = HashicorpOperationContext;

    async fn async_read_secret(
        &self,
        name: &str,
        context: &Self::Context,
    ) -> Result<Option<SecretValue>, Error> {
        HashicorpVault::async_read_secret_with_context(self, name, context).await
    }
}
