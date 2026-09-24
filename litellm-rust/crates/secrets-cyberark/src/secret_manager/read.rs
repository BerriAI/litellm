use super::*;

impl CyberArkSecretManager {
    pub async fn async_read_secret(&self, name: &str) -> Result<Option<SecretValue>, Error> {
        self.async_read_secret_with_context(name, &CyberarkOperationContext::default())
            .await
    }

    pub async fn async_read_secret_with_context(
        &self,
        name: &str,
        context: &CyberarkOperationContext,
    ) -> Result<Option<SecretValue>, Error> {
        self.read_with_retry(name, context, AuthenticationRetry::Unauthorized)
            .await
    }

    pub async fn read_with_retry(
        &self,
        name: &str,
        context: &CyberarkOperationContext,
        retry: AuthenticationRetry,
    ) -> Result<Option<SecretValue>, Error> {
        validate_secret_name(name)?;
        let read = self.secrets.read(
            name.to_owned(),
            self.read_uncached_with_retry(name, context, retry),
        );
        match context.timeout {
            Some(timeout) => tokio::time::timeout(timeout, read)
                .await
                .map_err(|_| Error::Timeout)?,
            None => read.await,
        }
    }

    pub(super) async fn read_uncached(
        &self,
        name: &str,
        context: &CyberarkOperationContext,
    ) -> Result<Option<SecretValue>, Error> {
        self.read_uncached_with_retry(name, context, AuthenticationRetry::Unauthorized)
            .await
    }

    pub async fn read_fresh_with_retry(
        &self,
        name: &str,
        context: &CyberarkOperationContext,
        retry: AuthenticationRetry,
    ) -> Result<Option<SecretValue>, Error> {
        validate_secret_name(name)?;
        self.secrets
            .refresh(
                name.to_owned(),
                self.read_uncached_with_retry(name, context, retry),
            )
            .await
    }

    pub async fn invalidate_cached_secret(&self, name: &str) {
        self.secrets.invalidate(&name.to_owned()).await;
    }

    pub(super) async fn read_uncached_with_retry(
        &self,
        name: &str,
        context: &CyberarkOperationContext,
        retry: AuthenticationRetry,
    ) -> Result<Option<SecretValue>, Error> {
        let had_cached_token = self.token.get(&()).await.is_some();
        let response = with_timeout(
            self.client
                .get(self.secret_url(name)?)
                .header("Authorization", self.authorization_header(context).await?),
            context,
        )
        .send()
        .await?;
        let response = if matches!(retry, AuthenticationRetry::Unauthorized)
            && had_cached_token
            && response.status() == reqwest::StatusCode::UNAUTHORIZED
        {
            self.token.invalidate(&()).await;
            with_timeout(
                self.client
                    .get(self.secret_url(name)?)
                    .header("Authorization", self.authorization_header(context).await?),
                context,
            )
            .send()
            .await?
        } else {
            response
        };
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !response.status().is_success() {
            return Err(Error::Status(response.status().as_u16()));
        }
        let value = SecretValue::new(response.text().await?);
        Ok(Some(value))
    }
}

impl BaseSecretManager for CyberArkSecretManager {
    type Error = Error;
    type Context = CyberarkOperationContext;

    async fn async_read_secret(
        &self,
        name: &str,
        context: &Self::Context,
    ) -> Result<Option<SecretValue>, Error> {
        self.async_read_secret_with_context(name, context).await
    }
}
