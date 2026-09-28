use super::*;

const POLICY_LOAD_ATTEMPTS: u32 = 5;
const POLICY_LOAD_RETRY_DELAY: std::time::Duration = std::time::Duration::from_millis(200);

impl CyberArkSecretManager {
    pub async fn async_write_secret(
        &self,
        name: &str,
        value: &SecretValue,
        description: Option<&str>,
    ) -> Result<(), Error> {
        self.async_write_secret_with_context(
            name,
            value,
            description,
            &CyberarkOperationContext::default(),
        )
        .await
    }

    pub async fn async_write_secret_with_context(
        &self,
        name: &str,
        value: &SecretValue,
        _description: Option<&str>,
        context: &CyberarkOperationContext,
    ) -> Result<(), Error> {
        self.write_with_retry(name, value, context, AuthenticationRetry::Unauthorized)
            .await
            .map_err(|failure| failure.source)
    }

    pub async fn write_with_retry(
        &self,
        name: &str,
        value: &SecretValue,
        context: &CyberarkOperationContext,
        retry: AuthenticationRetry,
    ) -> Result<(), WriteFailure> {
        validate_secret_name(name).map_err(|source| WriteFailure::local(source.into()))?;
        self.ensure_variable_exists(name, context).await;
        let url = self.secret_url(name).map_err(WriteFailure::local)?;
        let response = self.post_value(&url, value, context).await?;
        let response = if matches!(retry, AuthenticationRetry::Unauthorized)
            && response.status() == reqwest::StatusCode::UNAUTHORIZED
        {
            self.token.invalidate(&()).await;
            self.post_value(&url, value, context).await?
        } else {
            response
        };
        if !response.status().is_success() {
            return Err(WriteFailure::request(
                Error::Status(response.status().as_u16()),
                url,
            ));
        }
        self.secrets.insert(name.to_owned(), value.clone()).await;
        Ok(())
    }

    async fn post_value(
        &self,
        url: &reqwest::Url,
        value: &SecretValue,
        context: &CyberarkOperationContext,
    ) -> Result<reqwest::Response, WriteFailure> {
        let authorization =
            self.authorization_header(context)
                .await
                .map_err(|source| WriteFailure {
                    source,
                    request_url: self.authentication_url().ok(),
                    authentication: true,
                })?;
        with_timeout(
            self.client
                .post(url.clone())
                .header("Authorization", authorization)
                .body(value.expose().to_owned()),
            context,
        )
        .send()
        .await
        .map_err(|source| WriteFailure::request(source.into(), url.clone()))
    }

    pub(super) async fn ensure_variable_exists(
        &self,
        name: &str,
        context: &CyberarkOperationContext,
    ) {
        let policy_url = self
            .endpoint
            .join(&format!("policies/{}/policy/root", self.account));
        let Ok(policy_url) = policy_url else {
            litellm_tracing::warn!("Could not build CyberArk policy endpoint");
            return;
        };
        let Ok(authorization) = self.authorization_header(context).await else {
            litellm_tracing::warn!(
                "Could not authenticate while ensuring CyberArk variable exists"
            );
            return;
        };
        let body = format!(
            "- !variable {}\n",
            serde_json::to_string(name).expect("serializing a string cannot fail")
        );
        let _policy_load = self.policy_load_lock.lock().await;
        for attempt in 0..POLICY_LOAD_ATTEMPTS {
            let response = with_timeout(
                self.client
                    .post(policy_url.clone())
                    .header("Authorization", authorization.clone())
                    .header("Content-Type", "application/x-yaml")
                    .body(body.clone()),
                context,
            )
            .send()
            .await;
            match response {
                Ok(response)
                    if response.status() == reqwest::StatusCode::CONFLICT
                        && attempt + 1 < POLICY_LOAD_ATTEMPTS =>
                {
                    tokio::time::sleep(POLICY_LOAD_RETRY_DELAY * 2_u32.pow(attempt)).await;
                }
                Ok(response) if response.status().is_success() => return,
                Ok(response) if response.status() == reqwest::StatusCode::UNPROCESSABLE_ENTITY => {
                    litellm_tracing::debug!(
                        "CyberArk variable policy was rejected as unprocessable"
                    );
                    return;
                }
                Ok(response) => {
                    litellm_tracing::warn!(
                        "Could not ensure CyberArk variable exists: {}",
                        response.status()
                    );
                    return;
                }
                Err(error) => {
                    litellm_tracing::warn!("Error ensuring CyberArk variable exists: {error}");
                    return;
                }
            }
        }
    }

    pub async fn async_rotate_secret(
        &self,
        current_name: &str,
        new_name: &str,
        value: &SecretValue,
    ) -> Result<(), RotationError<(), Error>> {
        self.async_rotate_secret_with_context(
            current_name,
            new_name,
            value,
            &CyberarkOperationContext::default(),
        )
        .await
    }

    pub async fn async_rotate_secret_with_context(
        &self,
        current_name: &str,
        new_name: &str,
        value: &SecretValue,
        context: &CyberarkOperationContext,
    ) -> Result<(), RotationError<(), Error>> {
        async_rotate_secret(self, current_name, new_name, value, context).await
    }

    pub async fn async_delete_secret(
        &self,
        name: &str,
        recovery_window_in_days: Option<u32>,
    ) -> Result<DeleteOutcome, Error> {
        self.async_delete_secret_with_context(
            name,
            recovery_window_in_days,
            &CyberarkOperationContext::default(),
        )
        .await
    }

    pub async fn async_delete_secret_with_context(
        &self,
        name: &str,
        _recovery_window_in_days: Option<u32>,
        _context: &CyberarkOperationContext,
    ) -> Result<DeleteOutcome, Error> {
        litellm_tracing::warn!(
            "CyberArk Conjur does not support direct secret deletion. Secrets must be removed through policy updates."
        );
        self.secrets.invalidate(&name.to_owned()).await;
        Ok(DeleteOutcome::NotSupported)
    }
}

impl SecretWriter for CyberArkSecretManager {
    type WriteResponse = ();

    async fn async_write_secret(
        &self,
        name: &str,
        value: &SecretValue,
        context: &SecretWriteContext<Self::Context>,
    ) -> Result<(), Error> {
        self.async_write_secret_with_context(
            name,
            value,
            context.description.as_deref(),
            &context.operation,
        )
        .await
    }
}

impl SecretDeleter for CyberArkSecretManager {
    type DeleteResponse = DeleteOutcome;

    async fn async_delete_secret(
        &self,
        name: &str,
        context: &Self::Context,
    ) -> Result<DeleteOutcome, Error> {
        self.async_delete_secret_with_context(name, None, context)
            .await
    }
}

impl SecretRotator for CyberArkSecretManager {
    type RotationResponse = ();

    async fn async_read_secret_fresh(
        &self,
        name: &str,
        context: &Self::Context,
    ) -> Result<Option<SecretValue>, Error> {
        validate_secret_name(name)?;
        let read = self
            .secrets
            .refresh(name.to_owned(), self.read_uncached(name, context));
        match context.timeout {
            Some(timeout) => tokio::time::timeout(timeout, read)
                .await
                .map_err(|_| Error::Timeout)?,
            None => read.await,
        }
    }

    async fn async_write_replacement(
        &self,
        current_name: &str,
        new_name: &str,
        value: &SecretValue,
        context: &Self::Context,
    ) -> Result<(), Error> {
        SecretWriter::async_write_secret(
            self,
            new_name,
            value,
            &SecretWriteContext::rotated_from(current_name, context.clone()),
        )
        .await
    }
}
