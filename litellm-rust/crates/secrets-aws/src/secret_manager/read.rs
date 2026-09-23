use super::*;

impl AwsSecretsManagerV2 {
    pub async fn read_secret_for_resolver(
        &self,
        name: &str,
        primary_name: Option<&str>,
        environment: &(dyn Lookup + Sync),
    ) -> Result<Option<Secret>, Error> {
        self.read_secret_for_resolver_with_policy(name, primary_name, environment, false)
            .await
    }

    pub async fn read_secret_for_python(
        &self,
        name: &str,
        primary_name: Option<&str>,
        environment: &(dyn Lookup + Sync),
    ) -> Result<Option<Secret>, Error> {
        self.read_secret_for_resolver_with_policy(name, primary_name, environment, true)
            .await
    }

    pub(super) async fn read_secret_for_resolver_with_policy(
        &self,
        name: &str,
        primary_name: Option<&str>,
        environment: &(dyn Lookup + Sync),
        missing_is_error: bool,
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
                .and_then(|value| match value {
                    None if missing_is_error => Err(Error::MissingString),
                    value => Ok(value.map(Secret::String)),
                }),
            Some(primary) => {
                let value = if bootstrap_key(primary) {
                    environment.get(primary).map(SecretValue::new)
                } else {
                    self.async_read_secret(primary).await?
                };
                let Some(value) = value else {
                    return if missing_is_error {
                        Err(Error::PrimarySecret)
                    } else {
                        Ok(None)
                    };
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

    pub(super) async fn async_read_secret_with_client(
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
}

impl BaseSecretManager for AwsSecretsManagerV2 {
    type Error = Error;
    type Context = AwsOperationContext;

    async fn async_read_secret(
        &self,
        name: &str,
        context: &Self::Context,
    ) -> Result<Option<SecretValue>, Error> {
        let client = self.client_for_context(context)?;
        Self::async_read_secret_with_client(&client, name).await
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
