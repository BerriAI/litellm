use super::*;
use aws_sdk_secretsmanager::config::retry::RetryConfig;
use litellm_secrets_types::PythonSecretRead;

#[derive(Clone, Copy)]
enum ReadPolicy {
    Native,
    Python,
}

impl AwsSecretsManagerV2 {
    pub async fn read_secret_for_resolver(
        &self,
        name: &str,
        primary_name: Option<&str>,
        environment: &(dyn Lookup + Sync),
    ) -> Result<Option<Secret>, Error> {
        let payload = self
            .read_payload(name, primary_name, environment, ReadPolicy::Native)
            .await?;
        resolve_payload(payload, name)
    }

    pub async fn read_secret_for_python(
        &self,
        name: &str,
        primary_name: Option<&str>,
        environment: &(dyn Lookup + Sync),
    ) -> Result<Option<Secret>, Error> {
        let payload = self
            .read_payload_for_python(name, primary_name, environment)
            .await?;
        resolve_payload(payload, name)
    }

    pub async fn read_payload_for_python(
        &self,
        name: &str,
        primary_name: Option<&str>,
        environment: &(dyn Lookup + Sync),
    ) -> Result<PythonSecretRead, Error> {
        self.read_payload(name, primary_name, environment, ReadPolicy::Python)
            .await
    }

    pub async fn read_provider_payload_for_python(
        &self,
        name: &str,
        primary_name: Option<&str>,
        context: &AwsOperationContext,
        synchronous: bool,
        environment: &(dyn Lookup + Sync),
    ) -> Result<PythonSecretRead, Error> {
        if synchronous && is_bootstrap_key(name) {
            return Ok(PythonSecretRead::Value(
                environment
                    .get(name)
                    .map(SecretValue::new)
                    .map(Secret::String),
            ));
        }
        if let Some(primary) = primary_name.filter(|value| !value.is_empty()) {
            let value = if synchronous && is_bootstrap_key(primary) {
                environment.get(primary).map(SecretValue::new)
            } else {
                self.read_with_policy(primary, ReadPolicy::Python).await?
            };
            return Ok(match value.filter(|value| !value.expose().is_empty()) {
                Some(value) => PythonSecretRead::PrimaryJson(value),
                None => PythonSecretRead::Value(None),
            });
        }
        let client = self.client_for_context(context)?;
        let value = match Self::read_with_client(&client, name, ReadPolicy::Python).await {
            Err(Error::Read(_) | Error::MissingString | Error::Timeout) => None,
            result => result?,
        };
        Ok(PythonSecretRead::Value(value.map(Secret::String)))
    }

    async fn read_payload(
        &self,
        name: &str,
        primary_name: Option<&str>,
        environment: &(dyn Lookup + Sync),
        policy: ReadPolicy,
    ) -> Result<PythonSecretRead, Error> {
        if is_bootstrap_key(name) {
            return Ok(PythonSecretRead::Value(
                environment
                    .get(name)
                    .map(SecretValue::new)
                    .map(Secret::String),
            ));
        }
        match primary_name.filter(|name| !name.is_empty()) {
            None => self
                .read_with_policy(name, policy)
                .await
                .map(|value| PythonSecretRead::Value(value.map(Secret::String))),
            Some(primary) => {
                let value = if is_bootstrap_key(primary) {
                    environment.get(primary).map(SecretValue::new)
                } else {
                    self.read_with_policy(primary, policy).await?
                };
                let Some(value) = value else {
                    return Ok(PythonSecretRead::Value(None));
                };
                if matches!(policy, ReadPolicy::Python) && value.expose().is_empty() {
                    return Ok(PythonSecretRead::Value(None));
                }
                Ok(PythonSecretRead::PrimaryJson(value))
            }
        }
    }

    async fn read_with_policy(
        &self,
        name: &str,
        policy: ReadPolicy,
    ) -> Result<Option<SecretValue>, Error> {
        match (
            Self::read_with_client(&self.client, name, policy).await,
            policy,
        ) {
            (Err(Error::Read(_) | Error::MissingString | Error::Timeout), ReadPolicy::Python) => {
                Ok(None)
            }
            (result, _) => result,
        }
    }

    pub async fn async_read_secret(&self, name: &str) -> Result<Option<SecretValue>, Error> {
        Self::async_read_secret_with_client(&self.client, name).await
    }

    pub(super) async fn async_read_secret_with_client(
        client: &Client,
        name: &str,
    ) -> Result<Option<SecretValue>, Error> {
        Self::read_with_client(client, name, ReadPolicy::Native).await
    }

    async fn read_with_client(
        client: &Client,
        name: &str,
        policy: ReadPolicy,
    ) -> Result<Option<SecretValue>, Error> {
        let request = client.get_secret_value().secret_id(name);
        let response = match policy {
            ReadPolicy::Native => request.send().await,
            ReadPolicy::Python => {
                request
                    .customize()
                    .config_override(
                        aws_sdk_secretsmanager::config::Builder::new()
                            .retry_config(RetryConfig::disabled()),
                    )
                    .send()
                    .await
            }
        };
        match response {
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

pub fn is_bootstrap_key(name: &str) -> bool {
    matches!(
        name,
        AWS_ACCESS_KEY_ID
            | AWS_SECRET_ACCESS_KEY
            | AWS_REGION_NAME
            | AWS_REGION
            | AWS_BEDROCK_RUNTIME_ENDPOINT
    )
}

fn resolve_payload(payload: PythonSecretRead, name: &str) -> Result<Option<Secret>, Error> {
    match payload {
        PythonSecretRead::Value(value) => Ok(value),
        PythonSecretRead::PrimaryJson(document) => {
            let object: Value =
                serde_json::from_str(document.expose()).map_err(|_| Error::PrimarySecret)?;
            let object = object.as_object().ok_or(Error::PrimarySecret)?;
            Ok(object.get(name).cloned().map(Secret::from_json))
        }
    }
}
