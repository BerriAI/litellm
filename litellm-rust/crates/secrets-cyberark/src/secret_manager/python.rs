use super::*;

pub(super) enum AuthenticationRetry {
    Never,
    Unauthorized,
}

#[derive(veil::Redact)]
pub struct PythonWriteFailure {
    pub source: Error,
    #[redact]
    pub request_url: Option<reqwest::Url>,
    pub authentication: bool,
}

impl PythonWriteFailure {
    pub(super) fn local(source: Error) -> Self {
        Self {
            source,
            request_url: None,
            authentication: false,
        }
    }

    pub(super) fn request(source: Error, url: reqwest::Url) -> Self {
        Self {
            source,
            request_url: Some(url),
            authentication: false,
        }
    }
}

#[derive(Debug)]
pub enum PythonRotationFailure {
    CurrentMissing,
    Write(PythonWriteFailure),
    ReplacementMissing,
    ReplacementMismatch,
}

impl CyberArkSecretManager {
    pub async fn read_for_python(&self, name: &str) -> Result<Option<SecretValue>, Error> {
        self.read_with_retry(
            name,
            &CyberarkOperationContext::default(),
            AuthenticationRetry::Never,
        )
        .await
    }

    pub async fn write_for_python(
        &self,
        name: &str,
        value: &SecretValue,
    ) -> Result<(), PythonWriteFailure> {
        self.write_with_retry(
            name,
            value,
            &CyberarkOperationContext::default(),
            AuthenticationRetry::Never,
        )
        .await
    }

    async fn read_fresh_for_python(&self, name: &str) -> Option<SecretValue> {
        validate_secret_name(name).ok()?;
        self.secrets
            .refresh(
                name.to_owned(),
                self.read_uncached_with_retry(
                    name,
                    &CyberarkOperationContext::default(),
                    AuthenticationRetry::Never,
                ),
            )
            .await
            .ok()
            .flatten()
    }

    pub async fn rotate_for_python(
        &self,
        current_name: &str,
        new_name: &str,
        value: &SecretValue,
    ) -> Result<(), PythonRotationFailure> {
        if self.read_fresh_for_python(current_name).await.is_none() {
            return Err(PythonRotationFailure::CurrentMissing);
        }
        self.write_for_python(new_name, value)
            .await
            .map_err(PythonRotationFailure::Write)?;
        let Some(actual) = self.read_fresh_for_python(new_name).await else {
            return Err(PythonRotationFailure::ReplacementMissing);
        };
        if actual != *value {
            return Err(PythonRotationFailure::ReplacementMismatch);
        }
        if current_name != new_name {
            self.secrets.invalidate(&current_name.to_owned()).await;
        }
        Ok(())
    }
}
