use super::*;
use rustify::{client::Client as _, endpoint::Endpoint};
use vaultrs::api::kv2::requests::{
    DeleteLatestSecretVersionRequest, ReadSecretRequest, SetSecretRequest,
};

#[derive(Debug)]
pub enum PythonFailureStage {
    Mutation,
    Current(String),
    Replacement(String),
}

#[derive(veil::Redact)]
pub enum PythonFailureKind {
    Local(Error),
    UnsafeName(#[redact] String),
    Authentication {
        source: Error,
        url: String,
        certificate: bool,
    },
    Http {
        method: String,
        url: String,
        status: u16,
        #[redact]
        body: Vec<u8>,
    },
    Transport(#[redact] RustifyClientError),
    Timeout {
        method: String,
        elapsed: Duration,
    },
    Json(#[redact] Vec<u8>),
    MissingGet(#[redact] Value),
    ValueMismatch {
        expected: SecretValue,
        #[redact]
        actual: Value,
    },
}

#[derive(Debug)]
pub struct PythonFailure {
    pub kind: Box<PythonFailureKind>,
    pub stage: PythonFailureStage,
}

impl From<PythonFailureKind> for PythonFailure {
    fn from(kind: PythonFailureKind) -> Self {
        Self {
            kind: Box::new(kind),
            stage: PythonFailureStage::Mutation,
        }
    }
}

impl PythonFailure {
    fn during(self, stage: PythonFailureStage) -> Self {
        Self { stage, ..self }
    }
}

impl HashicorpVault {
    pub async fn write_for_python(
        &self,
        name: &str,
        value: &SecretValue,
        context: &SecretWriteContext<HashicorpOperationContext>,
    ) -> Result<Vec<u8>, PythonFailure> {
        let location = self.python_location(name, &context.operation)?;
        let data = super::write::write_data(value, context).map_err(PythonFailureKind::Local)?;
        let response = self
            .python_request(
                &location,
                SetSecretRequest {
                    mount: location.mount.clone(),
                    path: location.path.clone(),
                    data,
                    options: None,
                },
                &context.operation,
            )
            .await?;
        self.cache
            .invalidate_where(move |key| key.location == location);
        Ok(response)
    }

    pub async fn delete_for_python(
        &self,
        name: &str,
        context: &HashicorpOperationContext,
    ) -> Result<(), PythonFailure> {
        let location = self.python_location(name, context)?;
        self.python_request(
            &location,
            DeleteLatestSecretVersionRequest {
                mount: location.mount.clone(),
                path: location.path.clone(),
            },
            context,
        )
        .await?;
        self.cache
            .invalidate_where(move |key| key.location == location);
        Ok(())
    }

    pub async fn rotate_for_python(
        &self,
        current_name: &str,
        new_name: &str,
        value: &SecretValue,
        context: &HashicorpOperationContext,
    ) -> Result<Vec<u8>, PythonFailure> {
        let current_location = self.python_location(current_name, context)?;
        self.python_read_response(&current_location, context)
            .await
            .map_err(|failure| {
                failure.during(PythonFailureStage::Current(current_name.to_owned()))
            })?;
        let response = self
            .write_for_python(
                new_name,
                value,
                &SecretWriteContext {
                    description: Some(format!("Rotated from {current_name}")),
                    operation: context.clone(),
                    ..SecretWriteContext::default()
                },
            )
            .await?;
        let parsed: Value = serde_json::from_slice(&response)
            .map_err(|_| PythonFailureKind::Json(response.clone()))?;
        if parsed.get("status").and_then(Value::as_str) == Some("error") {
            return Ok(response);
        }
        let new_location = self.python_location(new_name, context)?;
        let verification = self
            .python_read_response(&new_location, context)
            .await
            .map_err(|failure| {
                failure.during(PythonFailureStage::Replacement(new_name.to_owned()))
            })?;
        let parsed: Value = serde_json::from_slice(&verification).map_err(|_| {
            PythonFailure::from(PythonFailureKind::Json(verification))
                .during(PythonFailureStage::Replacement(new_name.to_owned()))
        })?;
        let actual = verification_value(&parsed, &data_key(context)).map_err(|failure| {
            PythonFailure::from(failure)
                .during(PythonFailureStage::Replacement(new_name.to_owned()))
        })?;
        if actual.as_str() != Some(value.expose()) {
            return Err(PythonFailureKind::ValueMismatch {
                expected: value.clone(),
                actual,
            }
            .into());
        }
        if current_name != new_name {
            let _ = self.delete_for_python(current_name, context).await;
        }
        self.cache
            .invalidate_where(move |key| key.location == new_location);
        Ok(response)
    }

    fn python_location(
        &self,
        name: &str,
        context: &HashicorpOperationContext,
    ) -> Result<SecretLocation, PythonFailure> {
        self.secret_location_with_context(name, context)
            .map_err(|error| {
                match error {
                    Error::InvalidSecretName(_) => PythonFailureKind::UnsafeName(name.to_owned()),
                    error => PythonFailureKind::Local(error),
                }
                .into()
            })
    }

    async fn python_read_response(
        &self,
        location: &SecretLocation,
        context: &HashicorpOperationContext,
    ) -> Result<Vec<u8>, PythonFailure> {
        self.python_request(
            location,
            ReadSecretRequest {
                mount: location.mount.clone(),
                path: location.path.clone(),
                version: None,
            },
            context,
        )
        .await
    }

    async fn python_request(
        &self,
        location: &SecretLocation,
        endpoint: impl Endpoint,
        context: &HashicorpOperationContext,
    ) -> Result<Vec<u8>, PythonFailure> {
        let client = self.client_for_location(location).await.map_err(|source| {
            let certificate = self.config.approle.is_none();
            let mount = self
                .config
                .approle
                .as_ref()
                .map_or("cert", |auth| auth.mount_path.as_str());
            PythonFailureKind::Authentication {
                source,
                certificate,
                url: format!("{}/v1/auth/{mount}/login", self.config.address),
            }
        })?;
        let request = endpoint
            .with_middleware(&client.middle)
            .request(client.http.base())
            .map_err(PythonFailureKind::Transport)?;
        let method = request.method().to_string();
        let url = format!(
            "{}/v1/{}{}/data/{}",
            self.config.address,
            location
                .namespace
                .as_ref()
                .map(|ns| format!("{ns}/"))
                .unwrap_or_default(),
            location.mount,
            location.path
        );
        let started = Instant::now();
        let response = match context.timeout {
            Some(timeout) => tokio::time::timeout(timeout, client.http.send(request))
                .await
                .map_err(|_| PythonFailureKind::Timeout {
                    method: method.clone(),
                    elapsed: started.elapsed(),
                })?,
            None => client.http.send(request).await,
        }
        .map_err(PythonFailureKind::Transport)?;
        if !response.status().is_success() {
            return Err(PythonFailureKind::Http {
                method,
                url,
                status: response.status().as_u16(),
                body: response.into_body(),
            }
            .into());
        }
        Ok(response.into_body())
    }
}

fn verification_value(document: &Value, key: &str) -> Result<Value, PythonFailureKind> {
    let object = document
        .as_object()
        .ok_or_else(|| PythonFailureKind::MissingGet(document.clone()))?;
    let Some(outer) = object.get("data") else {
        return Ok(Value::Null);
    };
    let object = outer
        .as_object()
        .ok_or_else(|| PythonFailureKind::MissingGet(outer.clone()))?;
    let Some(inner) = object.get("data") else {
        return Ok(Value::Null);
    };
    let object = inner
        .as_object()
        .ok_or_else(|| PythonFailureKind::MissingGet(inner.clone()))?;
    Ok(object.get(key).cloned().unwrap_or(Value::Null))
}
