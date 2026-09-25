use std::collections::HashMap;

use litellm_secrets::{
    SecretValue,
    hashicorp::{Error, HashicorpVault, RawOperationError},
};
use litellm_secrets_types::{HashicorpOperationContext, SecretWriteContext};
use serde_json::value::RawValue;

#[derive(Debug)]
pub(crate) enum FailureStage {
    Mutation,
    Current(String),
    Replacement(String),
}

#[derive(veil::Redact)]
pub(crate) enum FailureKind {
    Native(RawOperationError),
    UnsafeName(#[redact] String),
    Json(#[redact] Vec<u8>),
    MissingGet(#[redact] Vec<u8>),
    ValueMismatch {
        expected: SecretValue,
        #[redact]
        actual: Vec<u8>,
    },
}

#[derive(Debug)]
pub(crate) struct Failure {
    pub kind: Box<FailureKind>,
    pub stage: FailureStage,
}

impl From<FailureKind> for Failure {
    fn from(kind: FailureKind) -> Self {
        Self {
            kind: Box::new(kind),
            stage: FailureStage::Mutation,
        }
    }
}

impl Failure {
    fn during(self, stage: FailureStage) -> Self {
        if matches!(*self.kind, FailureKind::UnsafeName(_)) {
            self
        } else {
            Self { stage, ..self }
        }
    }
}

fn native_failure(name: &str, error: RawOperationError) -> Failure {
    match error {
        RawOperationError::Local(Error::InvalidSecretName(_)) => {
            FailureKind::UnsafeName(name.to_owned()).into()
        }
        error => FailureKind::Native(error).into(),
    }
}

pub(crate) async fn write(
    client: &HashicorpVault,
    name: &str,
    value: &SecretValue,
    context: &SecretWriteContext<HashicorpOperationContext>,
) -> Result<Vec<u8>, Failure> {
    client
        .write_raw(name, value, context)
        .await
        .map_err(|error| native_failure(name, error))
}

pub(crate) async fn delete(
    client: &HashicorpVault,
    name: &str,
    context: &HashicorpOperationContext,
) -> Result<(), Failure> {
    client
        .delete_raw(name, context)
        .await
        .map_err(|error| native_failure(name, error))
}

pub(crate) async fn rotate(
    client: &HashicorpVault,
    current_name: &str,
    new_name: &str,
    value: &SecretValue,
    context: &HashicorpOperationContext,
) -> Result<Vec<u8>, Failure> {
    client
        .read_raw(current_name, context)
        .await
        .map_err(|error| {
            native_failure(current_name, error)
                .during(FailureStage::Current(current_name.to_owned()))
        })?;
    let response = write(
        client,
        new_name,
        value,
        &SecretWriteContext {
            description: Some(format!("Rotated from {current_name}")),
            operation: context.clone(),
            ..SecretWriteContext::default()
        },
    )
    .await?;
    let parsed: &RawValue =
        serde_json::from_slice(&response).map_err(|_| FailureKind::Json(response.clone()))?;
    let status = raw_object_field(parsed, "status")
        .ok()
        .flatten()
        .and_then(|value| serde_json::from_slice::<String>(value.get().as_bytes()).ok());
    if status.as_deref() == Some("error") {
        return Ok(response);
    }
    let verification = client.read_raw(new_name, context).await.map_err(|error| {
        native_failure(new_name, error).during(FailureStage::Replacement(new_name.to_owned()))
    })?;
    let parsed: &RawValue = serde_json::from_slice(&verification).map_err(|_| {
        Failure::from(FailureKind::Json(verification.clone()))
            .during(FailureStage::Replacement(new_name.to_owned()))
    })?;
    let data_key = context
        .data_key
        .as_deref()
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .unwrap_or("key");
    let actual = verification_value(parsed, data_key).map_err(|failure| {
        Failure::from(failure).during(FailureStage::Replacement(new_name.to_owned()))
    })?;
    let actual_string = serde_json::from_slice::<String>(actual.get().as_bytes()).ok();
    if actual_string.as_deref() != Some(value.expose()) {
        return Err(FailureKind::ValueMismatch {
            expected: value.clone(),
            actual: actual.get().as_bytes().to_vec(),
        }
        .into());
    }
    if current_name != new_name {
        let _ = delete(client, current_name, context).await;
    }
    Ok(response)
}

fn verification_value<'a>(document: &'a RawValue, key: &str) -> Result<&'a RawValue, FailureKind> {
    let Some(outer) = raw_object_field(document, "data")? else {
        return Ok(RawValue::NULL);
    };
    let Some(inner) = raw_object_field(outer, "data")? else {
        return Ok(RawValue::NULL);
    };
    Ok(raw_object_field(inner, key)?.unwrap_or(RawValue::NULL))
}

fn raw_object_field<'a>(
    document: &'a RawValue,
    key: &str,
) -> Result<Option<&'a RawValue>, FailureKind> {
    let object: HashMap<String, &RawValue> = serde_json::from_slice(document.get().as_bytes())
        .map_err(|_| FailureKind::MissingGet(document.get().as_bytes().to_vec()))?;
    Ok(object.get(key).copied())
}
