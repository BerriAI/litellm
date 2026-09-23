use std::time::Duration;

use super::operations::PythonReadRequest;
use litellm_secrets::{KeyManagementSystem, SecretValue};
use litellm_secrets_types::{
    AwsOperationContext, CyberarkOperationContext, GoogleOperationContext,
    HashicorpOperationContext, SecretOperationContext,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

pub(super) fn read_request(
    system: KeyManagementSystem,
    secret_name: String,
    optional_params: Option<&Bound<'_, PyAny>>,
    timeout: Option<&Bound<'_, PyAny>>,
    primary_secret_name: Option<String>,
    synchronous: bool,
) -> PyResult<PythonReadRequest> {
    let context = match system {
        KeyManagementSystem::AwsSecretManager => {
            let ignored = primary_secret_name
                .as_ref()
                .is_some_and(|value| !value.is_empty())
                || (synchronous
                    && litellm_secrets::aws::secret_manager::is_bootstrap_key(&secret_name));
            SecretOperationContext::Aws(if ignored {
                AwsOperationContext::default()
            } else {
                aws_context(optional_params, timeout)?
            })
        }
        KeyManagementSystem::HashicorpVault => {
            SecretOperationContext::Hashicorp(vault_context(optional_params)?)
        }
        KeyManagementSystem::Cyberark => {
            SecretOperationContext::Cyberark(CyberarkOperationContext::default())
        }
        KeyManagementSystem::GoogleSecretManager => {
            SecretOperationContext::Google(GoogleOperationContext::default())
        }
        _ => {
            return Err(PyValueError::new_err(
                "secret manager does not support provider reads",
            ));
        }
    };
    Ok(PythonReadRequest {
        secret_name,
        primary_secret_name,
        context,
        synchronous,
    })
}

fn string_field(params: Option<&Bound<'_, PyDict>>, name: &str) -> PyResult<Option<String>> {
    let value = params
        .map(|params| params.get_item(name))
        .transpose()?
        .flatten();
    match value {
        Some(value) if value.is_truthy()? => value.extract().map(Some),
        _ => Ok(None),
    }
}

fn aws_context(
    params: Option<&Bound<'_, PyAny>>,
    timeout: Option<&Bound<'_, PyAny>>,
) -> PyResult<AwsOperationContext> {
    let params = params
        .filter(|value| !value.is_none())
        .map(|value| value.cast::<PyDict>())
        .transpose()?;
    Ok(AwsOperationContext {
        access_key_id: string_field(params, "aws_access_key_id")?.map(SecretValue::new),
        secret_access_key: string_field(params, "aws_secret_access_key")?.map(SecretValue::new),
        session_token: string_field(params, "aws_session_token")?.map(SecretValue::new),
        region_name: string_field(params, "aws_region_name")?,
        role_name: string_field(params, "aws_role_name")?,
        session_name: string_field(params, "aws_session_name")?,
        external_id: string_field(params, "aws_external_id")?.map(SecretValue::new),
        profile_name: string_field(params, "aws_profile_name")?,
        web_identity_token: string_field(params, "aws_web_identity_token")?.map(SecretValue::new),
        sts_endpoint: string_field(params, "aws_sts_endpoint")?,
        bedrock_runtime_endpoint: string_field(params, "aws_bedrock_runtime_endpoint")?,
        timeout: read_timeout(timeout)?,
    })
}

fn read_timeout(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Duration>> {
    let Some(value) = value.filter(|value| !value.is_none()) else {
        return Ok(None);
    };
    let seconds = match value.extract::<f64>() {
        Ok(value) => Some(value),
        Err(_) => value.getattr("read")?.extract::<Option<f64>>()?,
    };
    seconds
        .map(|value| {
            Duration::try_from_secs_f64(value).map_err(|_| PyValueError::new_err("invalid timeout"))
        })
        .transpose()
}

fn vault_context(params: Option<&Bound<'_, PyAny>>) -> PyResult<HashicorpOperationContext> {
    let params = params.and_then(|value| value.cast::<PyDict>().ok());
    let nested = params
        .map(|params| params.get_item("secret_manager_settings"))
        .transpose()?
        .flatten();
    let source = nested
        .as_ref()
        .and_then(|value| value.cast::<PyDict>().ok())
        .or(params);
    Ok(HashicorpOperationContext {
        namespace: vault_field(source, "namespace")?,
        mount: vault_field(source, "mount")?,
        path_prefix: vault_field(source, "path_prefix")?,
        data_key: vault_field(source, "data")?,
        timeout: None,
    })
}

fn vault_field(params: Option<&Bound<'_, PyDict>>, name: &str) -> PyResult<Option<String>> {
    let value = params
        .map(|params| params.get_item(name))
        .transpose()?
        .flatten();
    match value {
        Some(value) if value.is_none() => Ok(None),
        Some(value) => value.str()?.extract().map(Some),
        None => Ok(None),
    }
}

pub(super) fn mutation_context(
    system: KeyManagementSystem,
    optional_params: Option<&Bound<'_, PyAny>>,
    timeout: Option<&Bound<'_, PyAny>>,
) -> PyResult<SecretOperationContext> {
    if system == KeyManagementSystem::HashicorpVault {
        return Ok(SecretOperationContext::Hashicorp(
            HashicorpOperationContext {
                timeout: read_timeout(timeout)?,
                ..vault_context(optional_params)?
            },
        ));
    }
    Ok(SecretOperationContext::Default)
}
