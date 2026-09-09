use litellm_auth::{AzureCredentialInputs, CallerTokenProvider};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict};

use crate::{PythonSecretProvider, SecretProviderContract};

const TOKEN_CONTRACT: SecretProviderContract = SecretProviderContract {
    value_name: "Azure AD token",
    failure_context: "Failed to get Azure AD token",
};

pub(crate) struct AzureBinding {
    inputs: AzureCredentialInputs,
    token_provider: Option<PythonSecretProvider>,
}

impl AzureBinding {
    pub(crate) fn from_arguments(py: Python<'_>, arguments: &Bound<'_, PyDict>) -> PyResult<Self> {
        let token_provider = arguments
            .get_item("azure_ad_token_provider")?
            .filter(|value| !value.is_none())
            .map(|value| PythonSecretProvider::new(value.unbind(), TOKEN_CONTRACT));
        let inputs = AzureCredentialInputs {
            token: optional_string(arguments, "azure_ad_token")?,
            has_token_provider: token_provider.is_some(),
            tenant_id: optional_string(arguments, "tenant_id")?,
            client_id: optional_string(arguments, "client_id")?,
            has_client_secret: has_nonempty_string(arguments, "client_secret")?,
            has_username: has_nonempty_string(arguments, "azure_username")?,
            has_password: has_nonempty_string(arguments, "azure_password")?,
            refresh: py
                .import("litellm")?
                .getattr("enable_azure_ad_token_refresh")?
                .is(PyBool::new(py, true)),
        };
        Ok(Self {
            inputs,
            token_provider,
        })
    }

    pub(crate) fn inputs(&self) -> &AzureCredentialInputs {
        &self.inputs
    }

    pub(crate) fn token_provider(&self) -> Option<&dyn CallerTokenProvider> {
        self.token_provider
            .as_ref()
            .map(|provider| provider as &dyn CallerTokenProvider)
    }

    pub(crate) fn take_error(&self) -> Option<PyErr> {
        self.token_provider
            .as_ref()
            .and_then(PythonSecretProvider::take_error)
    }
}

fn optional_string(arguments: &Bound<'_, PyDict>, name: &str) -> PyResult<Option<String>> {
    arguments
        .get_item(name)?
        .filter(|value| !value.is_none())
        .map(|value| value.extract::<String>())
        .transpose()
}

fn has_nonempty_string(arguments: &Bound<'_, PyDict>, name: &str) -> PyResult<bool> {
    Ok(optional_string(arguments, name)?.is_some_and(|value| !value.is_empty()))
}
