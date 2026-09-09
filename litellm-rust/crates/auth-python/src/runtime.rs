use litellm_auth::{AuthServices, CallerCredential, CallerTokenProvider, CredentialInputs};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::azure::AzureBinding;

pub struct PythonAuth {
    inputs: CredentialInputs,
    azure: AzureBinding,
}

impl PythonAuth {
    pub fn from_arguments(py: Python<'_>, arguments: &Bound<'_, PyDict>) -> PyResult<Self> {
        let azure = AzureBinding::from_arguments(py, arguments)?;
        let inputs = CredentialInputs {
            azure: azure.inputs().clone(),
        };
        Ok(Self { inputs, azure })
    }

    pub fn inputs(&self) -> &CredentialInputs {
        &self.inputs
    }

    pub fn take_error(&self) -> Option<PyErr> {
        self.azure.take_error()
    }
}

impl AuthServices for PythonAuth {
    fn token_provider(&self, credential: CallerCredential) -> Option<&dyn CallerTokenProvider> {
        match credential {
            CallerCredential::AzureAdToken => self.azure.token_provider(),
        }
    }
}
