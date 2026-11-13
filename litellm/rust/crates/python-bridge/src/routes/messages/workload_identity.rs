use std::fmt;

use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_auth::TokenProviderHandle;
use litellm_host_python::{PythonContext, attach_blocking, to_py};
use litellm_inference_messages::WorkloadIdentity;
use litellm_router_types::{ExtraParams, LitellmParams};
use litellm_secret_value::SecretValue;
use pyo3::{exceptions::PyBaseException, prelude::*};

use crate::credentials::CallerTokenProvider;

const MODULE: &str = "litellm.rust_bridge.host.routes.messages";

pub(super) struct PythonWorkloadIdentity {
    context: PythonContext,
}

impl PythonWorkloadIdentity {
    pub(super) fn new(context: PythonContext) -> Self {
        Self { context }
    }
}

impl WorkloadIdentity for PythonWorkloadIdentity {
    fn token<'a>(
        &'a self,
        litellm_params: &'a LitellmParams,
        api_base: Option<&'a str>,
    ) -> BoxFuture<'a, Result<Option<SecretValue>, litellm_auth::Error>> {
        let params = litellm_params.extra.clone();
        let api_base = api_base.map(str::to_owned);
        let context = self.context.clone();
        Box::pin(async move {
            attach_blocking(context, move |py| {
                mint(py, &params, api_base).map_err(|error| raised(py, error))
            })
            .await
            .map_err(|error| Python::attach(|py| raised(py, error)))
            .and_then(|minted| minted)
            .map(|token| token.map(SecretValue::new))
        })
    }
}

/// The caller's `azure_ad_token_provider`, called on the blocking pool when the Entra ID
/// scheme asks for a token, its exception reaching the caller as Python raised it.
pub(super) fn azure_ad_token_provider(
    context: PythonContext,
    provider: CallerTokenProvider,
) -> TokenProviderHandle {
    let provider = Arc::new(provider);
    TokenProviderHandle::from_callback(move || {
        let context = context.clone();
        let provider = Arc::clone(&provider);
        async move {
            attach_blocking(context, move |py| {
                provider.acquire(py).map_err(|error| raised(py, error))
            })
            .await
            .map_err(|error| Python::attach(|py| raised(py, error)))
            .and_then(|acquired| acquired)
        }
    })
}

fn mint(
    py: Python<'_>,
    params: &ExtraParams,
    api_base: Option<String>,
) -> PyResult<Option<String>> {
    py.import(MODULE)?
        .getattr("workload_identity_token")?
        .call1((to_py(py, &**params)?, api_base))?
        .extract()
}

#[derive(thiserror::Error)]
#[error("Python workload identity federation failed")]
struct PythonWorkloadIdentityError(Py<PyBaseException>);

impl fmt::Debug for PythonWorkloadIdentityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PythonWorkloadIdentityError")
    }
}

fn raised(py: Python<'_>, error: PyErr) -> litellm_auth::Error {
    litellm_auth::Error::provider(PythonWorkloadIdentityError(error.into_value(py)))
}

pub(crate) fn python_error(
    py: Python<'_>,
    error: &(dyn std::error::Error + 'static),
) -> Option<PyErr> {
    litellm_error::find::<PythonWorkloadIdentityError>(error)
        .map(|error| PyErr::from_value(error.0.clone_ref(py).into_bound(py).into_any()))
}
