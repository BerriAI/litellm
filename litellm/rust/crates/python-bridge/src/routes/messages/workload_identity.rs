use std::fmt;

use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_auth::TokenProviderHandle;
use litellm_host_python::{PythonContext, attach_blocking, to_py};
use litellm_inference_messages::WorkloadIdentity;
use litellm_router_types::{ExtraParams, LitellmParams};
use litellm_secret_value::SecretValue;
use pyo3::{
    exceptions::PyBaseException,
    prelude::*,
    types::{PyBytes, PyDict},
};

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
                provider
                    .acquire_or(
                        py,
                        |token| non_string_token(py, token),
                        azure_ad_token_failure,
                    )
                    .map_err(|error| raised(py, error))
            })
            .await
            .map_err(|error| Python::attach(|py| raised(py, error)))
            .and_then(|acquired| acquired)
        }
    })
}

/// Python's own `get_azure_ad_token` over the call's arguments, for the Entra ID flows only
/// Python implements. No token comes back as an empty one, which sends the call unauthenticated.
pub(super) fn host_azure_ad_token(
    context: PythonContext,
    arguments: Py<PyDict>,
) -> TokenProviderHandle {
    let arguments = Arc::new(arguments);
    TokenProviderHandle::from_callback(move || {
        let context = context.clone();
        let arguments = Arc::clone(&arguments);
        async move {
            attach_blocking(context, move |py| {
                azure_ad_token(py, arguments.bind(py)).map_err(|error| raised(py, error))
            })
            .await
            .map_err(|error| Python::attach(|py| raised(py, error)))
            .and_then(|minted| minted)
        }
    })
}

fn azure_ad_token(
    py: Python<'_>,
    arguments: &Bound<'_, PyDict>,
) -> PyResult<litellm_auth::ResolvedCredential> {
    let token: String = py
        .import(MODULE)?
        .getattr("azure_ad_token")?
        .call1((arguments,))?
        .extract()?;
    Ok(litellm_auth::ResolvedCredential::AccessToken {
        token: SecretValue::new(token),
        expires_on: None,
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

/// What Python's `get_azure_ad_token` logs and raises for a provider that returned `token`, not a string.
fn non_string_token(py: Python<'_>, token: &Bound<'_, PyAny>) -> PyResult<PyErr> {
    let raised = py
        .import(MODULE)?
        .getattr("non_string_azure_ad_token")?
        .call1((token,))?;
    Ok(PyErr::from_value(raised))
}

/// What Python's `get_azure_ad_token` logs and raises from the caller's provider failing with `error`.
fn azure_ad_token_failure(py: Python<'_>, error: PyErr) -> PyErr {
    let failure = py.import(MODULE).and_then(|module| {
        module
            .getattr("azure_ad_token_failure")?
            .call1((error.value(py),))
    });
    match failure {
        Ok(failure) => {
            let failure = PyErr::from_value(failure);
            failure.set_context(py, Some(error));
            failure
        }
        Err(mapping) => {
            mapping.set_context(py, Some(error));
            mapping
        }
    }
}

/// What Python's `get_azure_ad_token` raises for a native client secret flow that failed with
/// `error`, or `None` when the failure did not come from the token endpoint.
pub(crate) fn entra_token_error(
    py: Python<'_>,
    error: &(dyn std::error::Error + 'static),
) -> Option<PyErr> {
    let failure = litellm_error::find::<litellm_auth_azure::EntraTokenFailure>(error)?;
    let raised = py
        .import(MODULE)
        .and_then(|module| match failure.response() {
            Some(response) => module.getattr("entra_token_response_failure")?.call1((
                response.status,
                response.headers.clone(),
                PyBytes::new(py, &response.body),
            )),
            None => module
                .getattr("entra_token_failure")?
                .call1((std::error::Error::source(failure)
                    .map(|cause| litellm_error::Report::new(cause).to_string())
                    .unwrap_or_else(|| litellm_error::Report::new(failure).to_string()),)),
        });
    match raised {
        Ok(raised) if raised.is_none() => None,
        Ok(raised) => Some(PyErr::from_value(raised)),
        Err(mapping) => Some(mapping),
    }
}

pub(crate) fn python_error(
    py: Python<'_>,
    error: &(dyn std::error::Error + 'static),
) -> Option<PyErr> {
    litellm_error::find::<PythonWorkloadIdentityError>(error)
        .map(|error| PyErr::from_value(error.0.clone_ref(py).into_bound(py).into_any()))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use litellm_auth_azure::{EntraTokenFailure, TokenResponse};
    use pyo3::types::PyList;

    use super::*;

    static HOST: Mutex<()> = Mutex::new(());

    const FAKE_HOST: &std::ffi::CStr = c"
import sys, types
previous = sys.modules.get('litellm.rust_bridge.host.routes.messages')
calls = []
def entra_token_response_failure(status, headers, body):
    calls.append(('response', status, headers, body))
    return None if status == 200 else RuntimeError('mapped response')
def entra_token_failure(reason):
    calls.append(('reason', reason))
    return RuntimeError('mapped reason')
def azure_ad_token_failure(error):
    calls.append(('provider', error))
    return RuntimeError('mapped provider')
def non_string_azure_ad_token(token):
    calls.append(('non_string', token))
    return TypeError('mapped token')
host = types.ModuleType('litellm.rust_bridge.host.routes.messages')
for function in (entra_token_response_failure, entra_token_failure, azure_ad_token_failure, non_string_azure_ad_token):
    setattr(host, function.__name__, function)
for name in ('litellm', 'litellm.rust_bridge', 'litellm.rust_bridge.host', 'litellm.rust_bridge.host.routes'):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules['litellm.rust_bridge.host.routes.messages'] = host
";

    const RESTORE_HOST: &std::ffi::CStr = c"
if previous is None:
    sys.modules.pop('litellm.rust_bridge.host.routes.messages', None)
else:
    sys.modules['litellm.rust_bridge.host.routes.messages'] = previous
";

    fn with_fake_host(body: impl FnOnce(Python<'_>, &Bound<'_, PyDict>)) {
        let _guard = HOST
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(FAKE_HOST, Some(&locals), Some(&locals)).unwrap();
            body(py, &locals);
            py.run(RESTORE_HOST, Some(&locals), Some(&locals)).unwrap();
        });
    }

    fn calls(locals: &Bound<'_, PyDict>) -> String {
        locals
            .get_item("calls")
            .unwrap()
            .unwrap()
            .cast_into::<PyList>()
            .unwrap()
            .repr()
            .unwrap()
            .to_string()
    }

    fn token_response(status: u16, headers: &[(&str, &str)]) -> TokenResponse {
        TokenResponse {
            status,
            headers: headers
                .iter()
                .map(|(name, value)| (name.to_string(), value.to_string()))
                .collect(),
            body: b"{\"error\":\"invalid_client\"}".to_vec(),
        }
    }

    fn acquisition_failure(failure: EntraTokenFailure) -> litellm_auth::Error {
        litellm_auth::Error::CredentialAcquisition(litellm_auth::ErrorDetail::failed(
            "Azure token request",
            failure,
        ))
    }

    #[rstest::rstest]
    #[case::endpoint_answer(
        EntraTokenFailure::Response(token_response(401, &[("content-type", "application/json")])),
        Some("mapped response"),
        "[('response', 401, [('content-type', 'application/json')], b'{\"error\":\"invalid_client\"}')]"
    )]
    #[case::malformed_retry_after_replays_its_answer(
        EntraTokenFailure::InvalidRetryAfter(token_response(500, &[("retry-after", "soon")])),
        Some("mapped response"),
        "[('response', 500, [('retry-after', 'soon')], b'{\"error\":\"invalid_client\"}')]"
    )]
    #[case::answer_python_does_not_reject(
        EntraTokenFailure::Response(token_response(200, &[])),
        None,
        "[('response', 200, [], b'{\"error\":\"invalid_client\"}')]"
    )]
    #[case::transport_failure_gives_its_cause(
        EntraTokenFailure::Transport(azure_core::Error::with_message(
            azure_core::error::ErrorKind::Connection,
            "connection refused",
        )),
        Some("mapped reason"),
        "[('reason', 'connection refused')]"
    )]
    fn a_native_entra_failure_raises_what_the_host_builds(
        #[case] failure: EntraTokenFailure,
        #[case] raised: Option<&str>,
        #[case] asked: &str,
    ) {
        let error = acquisition_failure(failure);
        with_fake_host(|py, locals| {
            let mapped = entra_token_error(py, &error);

            assert_eq!(
                mapped.map(|mapped| mapped.value(py).str().unwrap().to_string()),
                raised.map(str::to_string)
            );
            assert_eq!(calls(locals), asked);
        });
    }

    #[rstest::rstest]
    fn a_failure_from_elsewhere_is_not_an_entra_token_failure() {
        let error = litellm_auth::Error::CredentialAcquisition(litellm_auth::ErrorDetail::failed(
            "host credential lookup",
            std::io::Error::from(std::io::ErrorKind::PermissionDenied),
        ));
        with_fake_host(|py, locals| {
            assert!(entra_token_error(py, &error).is_none());
            assert_eq!(calls(locals), "[]");
        });
    }

    #[rstest::rstest]
    #[case::raising_provider(
        c"def provider():\n    raise ValueError('token service unavailable')",
        "RuntimeError",
        "mapped provider",
        "[('provider', ValueError('token service unavailable'))]"
    )]
    #[case::non_string_token(
        c"def provider():\n    return 1",
        "TypeError",
        "mapped token",
        "[('non_string', 1)]"
    )]
    #[case::type_error_is_raised_as_is(
        c"def provider():\n    raise TypeError('bad call')",
        "TypeError",
        "bad call",
        "[]"
    )]
    fn a_caller_provider_failure_raises_what_get_azure_ad_token_raises(
        #[case] source: &std::ffi::CStr,
        #[case] class: &str,
        #[case] message: &str,
        #[case] asked: &str,
    ) {
        with_fake_host(|py, locals| {
            py.run(source, Some(locals), Some(locals)).unwrap();
            py.run(
                c"kwargs = {'azure_ad_token_provider': provider}",
                Some(locals),
                Some(locals),
            )
            .unwrap();
            let provider = crate::credentials::azure_ad_token_provider(
                &locals
                    .get_item("kwargs")
                    .unwrap()
                    .unwrap()
                    .cast_into::<PyDict>()
                    .unwrap(),
            )
            .unwrap()
            .unwrap();

            let error = provider
                .acquire_or(
                    py,
                    |token| non_string_token(py, token),
                    azure_ad_token_failure,
                )
                .unwrap_err();

            assert_eq!(error.get_type(py).name().unwrap().to_string(), class);
            assert_eq!(error.value(py).str().unwrap().to_string(), message);
            assert_eq!(calls(locals), asked);
        });
    }
}
