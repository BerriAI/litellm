mod host;
mod mcp_gateway;
mod modify_params;
pub(crate) mod workload_identity;

use host::MessagesPythonHost;
use litellm_callbacks_legacy_python::LoggingOperation;
use modify_params::ModifyParams;
use pyo3::prelude::*;

use super::NativeCall;

fn run_messages(py: Python<'_>, call: NativeCall<'_>, asynchronous: bool) -> PyResult<Py<PyAny>> {
    let (arguments, hooks) =
        crate::routes::call_hooks(py, LoggingOperation::Messages, &call, asynchronous)?;
    let modify_params = ModifyParams::new(&call.args, &call.kwargs)?;
    crate::routes::run_public_call_after_policy(
        py,
        arguments,
        move |py, arguments, request| {
            let route = litellm_inference_messages::MessagesRoute::new(
                crate::http::provider_client(py, arguments, asynchronous)?
                    .map_err(crate::http::client_error)?,
                crate::http::resources().auth.clone(),
                crate::secrets::source(py)?,
            )
            .with_workload_identity(std::sync::Arc::new(
                workload_identity::PythonWorkloadIdentity::new(
                    litellm_host_python::PythonContext::capture(py)?,
                ),
            ));
            Ok(litellm_host::call::hosted_call(
                request,
                None,
                move |(call, selection): (_, crate::cache::Selection),
                      services,
                      interceptors,
                      observers| async move {
                    let route = route.with_mcp_gateway(std::sync::Arc::new(
                        mcp_gateway::PythonMcpGateway::new(services.clone()),
                    ));
                    let (cache, options) = selection.attach(services);
                    let route = match cache {
                        Some(cache) => route.with_cache(cache),
                        None => route,
                    };
                    route
                        .execute(
                            call,
                            &interceptors,
                            litellm_inference::CallOptions {
                                cache: Some(options.policy),
                                observers,
                            },
                        )
                        .await
                },
            ))
        },
        MessagesPythonHost::new(call.view()?, asynchronous),
        hooks,
        modify_params,
        asynchronous,
    )
}

/// Python's sync Messages handler raises for every native provider before it sends anything
/// or opens a logging object (`BaseLLMHTTPHandler.anthropic_messages_handler`), and Rust copies
/// that until both sides serve sync calls.
const SYNC_NOT_IMPLEMENTED: &str = "anthropic_messages_handler is not implemented for sync calls";

#[pyfunction]
#[pyo3(signature = (call))]
pub(crate) fn messages(py: Python<'_>, call: NativeCall<'_>) -> PyResult<Py<PyAny>> {
    let gateway = py
        .import(host::ROUTE_HOST_MODULE)?
        .getattr("gateway")?
        .call1((call.resolved()?, false))?;
    drop(call);
    if !gateway.is_none() {
        let ready = mcp_gateway::Ready::new(gateway.unbind());
        return Ok(Py::new(
            py,
            litellm_host_python::Execution::new(ready, crate::routes::execution_steps),
        )?
        .into_any());
    }
    Err(pyo3::exceptions::PyValueError::new_err(
        SYNC_NOT_IMPLEMENTED,
    ))
}

#[pyfunction]
pub(crate) fn amessages(py: Python<'_>, call: NativeCall<'_>) -> PyResult<Py<PyAny>> {
    run_messages(py, call, true)
}
