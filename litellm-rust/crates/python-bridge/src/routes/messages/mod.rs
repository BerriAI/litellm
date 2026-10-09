mod host;

use crate::errors::RustBridgeDeclined;
use host::{MessagesPythonHost, ROUTE_HOST_MODULE};
use litellm_host::call::Operation;
use pyo3::prelude::*;

use super::NativeCall;

fn run_messages(py: Python<'_>, call: NativeCall<'_>, asynchronous: bool) -> PyResult<Py<PyAny>> {
    if let Some(reason) = py
        .import(ROUTE_HOST_MODULE)?
        .getattr("decline_reason")?
        .call1((call.resolved()?,))?
        .extract::<Option<String>>()?
    {
        return Err(RustBridgeDeclined::new_err(reason));
    }
    let (arguments, hooks) =
        crate::routes::call_hooks(py, Operation::Messages, &call, asynchronous)?;
    crate::routes::run_public_call(
        py,
        arguments,
        move |py, arguments, request| {
            let route = litellm_inference_messages::MessagesRoute::new(
                crate::http::provider_client(py, arguments, asynchronous)?
                    .map_err(crate::http::client_error)?,
                crate::http::resources().auth.clone(),
                crate::secrets::source(py)?,
            );
            Ok(litellm_host::call::hosted_call(
                request,
                move |(call, selection): (_, crate::cache::Selection), services, interceptors| async move {
                    let (cache, options) = selection.attach(services);
                    let route = match cache {
                        Some(cache) => route.with_cache(cache),
                        None => route,
                    };
                    route
                        .execute(call, &interceptors, Some(options.policy))
                        .await
                },
            ))
        },
        MessagesPythonHost::new(call.resolved()?.unbind(), asynchronous),
        hooks,
        asynchronous,
    )
}

#[pyfunction]
pub(crate) fn messages(py: Python<'_>, call: NativeCall<'_>) -> PyResult<Py<PyAny>> {
    run_messages(py, call, false)
}

#[pyfunction]
pub(crate) fn amessages(py: Python<'_>, call: NativeCall<'_>) -> PyResult<Py<PyAny>> {
    run_messages(py, call, true)
}
