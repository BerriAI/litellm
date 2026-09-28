mod host;

use host::MessagesPythonHost;
use litellm_types::Operation;
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};

fn run_messages(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
    asynchronous: bool,
) -> PyResult<Py<PyAny>> {
    let cache_call_type = "anthropic_messages";
    crate::cache::v2::admit(py, &kwargs, cache_call_type)?;
    let (arguments, hooks) = crate::routes::call_hooks(
        py,
        Operation::Messages,
        &request,
        &args,
        &kwargs,
        asynchronous,
    )?;
    crate::routes::run_public_call(
        py,
        arguments,
        move |py, arguments, request| {
            let builder = litellm_core::messages::MessagesRoute::builder()
                .with_http(
                    crate::http::provider_client(py, arguments, asynchronous)?
                        .map_err(crate::http::client_error)?,
                )
                .with_auth(crate::http::resources().auth.clone())
                .with_secrets(crate::secrets::source(py)?);
            let (cache, cache_options) =
                crate::cache::v2::configured(py, arguments, cache_call_type)?;
            let builder = match cache {
                Some(cache) => builder.with_cache(litellm_cache_response::ScopedCache::new(
                    cache,
                    litellm_cache_response::CacheScope::Shared,
                )),
                None => builder,
            };
            Ok(builder.build().machine(request, cache_options))
        },
        MessagesPythonHost::new(request.unbind()),
        hooks,
        asynchronous,
    )
}

#[pyfunction]
pub(crate) fn messages(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    run_messages(py, request, args, kwargs, false)
}

#[pyfunction]
pub(crate) fn amessages(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    run_messages(py, request, args, kwargs, true)
}
