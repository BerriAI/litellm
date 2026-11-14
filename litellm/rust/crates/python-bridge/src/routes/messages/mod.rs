mod agentic;
mod answer;
mod host;
mod modify_params;
mod pre_request;
mod safeguards;
mod transport;
pub(crate) mod workload_identity;

use host::MessagesPythonHost;
use litellm_callbacks_legacy_python::LoggingOperation;
use modify_params::ModifyParams;
use pyo3::prelude::*;

use super::NativeCall;

fn run_messages(py: Python<'_>, call: NativeCall<'_>, asynchronous: bool) -> PyResult<Py<PyAny>> {
    let (arguments, hooks) =
        crate::routes::call_hooks(py, LoggingOperation::Messages, &call, asynchronous)?;
    let hooked_request = agentic::HookedRequest::default();
    let hooks = litellm_host_python::HookChain::new()
        .with(agentic::AgenticLoop::new(
            asynchronous,
            std::sync::Arc::clone(&hooked_request),
        ))
        .with(hooks);
    let modify_params = ModifyParams::new(&call.args, &call.kwargs)?;
    let classifies = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let classified = std::sync::Arc::clone(&classifies);
    crate::routes::run_public_call_after_policy(
        py,
        arguments,
        move |py, arguments, request| {
            let hooked = pre_request::runs_pre_request_hooks(py)?;
            let route = litellm_inference_messages::MessagesRoute::new(
                crate::http::messages_client(py, arguments, asynchronous)?
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
                move |(call, selection, hosted): (_, crate::cache::Selection, bool),
                      services,
                      interceptors,
                      observers| async move {
                    let route = if hosted {
                        route.with_transport(std::sync::Arc::new(transport::PythonTransport::new(
                            services.clone(),
                        )))
                    } else {
                        route
                    };
                    let route = match hooked {
                        true => route.with_pre_request_hooks(std::sync::Arc::new(
                            pre_request::HostPreRequestHooks::new(services.clone()),
                        )),
                        false => route,
                    };
                    let route = match classified.load(std::sync::atomic::Ordering::Acquire) {
                        true => route.with_safeguards(std::sync::Arc::new(
                            safeguards::PythonSafeguards::new(services.clone()),
                        )),
                        false => route,
                    };
                    let route = route.with_host_answer(std::sync::Arc::new(
                        answer::PythonAnswer::new(services.clone()),
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
        MessagesPythonHost::new(call.view()?, asynchronous, hooked_request, classifies),
        hooks,
        modify_params,
        asynchronous,
    )
}

/// Python's sync Messages handler answers a mock without logging, hands proxy MCP tools to the
/// gateway, and raises for every other native provider call before it sends anything or opens a
/// logging object (`BaseLLMHTTPHandler.anthropic_messages_handler`), and Rust copies that until
/// both sides serve sync calls.
const SYNC_NOT_IMPLEMENTED: &str = "anthropic_messages_handler is not implemented for sync calls";

#[pyfunction]
#[pyo3(signature = (call))]
pub(crate) fn messages(py: Python<'_>, call: NativeCall<'_>) -> PyResult<Py<PyAny>> {
    let answer = py
        .import(host::ROUTE_HOST_MODULE)?
        .getattr("sync_answer")?
        .call1((call.resolved()?,))?;
    if answer.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            SYNC_NOT_IMPLEMENTED,
        ));
    }
    Ok(answer.unbind())
}

#[pyfunction]
pub(crate) fn amessages(py: Python<'_>, call: NativeCall<'_>) -> PyResult<Py<PyAny>> {
    run_messages(py, call, true)
}
