use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyCFunction;

macro_rules! unimplemented_lifecycle_route {
    ($route:ident, $sync:ident, $asynchronous:ident) => {
        fn decline() -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
            use litellm_core::call_lifecycle::admission::{
                UnimplementedRoute, admit_unimplemented,
            };
            match admit_unimplemented(UnimplementedRoute::$route) {
                Ok(never) => match never {},
                Err(route) => Err($crate::errors::RustBridgeDeclined::new_err(format!(
                    "{route} native lifecycle is not implemented"
                ))),
            }
        }

        #[pyo3::pyfunction]
        fn $sync(
            request: pyo3::Bound<'_, pyo3::PyAny>,
            args: pyo3::Bound<'_, pyo3::types::PyTuple>,
            kwargs: pyo3::Bound<'_, pyo3::types::PyDict>,
            host: pyo3::Bound<'_, pyo3::PyAny>,
        ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
            let _ = (request, args, kwargs, host);
            decline()
        }

        #[pyo3::pyfunction]
        fn $asynchronous(
            request: pyo3::Bound<'_, pyo3::PyAny>,
            args: pyo3::Bound<'_, pyo3::types::PyTuple>,
            kwargs: pyo3::Bound<'_, pyo3::types::PyDict>,
            host: pyo3::Bound<'_, pyo3::PyAny>,
        ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
            let _ = (request, args, kwargs, host);
            decline()
        }

        pub(super) fn register(
            module: &pyo3::Bound<'_, pyo3::types::PyModule>,
        ) -> pyo3::PyResult<()> {
            $crate::routes::definition::add_function(
                module,
                pyo3::wrap_pyfunction!($sync, module)?,
            )?;
            $crate::routes::definition::add_function(
                module,
                pyo3::wrap_pyfunction!($asynchronous, module)?,
            )
        }
    };
}

#[cfg(test)]
macro_rules! bridge_route {
    (
        sync = $sync_name:ident,
        asynchronous = $async_name:ident,
        inputs = $inputs:ident,
        required = { $($(#[$required_attr:meta])* $required_name:ident: $required_type:ty),+ $(,)? },
        optional = { $($(#[$optional_attr:meta])* $optional_name:ident: $optional_type:ty),* $(,)? },
        prepare = $prepare:path,
        errors = $map_error:path
        $(, extra = [$($extra:ident),* $(,)?])?
        $(,)?
    ) => {
        struct $inputs {
            $($required_name: $required_type,)*
            $($optional_name: $optional_type),*
        }

        #[pyfunction]
        #[pyo3(signature = ($($required_name),*, $($optional_name=None),*))]
        #[allow(clippy::too_many_arguments)]
        fn $sync_name(
            py: pyo3::Python<'_>,
            $($(#[$required_attr])* $required_name: $required_type,)*
            $($(#[$optional_attr])* $optional_name: $optional_type,)*
        ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
            let future = $prepare($inputs {
                $($required_name,)*
                $($optional_name),*
            })?;
            $crate::execution::run_sync(py, future, $map_error)
        }

        #[pyfunction]
        #[pyo3(signature = ($($required_name),*, $($optional_name=None),*))]
        #[allow(clippy::too_many_arguments)]
        fn $async_name(
            py: pyo3::Python<'_>,
            $($(#[$required_attr])* $required_name: $required_type,)*
            $($(#[$optional_attr])* $optional_name: $optional_type,)*
        ) -> pyo3::PyResult<pyo3::Bound<'_, pyo3::PyAny>> {
            let future = $prepare($inputs {
                $($required_name,)*
                $($optional_name),*
            })?;
            $crate::execution::run_async(py, future, $map_error)
        }

        pub(super) fn register(
            module: &pyo3::Bound<'_, pyo3::types::PyModule>,
        ) -> pyo3::PyResult<()> {
            $($($crate::routes::definition::add_function(module, pyo3::wrap_pyfunction!($extra, module)?)?;)*)?
            $crate::routes::definition::add_function(module, pyo3::wrap_pyfunction!($sync_name, module)?)?;
            $crate::routes::definition::add_function(module, pyo3::wrap_pyfunction!($async_name, module)?)?;
            Ok(())
        }

        #[cfg(feature = "trace-parity")]
        mod trace {
            use pyo3::prelude::*;
            use super::{$inputs, $map_error, $prepare};

            #[pyfunction]
            #[pyo3(signature = ($($required_name),*, $($optional_name=None),*))]
            #[allow(clippy::too_many_arguments)]
            fn $sync_name(
                py: pyo3::Python<'_>,
                $($(#[$required_attr])* $required_name: $required_type,)*
                $($(#[$optional_attr])* $optional_name: $optional_type,)*
            ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                let future = $prepare($inputs {
                    $($required_name,)*
                    $($optional_name),*
                })?;
                $crate::execution::run_sync(
                    py,
                    $crate::function_trace::capture(future),
                    $map_error,
                )
            }

            #[pyfunction]
            #[pyo3(signature = ($($required_name),*, $($optional_name=None),*))]
            #[allow(clippy::too_many_arguments)]
            fn $async_name(
                py: pyo3::Python<'_>,
                $($(#[$required_attr])* $required_name: $required_type,)*
                $($(#[$optional_attr])* $optional_name: $optional_type,)*
            ) -> pyo3::PyResult<pyo3::Bound<'_, pyo3::PyAny>> {
                let future = $prepare($inputs {
                    $($required_name,)*
                    $($optional_name),*
                })?;
                $crate::execution::run_async(
                    py,
                    $crate::function_trace::capture(future),
                    $map_error,
                )
            }

            pub(super) fn register(
                module: &pyo3::Bound<'_, pyo3::types::PyModule>,
            ) -> pyo3::PyResult<()> {
                $crate::routes::definition::add_function(
                    module,
                    pyo3::wrap_pyfunction!($sync_name, module)?,
                )?;
                $crate::routes::definition::add_function(
                    module,
                    pyo3::wrap_pyfunction!($async_name, module)?,
                )?;
                Ok(())
            }
        }

        #[cfg(feature = "trace-parity")]
        pub(super) fn register_trace(
            module: &pyo3::Bound<'_, pyo3::types::PyModule>,
        ) -> pyo3::PyResult<()> {
            trace::register(module)
        }
    };
}

pub(super) fn add_function(
    module: &Bound<'_, PyModule>,
    function: Bound<'_, PyCFunction>,
) -> PyResult<()> {
    let name: String = function.getattr("__name__")?.extract()?;
    if module.hasattr(&name)? {
        return Err(PyRuntimeError::new_err(format!(
            "duplicate native route: {name}"
        )));
    }
    module.add_function(function)
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::sync::atomic::{AtomicBool, Ordering};

    use litellm_core::error::Error;
    use pyo3::exceptions::PyLookupError;
    use pyo3::types::{PyDict, PyList};

    use super::*;

    mod synthetic {
        use std::future::{Future, pending};

        use super::*;

        static FUTURE_DROPPED: AtomicBool = AtomicBool::new(false);

        struct DropGuard;

        impl Drop for DropGuard {
            fn drop(&mut self) {
                FUTURE_DROPPED.store(true, Ordering::SeqCst);
            }
        }

        #[pyfunction]
        fn future_dropped() -> bool {
            FUTURE_DROPPED.load(Ordering::SeqCst)
        }

        bridge_route! {
            sync = echo,
            asynchronous = aecho,
            inputs = EchoInputs,
            required = { value: String },
            optional = {},
            prepare = prepare_echo,
            errors = map_error,
            extra = [future_dropped],
        }

        fn prepare_echo(
            inputs: EchoInputs,
        ) -> PyResult<impl Future<Output = Result<String, Error>> + Send + 'static> {
            FUTURE_DROPPED.store(false, Ordering::SeqCst);
            let drop_guard = (inputs.value == "pending").then_some(DropGuard);
            Ok(execute_echo(inputs, drop_guard))
        }

        #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
        async fn execute_echo(
            inputs: EchoInputs,
            drop_guard: Option<DropGuard>,
        ) -> Result<String, Error> {
            let _drop_guard = drop_guard;
            tokio::task::yield_now().await;
            match inputs.value.as_str() {
                "error" => Err(Error::InvalidRequest("synthetic error".to_string())),
                "map_panic" => Err(Error::InvalidRequest("panic in mapper".to_string())),
                "panic" => panic!("synthetic panic"),
                "pending" => {
                    pending::<()>().await;
                    unreachable!()
                }
                _ => Ok(inputs.value),
            }
        }

        fn map_error(error: Error) -> PyErr {
            if matches!(&error, Error::InvalidRequest(message) if message == "panic in mapper") {
                panic!("synthetic mapper panic")
            }
            PyLookupError::new_err(error.to_string())
        }
    }

    #[test]
    fn sync_and_async_route_signatures_match_the_python_contract() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "routes").expect("module should be created");
            crate::routes::register(&module).expect("routes should register");
            let routes = [
                ("ocr", "aocr", "(request, args, kwargs, host)"),
                (
                    "transcription",
                    "atranscription",
                    "(request, args, kwargs, host)",
                ),
                ("messages", "amessages", "(request, args, kwargs, host)"),
                (
                    "chat_completions",
                    "achat_completions",
                    "(request, args, kwargs, host)",
                ),
            ];

            for (sync_name, async_name, expected) in routes {
                let sync_signature: String = module
                    .getattr(sync_name)
                    .and_then(|function| function.getattr("__text_signature__"))
                    .and_then(|signature| signature.extract())
                    .expect("sync signature should be available");
                let async_signature: String = module
                    .getattr(async_name)
                    .and_then(|function| function.getattr("__text_signature__"))
                    .and_then(|signature| signature.extract())
                    .expect("async signature should be available");

                assert_eq!(sync_signature, expected);
                assert_eq!(async_signature, expected);
            }
        });
    }

    #[test]
    fn sync_and_async_routes_apply_the_same_input_validation() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "routes").expect("module should be created");
            crate::routes::register(&module).expect("routes should register");

            let invalid_messages = PyDict::new(py);
            let request = PyDict::new(py);
            request.set_item("model", "anthropic/model").unwrap();
            request.set_item("messages", &invalid_messages).unwrap();
            let sync_chat_error = module
                .getattr("chat_completions")
                .and_then(|function| function.call1((&request, (), PyDict::new(py), py.None())))
                .expect_err("sync chat should reject a non-list messages value");
            let async_chat_error = module
                .getattr("achat_completions")
                .and_then(|function| function.call1((&request, (), PyDict::new(py), py.None())))
                .expect_err("async chat should reject a non-list messages value");

            assert!(sync_chat_error.is_instance_of::<crate::errors::RustBridgeDeclined>(py));
            assert_eq!(async_chat_error.to_string(), sync_chat_error.to_string());

            let invalid_body = PyList::empty(py);
            let request = PyDict::new(py);
            request.set_item("model", "anthropic/model").unwrap();
            request.set_item("body", &invalid_body).unwrap();
            let sync_messages_error = module
                .getattr("messages")
                .and_then(|function| function.call1((&request, (), PyDict::new(py), py.None())))
                .expect_err("sync Messages should reject a non-dict body");
            let async_messages_error = module
                .getattr("amessages")
                .and_then(|function| function.call1((&request, (), PyDict::new(py), py.None())))
                .expect_err("async Messages should reject a non-dict body");

            assert!(sync_messages_error.is_instance_of::<crate::errors::RustBridgeDeclined>(py));
            assert_eq!(
                async_messages_error.to_string(),
                sync_messages_error.to_string()
            );

            let invalid_audio = PyList::empty(py);
            let request = PyDict::new(py);
            request.set_item("model", "bedrock/model").unwrap();
            request.set_item("audio", &invalid_audio).unwrap();
            let sync_transcription_error = module
                .getattr("transcription")
                .and_then(|function| function.call1((&request, (), PyDict::new(py), py.None())))
                .expect_err("sync transcription should reject a non-dict audio value");
            let async_transcription_error = module
                .getattr("atranscription")
                .and_then(|function| function.call1((&request, (), PyDict::new(py), py.None())))
                .expect_err("async transcription should reject a non-dict audio value");

            assert!(
                sync_transcription_error.is_instance_of::<crate::errors::RustBridgeDeclined>(py)
            );
            assert_eq!(
                async_transcription_error.to_string(),
                sync_transcription_error.to_string()
            );
        });
    }

    #[test]
    fn missing_and_explicit_none_optional_params_share_the_next_error() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "routes").expect("module should be created");
            crate::routes::register(&module).expect("routes should register");
            let messages = PyList::new(py, [PyDict::new(py)]).unwrap();
            let headers = PyList::empty(py);
            let omitted = PyDict::new(py);
            omitted.set_item("model", "anthropic/model").unwrap();
            omitted.set_item("messages", &messages).unwrap();
            omitted.set_item("extra_headers", &headers).unwrap();
            let explicit = omitted.copy().unwrap();
            explicit.set_item("optional_params", py.None()).unwrap();

            let omitted_error = module
                .getattr("chat_completions")
                .and_then(|function| function.call1((&omitted, (), PyDict::new(py), py.None())))
                .expect_err("omitted optional_params should reach header validation");
            let explicit_error = module
                .getattr("chat_completions")
                .and_then(|function| function.call1((&explicit, (), PyDict::new(py), py.None())))
                .expect_err("None optional_params should reach header validation");
            assert!(omitted_error.is_instance_of::<crate::errors::RustBridgeDeclined>(py));
            assert_eq!(explicit_error.to_string(), omitted_error.to_string());
        });
    }

    #[test]
    fn generated_routes_execute_sync_and_async_contracts() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "synthetic").expect("module should be created");
            synthetic::register(&module).expect("routes should register");

            let sync_value: String = module
                .getattr("echo")
                .and_then(|function| function.call1(("sync",)))
                .and_then(|value| value.extract())
                .expect("sync route should return its value");
            assert_eq!(sync_value, "sync");

            let sync_error = module
                .getattr("echo")
                .and_then(|function| function.call1(("error",)))
                .expect_err("sync route should map its error");
            assert!(sync_error.is_instance_of::<PyLookupError>(py));
            assert_eq!(
                sync_error.to_string(),
                "LookupError: invalid request: synthetic error"
            );

            let locals = PyDict::new(py);
            locals
                .set_item("routes", &module)
                .expect("module should enter Python locals");
            let code = CString::new(
                r#"
import asyncio

async def exercise():
    assert await routes.aecho("async") == "async"

    try:
        await routes.aecho("error")
    except LookupError as error:
        assert str(error) == "invalid request: synthetic error"
    else:
        raise AssertionError("mapped error was not raised")

    try:
        await routes.aecho("panic")
    except BaseException as error:
        assert type(error).__name__ == "PanicException"
        assert str(error) == "synthetic panic"
    else:
        raise AssertionError("panic was not raised")

    try:
        await routes.aecho("map_panic")
    except BaseException as error:
        assert type(error).__name__ == "PanicException"
        assert str(error) == "synthetic mapper panic"
    else:
        raise AssertionError("mapper panic was not raised")

    task = asyncio.ensure_future(routes.aecho("pending"))
    await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("cancelled route completed")

    for _ in range(100):
        if routes.future_dropped():
            break
        await asyncio.sleep(0.001)
    assert routes.future_dropped()

asyncio.run(exercise())
"#,
            )
            .expect("Python source should not contain null bytes");
            py.run(&code, Some(&locals), Some(&locals))
                .expect("async route contract should hold");
        });
    }

    #[cfg(feature = "trace-parity")]
    #[test]
    fn diagnostic_route_returns_the_response_and_filtered_trace() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "synthetic").expect("module should be created");
            synthetic::register_trace(&module).expect("trace routes should register");
            let locals = PyDict::new(py);
            locals
                .set_item("routes", &module)
                .expect("module should enter Python locals");
            let code = CString::new(
                r#"
result = routes.echo("traced")
assert result["response"] == "traced", result
assert [event["function"] for event in result["trace"]] == ["execute_echo"], result
failure = routes.echo("error")
assert failure["error"] == "invalid request: synthetic error", failure
assert [event["function"] for event in failure["trace"]] == ["execute_echo"], failure
"#,
            )
            .expect("Python source should not contain null bytes");
            py.run(&code, Some(&locals), Some(&locals))
                .expect("diagnostic route should return its response and trace");
        });
    }

    #[test]
    fn route_registration_rejects_duplicate_python_names() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "synthetic").expect("module should be created");
            synthetic::register(&module).expect("first registration should succeed");
            let error = synthetic::register(&module)
                .expect_err("duplicate registration should be rejected");

            assert_eq!(
                error.to_string(),
                "RuntimeError: duplicate native route: future_dropped"
            );
        });
    }
}
