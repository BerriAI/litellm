use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyCFunction;

macro_rules! bridge_route {
    (
        sync = $sync_name:ident,
        asynchronous = $async_name:ident,
        request = $inputs:ident,
        prepare = $prepare:path,
        errors = $map_error:path
        $(, extra = [$($extra:ident),* $(,)?])? $(,)?
    ) => {
        #[pyfunction]
        #[pyo3(signature = (request, *, context))]
        fn $sync_name(
            py: pyo3::Python<'_>,
            request: $inputs,
            context: $crate::marshal::NativeRequestContext,
        ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
            let future = $prepare(request, context)?;
            $crate::execution::run_sync(py, future, $map_error)
        }

        #[pyfunction]
        #[pyo3(signature = (request, *, context))]
        fn $async_name(
            py: pyo3::Python<'_>,
            request: $inputs,
            context: $crate::marshal::NativeRequestContext,
        ) -> pyo3::PyResult<pyo3::Bound<'_, pyo3::PyAny>> {
            let future = $prepare(request, context)?;
            $crate::execution::run_async(py, future, $map_error)
        }

        pub(super) fn register(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
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
            #[pyo3(signature = (request, *, context))]
            fn $sync_name(
                py: pyo3::Python<'_>,
                request: $inputs,
                context: $crate::marshal::NativeRequestContext,
            ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                let future = $prepare(request, context)?;
                $crate::execution::run_sync(py, $crate::function_trace::capture(future), $map_error)
            }

            #[pyfunction]
            #[pyo3(signature = (request, *, context))]
            fn $async_name(
                py: pyo3::Python<'_>,
                request: $inputs,
                context: $crate::marshal::NativeRequestContext,
            ) -> pyo3::PyResult<pyo3::Bound<'_, pyo3::PyAny>> {
                let future = $prepare(request, context)?;
                $crate::execution::run_async(py, $crate::function_trace::capture(future), $map_error)
            }

            pub(super) fn register(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
                $crate::routes::definition::add_function(module, pyo3::wrap_pyfunction!($sync_name, module)?)?;
                $crate::routes::definition::add_function(module, pyo3::wrap_pyfunction!($async_name, module)?)?;
                Ok(())
            }
        }

        #[cfg(feature = "trace-parity")]
        pub(super) fn register_trace(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
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
    use pyo3::types::PyDict;

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

        #[derive(FromPyObject)]
        struct EchoInputs {
            value: String,
        }

        bridge_route! {
            sync = echo,
            asynchronous = aecho,
            request = EchoInputs,
            prepare = prepare_echo,
            errors = map_error,
            extra = [future_dropped],
        }

        fn prepare_echo(
            inputs: EchoInputs,
            _context: crate::marshal::NativeRequestContext,
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
                ("ocr", "aocr", "(request, *, context)"),
                ("transcription", "atranscription", "(request, *, context)"),
                ("messages", "amessages", "(request, *, context)"),
                (
                    "chat_completions",
                    "achat_completions",
                    "(request, *, context)",
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
    fn routes_validate_dataclass_inputs_before_execution() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "routes").expect("module should be created");
            crate::routes::register(&module).expect("routes should register");
            let locals = crate::marshal::request_fixtures(py);
            locals.set_item("routes", module).unwrap();
            py.run(c"
for names, request, expected in [
    (('chat_completions', 'achat_completions'), Request(messages={}, optional_params={}), 'messages must be a list'),
    (('messages', 'amessages'), Request(body=[]), 'body must be a dict'),
    (('ocr', 'aocr'), Request(document={}, optional_params={}, options=Options(extra_headers=[])), 'extra_headers'),
    (('transcription', 'atranscription'), Request(audio={}, optional_params={}, options=Options(timeout_seconds='bad')), 'timeout_seconds'),
    (('transcription', 'atranscription'), Request(audio={}, optional_params={}, options=Options(provider_connection=[])), 'provider_connection'),
]:
    errors = []
    for name in names:
        try:
            getattr(routes, name)(request, context=context)
        except (ValueError, TypeError) as error:
            parts = []
            while error is not None:
                parts.append(str(error))
                error = error.__cause__
            errors.append(' / '.join(parts))
        else:
            raise AssertionError('invalid input reached execution')
    assert errors[0] == errors[1], errors
    assert expected in errors[0], (expected, errors)

for field in ('metadata', 'litellm_metadata', 'request_metadata_fields'):
    invalid_context = replace(context, **{field: object()})
    try:
        routes.chat_completions(Request(messages=[], optional_params={}), context=invalid_context)
    except (ValueError, TypeError) as error:
        assert field in str(error)
    else:
        raise AssertionError('invalid context reached execution')
", Some(&locals), Some(&locals)).expect("native input validation should match");
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
                .and_then(|function| {
                    let locals = crate::marshal::request_fixtures(py);
                    py.eval(c"Request(value=\"sync\")", Some(&locals), Some(&locals))
                        .and_then(|request| {
                            let kwargs = PyDict::new(py);
                            kwargs.set_item("context", locals.get_item("context")?.unwrap())?;
                            function.call((request,), Some(&kwargs))
                        })
                })
                .and_then(|value| value.extract())
                .expect("sync route should return its value");
            assert_eq!(sync_value, "sync");

            let sync_error = module
                .getattr("echo")
                .and_then(|function| {
                    let locals = crate::marshal::request_fixtures(py);
                    py.eval(c"Request(value=\"error\")", Some(&locals), Some(&locals))
                        .and_then(|request| {
                            let kwargs = PyDict::new(py);
                            kwargs.set_item("context", locals.get_item("context")?.unwrap())?;
                            function.call((request,), Some(&kwargs))
                        })
                })
                .expect_err("sync route should map its error");
            assert!(sync_error.is_instance_of::<PyLookupError>(py));
            assert_eq!(
                sync_error.to_string(),
                "LookupError: invalid request: synthetic error"
            );

            let locals = crate::marshal::request_fixtures(py);
            locals
                .set_item("routes", &module)
                .expect("module should enter Python locals");
            let code = CString::new(
                r#"
import asyncio

async def exercise():
    assert await routes.aecho(Request(value="async"), context=context) == "async"

    try:
        await routes.aecho(Request(value="error"), context=context)
    except LookupError as error:
        assert str(error) == "invalid request: synthetic error"
    else:
        raise AssertionError("mapped error was not raised")

    try:
        await routes.aecho(Request(value="panic"), context=context)
    except BaseException as error:
        assert type(error).__name__ == "PanicException"
        assert str(error) == "synthetic panic"
    else:
        raise AssertionError("panic was not raised")

    try:
        await routes.aecho(Request(value="map_panic"), context=context)
    except BaseException as error:
        assert type(error).__name__ == "PanicException"
        assert str(error) == "synthetic mapper panic"
    else:
        raise AssertionError("mapper panic was not raised")

    task = asyncio.ensure_future(routes.aecho(Request(value="pending"), context=context))
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
            let locals = crate::marshal::request_fixtures(py);
            locals
                .set_item("routes", &module)
                .expect("module should enter Python locals");
            let code = CString::new(
                r#"
result = routes.echo(Request(value="traced"), context=context)
assert result == {
    "response": "traced",
    "trace": [{"function": "execute_echo", "depth": 0}],
}
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
