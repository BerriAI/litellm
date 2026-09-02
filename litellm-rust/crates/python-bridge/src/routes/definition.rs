use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyCFunction;

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
            $($(#[$optional_attr])* $optional_name: $optional_type),*
        ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
            let future = $prepare($inputs {
                $($required_name,)*
                $($optional_name),*
            })?;
            $crate::routes::runtime::run_sync(py, future, $map_error)
        }

        #[pyfunction]
        #[pyo3(signature = ($($required_name),*, $($optional_name=None),*))]
        #[allow(clippy::too_many_arguments)]
        fn $async_name(
            py: pyo3::Python<'_>,
            $($(#[$required_attr])* $required_name: $required_type,)*
            $($(#[$optional_attr])* $optional_name: $optional_type),*
        ) -> pyo3::PyResult<pyo3::Bound<'_, pyo3::PyAny>> {
            let future = $prepare($inputs {
                $($required_name,)*
                $($optional_name),*
            })?;
            $crate::routes::runtime::run_async(py, future, $map_error)
        }

        pub(super) fn register(
            module: &pyo3::Bound<'_, pyo3::types::PyModule>,
        ) -> pyo3::PyResult<()> {
            $($($crate::routes::definition::add_function(module, pyo3::wrap_pyfunction!($extra, module)?)?;)*)?
            $crate::routes::definition::add_function(module, pyo3::wrap_pyfunction!($sync_name, module)?)?;
            $crate::routes::definition::add_function(module, pyo3::wrap_pyfunction!($async_name, module)?)?;
            Ok(())
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
            Ok(async move {
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
            })
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
                (
                    "ocr",
                    "aocr",
                    "(model, document, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None, trace=None)",
                ),
                (
                    "transcription",
                    "atranscription",
                    "(model, audio, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None)",
                ),
                (
                    "messages",
                    "amessages",
                    "(model, body, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, timeout_seconds=None)",
                ),
                (
                    "chat_completions",
                    "achat_completions",
                    "(model, messages, optional_params=None, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, timeout_seconds=None)",
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
            let sync_chat_error = module
                .getattr("chat_completions")
                .and_then(|function| function.call1(("model", &invalid_messages)))
                .expect_err("sync chat should reject a non-list messages value");
            let async_chat_error = module
                .getattr("achat_completions")
                .and_then(|function| function.call1(("model", &invalid_messages)))
                .expect_err("async chat should reject a non-list messages value");

            assert_eq!(
                sync_chat_error.to_string(),
                "ValueError: messages must be a list"
            );
            assert_eq!(async_chat_error.to_string(), sync_chat_error.to_string());

            let invalid_body = PyList::empty(py);
            let sync_messages_error = module
                .getattr("messages")
                .and_then(|function| function.call1(("model", &invalid_body)))
                .expect_err("sync Messages should reject a non-dict body");
            let async_messages_error = module
                .getattr("amessages")
                .and_then(|function| function.call1(("model", &invalid_body)))
                .expect_err("async Messages should reject a non-dict body");

            assert_eq!(
                sync_messages_error.to_string(),
                "ValueError: body must be a dict"
            );
            assert_eq!(
                async_messages_error.to_string(),
                sync_messages_error.to_string()
            );

            let invalid_headers = PyList::empty(py);
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("extra_headers", &invalid_headers)
                .expect("kwargs should accept extra_headers");
            let document = PyDict::new(py);

            for (sync_name, async_name) in [("ocr", "aocr"), ("transcription", "atranscription")] {
                let sync_error = module
                    .getattr(sync_name)
                    .and_then(|function| function.call(("model", &document), Some(&kwargs)))
                    .expect_err("sync route should reject non-dict extra_headers");
                let async_error = module
                    .getattr(async_name)
                    .and_then(|function| function.call(("model", &document), Some(&kwargs)))
                    .expect_err("async route should reject non-dict extra_headers");

                assert_eq!(
                    sync_error.to_string(),
                    "ValueError: extra_headers must be a dict"
                );
                assert_eq!(async_error.to_string(), sync_error.to_string());
            }
        });
    }

    #[test]
    fn route_input_validation_preserves_left_to_right_order() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "routes").expect("module should be created");
            crate::routes::register(&module).expect("routes should register");
            let invalid = PyList::empty(py);

            let chat_kwargs = PyDict::new(py);
            chat_kwargs
                .set_item("optional_params", &invalid)
                .expect("kwargs should accept optional_params");
            chat_kwargs
                .set_item("extra_headers", &invalid)
                .expect("kwargs should accept extra_headers");
            let invalid_messages = PyDict::new(py);
            let error = module
                .getattr("chat_completions")
                .and_then(|function| {
                    function.call(("model", &invalid_messages), Some(&chat_kwargs))
                })
                .expect_err("messages should be validated first");
            assert_eq!(error.to_string(), "ValueError: messages must be a list");

            let valid_messages = PyList::empty(py);
            let error = module
                .getattr("chat_completions")
                .and_then(|function| function.call(("model", &valid_messages), Some(&chat_kwargs)))
                .expect_err("optional_params should be validated before headers");
            assert_eq!(
                error.to_string(),
                "ValueError: optional_params must be a dict"
            );

            let headers_kwargs = PyDict::new(py);
            headers_kwargs
                .set_item("extra_headers", &invalid)
                .expect("kwargs should accept extra_headers");
            let invalid_body = PyList::empty(py);
            let error = module
                .getattr("messages")
                .and_then(|function| function.call(("model", &invalid_body), Some(&headers_kwargs)))
                .expect_err("body should be validated before headers");
            assert_eq!(error.to_string(), "ValueError: body must be a dict");

            let invalid_payload =
                PyModule::new(py, "invalid_payload").expect("invalid payload should be created");
            for name in ["ocr", "transcription"] {
                let error = module
                    .getattr(name)
                    .and_then(|function| {
                        function.call(("model", &invalid_payload), Some(&headers_kwargs))
                    })
                    .expect_err("payload should be validated before headers");
                assert!(!error.to_string().contains("extra_headers"));
            }
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

    #[test]
    fn messages_routes_only_decline_unsupported_requests() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "routes").expect("module should be created");
            crate::routes::register(&module).expect("routes should register");
            let body = PyDict::new(py);
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("custom_llm_provider", "openai")
                .expect("kwargs should accept provider");

            let error = module
                .getattr("messages")
                .and_then(|function| function.call(("model", &body), Some(&kwargs)))
                .expect_err("unsupported provider should decline");
            assert!(error.is_instance_of::<crate::errors::RustBridgeDeclined>(py));

            kwargs
                .set_item("custom_llm_provider", "anthropic")
                .expect("kwargs should accept provider");
            let error = module
                .getattr("messages")
                .and_then(|function| function.call(("model", &body), Some(&kwargs)))
                .expect_err("missing credentials should fail");
            assert!(error.is_instance_of::<crate::errors::RustUpstreamError>(py));
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
