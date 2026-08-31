use std::future::Future;

use litellm_core::error::CoreResult;
use pyo3::prelude::*;
use serde::Serialize;

mod runtime;

use runtime::{run_async, run_sync};

trait BridgeRoute<I>: Sized {
    type Output: Serialize + Send + 'static;

    fn from_python(py: Python<'_>, inputs: I) -> PyResult<Self>;

    fn run(self) -> impl Future<Output = CoreResult<Self::Output>> + Send + 'static;
}

macro_rules! bridge_route {
    (
        sync = $sync_name:ident,
        asynchronous = $async_name:ident,
        inputs = $inputs:ident,
        required = { $($required_name:ident: $required_type:ty),* $(,)? },
        optional = { $($optional_name:ident: $optional_type:ty),* $(,)? },
        call = $call:ty,
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
            $($required_name: $required_type,)*
            $($optional_name: $optional_type),*
        ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
            let call = <$call as crate::routes::BridgeRoute<$inputs>>::from_python(py, $inputs {
                $($required_name,)*
                $($optional_name),*
            })?;
            crate::routes::run_sync(
                py,
                <$call as crate::routes::BridgeRoute<$inputs>>::run(call),
                $map_error,
            )
        }

        #[pyfunction]
        #[pyo3(signature = ($($required_name),*, $($optional_name=None),*))]
        #[allow(clippy::too_many_arguments)]
        fn $async_name(
            py: pyo3::Python<'_>,
            $($required_name: $required_type,)*
            $($optional_name: $optional_type),*
        ) -> pyo3::PyResult<pyo3::Bound<'_, pyo3::PyAny>> {
            let call = <$call as crate::routes::BridgeRoute<$inputs>>::from_python(py, $inputs {
                $($required_name,)*
                $($optional_name),*
            })?;
            crate::routes::run_async(
                py,
                <$call as crate::routes::BridgeRoute<$inputs>>::run(call),
                $map_error,
            )
        }

        pub(super) fn register(
            module: &pyo3::Bound<'_, pyo3::types::PyModule>,
        ) -> pyo3::PyResult<()> {
            module.add_function(pyo3::wrap_pyfunction!($sync_name, module)?)?;
            module.add_function(pyo3::wrap_pyfunction!($async_name, module)?)?;
            $($(module.add_function(pyo3::wrap_pyfunction!($extra, module)?)?;)*)?
            Ok(())
        }
    };
}

macro_rules! routes {
    ($($route:ident),* $(,)?) => {
        $(mod $route;)*

        pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
            $($route::register(module)?;)*
            Ok(())
        }
    };
}

routes!(ocr, audio_transcription, messages, chat_completions);

#[cfg(test)]
mod tests {
    use pyo3::types::{PyDict, PyList};

    use super::*;

    #[test]
    fn sync_and_async_route_signatures_match_the_python_contract() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "routes").expect("module should be created");
            register(&module).expect("routes should register");
            let routes = [
                (
                    "ocr",
                    "aocr",
                    "(model, document, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None)",
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
            register(&module).expect("routes should register");

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
}
