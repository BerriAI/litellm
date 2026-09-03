//! Build the router by calling the Python proxy config reader (load time only).
//!
//! Embeds the interpreter via pyo3 and calls
//! `litellm.proxy.read_model_list.read_model_list`, which reuses the proxy's
//! `os.environ/` + secret-manager resolution. The GIL is taken **once at boot**
//! (and recorded in [`crate::gil`]); the realtime hot path never touches Python.
//!
//! Compiled only under the `python-config` feature.
use litellm_core::error::Error;
use litellm_core::router::{Deployment, Router};
use pyo3::prelude::*;

use crate::gil;

/// Load the router's `model_list` from `config_path` via the Python reader.
pub fn load_router_from_config(config_path: &str) -> Result<Router, Error> {
    gil::record_acquisition();
    Python::attach(|py| {
        let model_list = py
            .import("litellm.proxy.read_model_list")
            .and_then(|module| module.getattr("read_model_list"))
            .and_then(|reader| reader.call1((config_path,)))
            .map_err(|err| Error::Routing(format!("read_model_list failed: {err}")))?;

        let model_list_json: String = py
            .import("json")
            .and_then(|json| json.getattr("dumps"))
            .and_then(|dumps| dumps.call1((model_list,)))
            .and_then(|encoded| encoded.extract())
            .map_err(|err| Error::Routing(format!("serializing model_list failed: {err}")))?;

        let deployments: Vec<Deployment> = serde_json::from_str(&model_list_json)
            .map_err(|err| Error::Routing(format!("parsing model_list failed: {err}")))?;

        Ok(Router::new(deployments))
    })
}

#[cfg(all(test, feature = "python-config"))]
mod tests {
    use std::ffi::CStr;
    use std::sync::{Mutex, MutexGuard};

    use pyo3::PyAny;
    use pyo3::types::PyDict;

    use super::*;
    use crate::gil;

    const TWO_DEPLOYMENT_MODEL_LIST: &str = r#"[
        {
            "model_name": "gpt-realtime",
            "litellm_params": {
                "model": "openai/gpt-realtime",
                "api_key": "sk-primary",
                "api_base": "https://primary.example.com"
            }
        },
        {
            "model_name": "gpt-realtime",
            "litellm_params": {"model": "openai/gpt-realtime", "api_key": "sk-secondary"}
        }
    ]"#;

    const READER_RETURNING_PAYLOAD: &CStr = cr#"
import sys, types, json
litellm = types.ModuleType("litellm")
proxy = types.ModuleType("litellm.proxy")
reader = types.ModuleType("litellm.proxy.read_model_list")
def read_model_list(config_path):
    reader.calls.append(config_path)
    return json.loads(model_list_json)
reader.calls = []
reader.read_model_list = read_model_list
litellm.proxy = proxy
proxy.read_model_list = reader
sys.modules["litellm"] = litellm
sys.modules["litellm.proxy"] = proxy
sys.modules["litellm.proxy.read_model_list"] = reader
"#;

    const READER_RETURNING_UNSERIALIZABLE: &CStr = cr#"
import sys, types
litellm = types.ModuleType("litellm")
proxy = types.ModuleType("litellm.proxy")
reader = types.ModuleType("litellm.proxy.read_model_list")
def read_model_list(config_path):
    return set((1, 2))
reader.read_model_list = read_model_list
litellm.proxy = proxy
proxy.read_model_list = reader
sys.modules["litellm"] = litellm
sys.modules["litellm.proxy"] = proxy
sys.modules["litellm.proxy.read_model_list"] = reader
"#;

    const READER_ATTRIBUTE_MISSING: &CStr = cr#"
import sys, types
litellm = types.ModuleType("litellm")
proxy = types.ModuleType("litellm.proxy")
reader = types.ModuleType("litellm.proxy.read_model_list")
litellm.proxy = proxy
proxy.read_model_list = reader
sys.modules["litellm"] = litellm
sys.modules["litellm.proxy"] = proxy
sys.modules["litellm.proxy.read_model_list"] = reader
"#;

    const READER_MODULE_MISSING: &CStr = cr#"
import sys, types
litellm = types.ModuleType("litellm")
proxy = types.ModuleType("litellm.proxy")
litellm.proxy = proxy
sys.modules["litellm"] = litellm
sys.modules["litellm.proxy"] = proxy
"#;

    const STUB_MODULE_NAMES: [&str; 3] =
        ["litellm", "litellm.proxy", "litellm.proxy.read_model_list"];

    fn stub_lock() -> MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn sys_modules(py: Python<'_>) -> Bound<'_, PyDict> {
        py.import("sys")
            .and_then(|sys| sys.getattr("modules"))
            .expect("sys.modules should resolve")
            .cast_into::<PyDict>()
            .expect("sys.modules should be a dict")
    }

    struct StubRestore {
        previous: Vec<Option<Py<PyAny>>>,
    }

    impl Drop for StubRestore {
        fn drop(&mut self) {
            Python::attach(|py| {
                let modules = sys_modules(py);
                for (name, previous) in STUB_MODULE_NAMES.iter().zip(&self.previous) {
                    match previous {
                        Some(module) => modules
                            .set_item(name, module)
                            .expect("restoring sys.modules should succeed"),
                        None => {
                            let _ = modules.del_item(name);
                        }
                    }
                }
            });
        }
    }

    fn install_stub(py: Python<'_>, source: &CStr, globals: &Bound<'_, PyDict>) -> StubRestore {
        let modules = sys_modules(py);
        let previous = STUB_MODULE_NAMES
            .iter()
            .map(|name| {
                modules
                    .get_item(name)
                    .expect("reading sys.modules should succeed")
                    .map(Bound::unbind)
            })
            .collect();
        py.run(source, Some(globals), None)
            .expect("installing the stub modules should succeed");
        StubRestore { previous }
    }

    fn load_with_stub(
        _stub_lock: &MutexGuard<'static, ()>,
        source: &CStr,
        model_list_json: Option<&str>,
    ) -> Result<Router, Error> {
        Python::initialize();
        Python::attach(|py| {
            let globals = PyDict::new(py);
            if let Some(model_list_json) = model_list_json {
                globals
                    .set_item("model_list_json", model_list_json)
                    .expect("injecting the stub payload should succeed");
            }
            let _restore = install_stub(py, source, &globals);
            load_router_from_config("proxy_config.yaml")
        })
    }

    fn assert_routing_error(error: &Error, expected_prefix: &str) {
        assert!(
            matches!(error, Error::Routing(message) if message.starts_with(expected_prefix)),
            "expected a routing error starting with {expected_prefix:?}, got {error:?}"
        );
    }

    #[test]
    fn builds_a_router_from_the_resolved_model_list() {
        let _stub_lock = stub_lock();
        Python::initialize();
        Python::attach(|py| {
            let globals = PyDict::new(py);
            globals
                .set_item("model_list_json", TWO_DEPLOYMENT_MODEL_LIST)
                .expect("injecting the stub payload should succeed");
            let _restore = install_stub(py, READER_RETURNING_PAYLOAD, &globals);

            let acquisitions_before = gil::snapshot().total_acquisitions;
            let router = load_router_from_config("proxy_config.yaml")
                .expect("the stubbed model list should load");

            assert_eq!(
                gil::snapshot().total_acquisitions,
                acquisitions_before + 1,
                "loading the config must record its load-time GIL acquisition"
            );
            let deployments = router.deployments();
            assert_eq!(deployments.len(), 2);
            assert_eq!(deployments[0].model_name, "gpt-realtime");
            assert_eq!(deployments[0].litellm_params.model, "openai/gpt-realtime");
            assert_eq!(
                deployments[0].litellm_params.api_key.as_deref(),
                Some("sk-primary")
            );
            assert_eq!(
                deployments[0].litellm_params.api_base.as_deref(),
                Some("https://primary.example.com")
            );
            assert_eq!(
                deployments[1].litellm_params.api_key.as_deref(),
                Some("sk-secondary")
            );
            assert_eq!(deployments[1].litellm_params.api_base, None);
            assert!(router.has_deployment("gpt-realtime"));
            assert!(!router.has_deployment("unknown-model"));
            let chosen = router
                .get_available_deployment("gpt-realtime")
                .expect("the model_name group should have candidates");
            assert_eq!(chosen.model_name, "gpt-realtime");

            let received_paths: Vec<String> = sys_modules(py)
                .get_item("litellm.proxy.read_model_list")
                .expect("reading the stub module should succeed")
                .expect("the stub reader module should still be registered")
                .getattr("calls")
                .expect("the stub should expose its recorded calls")
                .extract()
                .expect("the recorded calls should be a list of strings");
            assert_eq!(received_paths, ["proxy_config.yaml"]);
        });
    }

    #[test]
    fn routing_error_when_entries_do_not_match_the_deployment_shape() {
        let stub_lock = stub_lock();
        let error = load_with_stub(
            &stub_lock,
            READER_RETURNING_PAYLOAD,
            Some(r#"[{"model_name": "gpt-realtime"}]"#),
        )
        .expect_err("a model list without litellm_params must not load");
        assert_routing_error(&error, "parsing model_list failed");
    }

    #[test]
    fn routing_error_when_the_model_list_is_not_json_serializable() {
        let stub_lock = stub_lock();
        let error = load_with_stub(&stub_lock, READER_RETURNING_UNSERIALIZABLE, None)
            .expect_err("a set is not a JSON-serializable model list");
        assert_routing_error(&error, "serializing model_list failed");
    }

    #[test]
    fn routing_error_when_the_reader_module_lacks_the_function() {
        let stub_lock = stub_lock();
        let error = load_with_stub(&stub_lock, READER_ATTRIBUTE_MISSING, None)
            .expect_err("a reader module without read_model_list must not load");
        assert_routing_error(&error, "read_model_list failed");
    }

    #[test]
    fn routing_error_when_the_reader_module_is_not_importable() {
        let stub_lock = stub_lock();
        let error = load_with_stub(&stub_lock, READER_MODULE_MISSING, None)
            .expect_err("a missing reader module must not load");
        assert_routing_error(&error, "read_model_list failed");
    }
}
