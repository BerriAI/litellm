//! Build the router by calling the Python proxy config reader (load time only).
//!
//! Embeds the interpreter via pyo3 and calls
//! `litellm.proxy.read_model_list.read_model_list`, which reuses the proxy's
//! `os.environ/` + secret-manager resolution. The GIL is taken **once at boot**
//! (and recorded in [`crate::gil`]); the realtime hot path never touches Python.
//!
//! Compiled only under the `python-config` feature.

use litellm_core::CoreResult;
use litellm_core::error::CoreError;
use litellm_core::router::{Deployment, Router};
use pyo3::prelude::*;

use crate::gil;

/// Load the router's `model_list` from `config_path` via the Python reader.
pub fn load_router_from_config(config_path: &str) -> CoreResult<Router> {
    gil::record_acquisition();
    Python::attach(|py| {
        let model_list = py
            .import("litellm.proxy.read_model_list")
            .and_then(|module| module.getattr("read_model_list"))
            .and_then(|reader| reader.call1((config_path,)))
            .map_err(|err| CoreError::Routing(format!("read_model_list failed: {err}")))?;

        let model_list_json: String = py
            .import("json")
            .and_then(|json| json.getattr("dumps"))
            .and_then(|dumps| dumps.call1((model_list,)))
            .and_then(|encoded| encoded.extract())
            .map_err(|err| CoreError::Routing(format!("serializing model_list failed: {err}")))?;

        let deployments: Vec<Deployment> = serde_json::from_str(&model_list_json)
            .map_err(|err| CoreError::Routing(format!("parsing model_list failed: {err}")))?;

        Ok(Router::new(deployments))
    })
}
