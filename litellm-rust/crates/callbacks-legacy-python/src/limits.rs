use std::sync::OnceLock;

use litellm_core_utils::token_counter::get_modified_max_tokens;
use litellm_host_python::from_py;
use litellm_token_counter::{CountableRequest, TokenCounter};
use pyo3::{prelude::*, types::PyDict};
use serde_json::Value;

use crate::PublicCall;

pub(crate) fn adjust_max_tokens(
    py: Python<'_>,
    call: &PublicCall,
    call_type: &str,
) -> PyResult<()> {
    if !matches!(
        call_type,
        "completion" | "acompletion" | "anthropic_messages"
    ) {
        return Ok(());
    }
    let sdk = py.import("litellm")?;
    if !sdk.getattr("modify_params")?.is(&true.into_pyobject(py)?) {
        return Ok(());
    }
    let Some(requested) = call
        .lookup(py, "max_tokens")?
        .filter(|value| !value.is_none())
    else {
        return Ok(());
    };
    let requested: i64 = requested.extract()?;
    let Some(model) = call.lookup(py, "model")? else {
        return Ok(());
    };
    let model: String = model.extract()?;
    let info = model_info(py, &model)?;
    let Some(info) = info else {
        return Ok(());
    };
    let counter = counter(py, &model)?;
    let messages = if call.args().bind(py).len() > 1 {
        call.args().bind(py).get_item(1)?
    } else {
        call.lookup(py, "messages")?
            .unwrap_or_else(|| py.None().into_bound(py))
    };
    let request = PyDict::new(py);
    request.set_item("model", &model)?;
    request.set_item("messages", messages)?;
    let request: CountableRequest = from_py(&request)?;
    let count = py
        .detach(|| counter.count_request(&request))
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    let adjusted = get_modified_max_tokens(
        requested,
        info.get("max_input_tokens").and_then(Value::as_i64),
        info.get("max_output_tokens")
            .or_else(|| info.get("max_tokens"))
            .and_then(Value::as_i64),
        count.input_tokens,
        None,
        None,
    );
    call.kwargs().bind(py).set_item("max_tokens", adjusted)
}

pub(crate) fn model_info(py: Python<'_>, model: &str) -> PyResult<Option<Value>> {
    let sdk = py.import("litellm")?;
    let costs = sdk.getattr("model_cost")?;
    let aliases = sdk.getattr("model_alias_map")?;
    let resolved = aliases.call_method1("get", (model, model))?;
    let resolved: String = resolved.extract()?;
    let stripped = resolved.strip_prefix("anthropic/").unwrap_or(&resolved);
    for key in [&resolved, stripped, &format!("anthropic/{stripped}")] {
        let value = costs.call_method1("get", (key,))?;
        if !value.is_none() {
            return from_py(&value).map(Some);
        }
    }
    Ok(None)
}

fn counter(py: Python<'_>, model: &str) -> PyResult<&'static TokenCounter> {
    static CLAUDE: OnceLock<TokenCounter> = OnceLock::new();
    static CL100K: OnceLock<TokenCounter> = OnceLock::new();
    let sdk = py.import("litellm")?;
    let anthropic = sdk.getattr("anthropic_models")?.contains(model)?
        && !model.contains("claude-3")
        && !sdk.getattr("disable_hf_tokenizer_download")?.is_truthy()?;
    let cell = if anthropic { &CLAUDE } else { &CL100K };
    if let Some(counter) = cell.get() {
        return Ok(counter);
    }
    let loaded = if anthropic {
        let json: String = py
            .import("litellm.utils")?
            .getattr("claude_json_str")?
            .extract()?;
        py.detach(|| TokenCounter::from_json(&json))
    } else {
        py.detach(|| TokenCounter::from_tiktoken("cl100k_base"))
    }
    .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    let _ = cell.set(loaded);
    cell.get().ok_or_else(litellm_host_python::missing_state)
}
