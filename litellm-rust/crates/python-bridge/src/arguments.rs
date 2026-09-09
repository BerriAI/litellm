use pyo3::prelude::*;

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct ChatAdmissionArguments<'py> {
    #[pyo3(default)]
    pub(crate) model: Option<String>,
    pub(crate) messages: Bound<'py, PyAny>,
    #[pyo3(default)]
    pub(crate) optional_params: Option<Bound<'py, PyAny>>,
    #[pyo3(default)]
    pub(crate) custom_llm_provider: Option<String>,
}

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct ChatBuildArguments<'py> {
    #[pyo3(item("messages"))]
    _messages: Bound<'py, PyAny>,
    #[pyo3(default)]
    pub(crate) api_key: Option<String>,
    #[pyo3(default)]
    pub(crate) api_base: Option<Bound<'py, PyAny>>,
    #[pyo3(default)]
    pub(crate) timeout_seconds: Option<f64>,
    #[pyo3(default)]
    pub(crate) litellm_call_id: Option<String>,
    #[pyo3(default)]
    pub(crate) extra_headers: Option<Bound<'py, PyAny>>,
    #[pyo3(default)]
    pub(crate) logging_api_key: Option<Bound<'py, PyAny>>,
}

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct MessagesArguments<'py> {
    #[pyo3(default)]
    pub(crate) model: Option<String>,
    pub(crate) body: Bound<'py, PyAny>,
    #[pyo3(default)]
    pub(crate) api_key: Option<String>,
    #[pyo3(default)]
    pub(crate) api_base: Option<String>,
    #[pyo3(default)]
    pub(crate) custom_llm_provider: Option<String>,
    #[pyo3(default)]
    pub(crate) extra_headers: Option<Bound<'py, PyAny>>,
    #[pyo3(default)]
    pub(crate) timeout_seconds: Option<f64>,
}
