//! The OCR-specific slice of the legacy `Logging` contract: `update_from_kwargs`,
//! `pre_call`, `post_call` and their payload-free shortcuts. Expires with the contract.

use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::Serialize;
use serde_json::{Map, Value};

use litellm_python_interop::{from_py, to_py};

use crate::{LegacyCallbacks, PythonLogger};

/// What the `Logging` object is told about the request it is about to see.
pub struct OcrRequestFacts<'a> {
    pub model: &'a str,
    pub custom_llm_provider: &'a str,
    pub url: &'a str,
    pub optional_params: &'a Value,
}

/// The body and headers callbacks were shown, read back after they ran.
pub struct OcrRequestPayload {
    pub body: Value,
    pub headers: Vec<(String, String)>,
}

/// The Python objects the legacy `Logging` contract must see again after the request has
/// been projected into Rust: callbacks receive the caller's own document and api key, and
/// may mutate the body and headers they are shown before the request is sent.
pub struct OcrCallbackRetention {
    document: Option<Py<PyAny>>,
    api_key: Py<PyAny>,
    secret_fields: Vec<&'static str>,
    retained_fields: Option<Py<PyDict>>,
    body: Option<Py<PyDict>>,
    headers: Option<Py<PyDict>>,
}

impl OcrCallbackRetention {
    pub fn new(
        document: Option<Py<PyAny>>,
        api_key: Py<PyAny>,
        secret_fields: Vec<&'static str>,
    ) -> Self {
        Self {
            document,
            api_key,
            secret_fields,
            retained_fields: None,
            body: None,
            headers: None,
        }
    }

    pub fn pre_call(
        &mut self,
        py: Python<'_>,
        kwargs: &Py<PyDict>,
        optional_params: &Map<String, Value>,
        document: &impl Serialize,
    ) -> PyResult<()> {
        let kwargs = kwargs.bind(py);
        let retained_fields = PyDict::new(py);
        for name in optional_params.keys() {
            if let Some(value) = kwargs.get_item(name)? {
                retained_fields.set_item(name, value)?;
            }
        }
        let document = match &self.document {
            Some(document) => document.clone_ref(py),
            None => to_py(py, document)?,
        };
        retained_fields.set_item("document", &document)?;
        self.document = Some(document);
        self.retained_fields = Some(retained_fields.unbind());
        Ok(())
    }

    pub fn during_call(
        &mut self,
        py: Python<'_>,
        logger: &PythonLogger,
        kwargs: &Py<PyDict>,
        facts: OcrRequestFacts<'_>,
        mut payload: OcrRequestPayload,
        retained_fields: &[String],
    ) -> PyResult<OcrRequestPayload> {
        logger.update_ocr(py, kwargs, &facts, &self.secret_fields)?;
        if !logger.callbacks_needed(py, "payload")? {
            logger.record_api_call_start(py)?;
            return Ok(payload);
        }
        if let Some(body) = payload.body.as_object_mut() {
            for name in retained_fields {
                body.remove(name);
            }
        }
        let body = to_py(py, &payload.body)?
            .into_bound(py)
            .cast_into::<PyDict>()?;
        if let Some(retained) = &self.retained_fields {
            for name in retained_fields {
                if let Some(value) = retained.bind(py).get_item(name)? {
                    body.set_item(name, value)?;
                }
            }
        }
        let headers = PyDict::new(py);
        for (name, value) in &payload.headers {
            headers.set_item(name, value)?;
        }
        self.body = Some(body.clone().unbind());
        self.headers = Some(headers.clone().unbind());
        logger.pre_ocr(
            py,
            &Some(self.api_key.clone_ref(py)),
            &body,
            &headers,
            facts.url,
        )?;
        let headers = headers
            .iter()
            .map(|(name, value)| Ok((name.extract::<String>()?, value.extract::<String>()?)))
            .collect::<PyResult<Vec<_>>>()?;
        payload.body = from_py(&body)?;
        payload.headers = headers;
        Ok(payload)
    }

    pub fn post_call(
        &self,
        py: Python<'_>,
        logger: &PythonLogger,
        original_response: &Value,
    ) -> PyResult<()> {
        if !logger.callbacks_needed(py, "payload")? {
            return Ok(());
        }
        logger.post_ocr(
            py,
            original_response,
            self.body.as_ref(),
            self.headers.as_ref(),
        )
    }

    pub fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.document)?;
        visit.call(&self.api_key)?;
        visit.call(&self.retained_fields)?;
        visit.call(&self.body)?;
        visit.call(&self.headers)
    }
}

trait OcrLogger {
    fn update_ocr(
        &self,
        py: Python<'_>,
        kwargs: &Py<PyDict>,
        facts: &OcrRequestFacts<'_>,
        secret_fields: &[&str],
    ) -> PyResult<()>;

    fn record_api_call_start(&self, py: Python<'_>) -> PyResult<()>;

    fn pre_ocr(
        &self,
        py: Python<'_>,
        api_key: &Option<Py<PyAny>>,
        body: &Bound<'_, PyDict>,
        headers: &Bound<'_, PyDict>,
        url: &str,
    ) -> PyResult<()>;

    fn post_ocr(
        &self,
        py: Python<'_>,
        original_response: &Value,
        body: Option<&Py<PyDict>>,
        headers: Option<&Py<PyDict>>,
    ) -> PyResult<()>;
}

impl OcrLogger for PythonLogger {
    fn update_ocr(
        &self,
        py: Python<'_>,
        kwargs: &Py<PyDict>,
        facts: &OcrRequestFacts<'_>,
        secret_fields: &[&str],
    ) -> PyResult<()> {
        let update = PyDict::new(py);
        update.set_item("kwargs", redact(py, kwargs.bind(py), secret_fields)?)?;
        update.set_item("model", facts.model)?;
        update.set_item(
            "optional_params",
            redact(
                py,
                &to_py(py, facts.optional_params)?
                    .into_bound(py)
                    .cast_into::<PyDict>()?,
                secret_fields,
            )?,
        )?;
        let params = PyDict::new(py);
        params.set_item(
            "litellm_call_id",
            kwargs.bind(py).get_item("litellm_call_id")?,
        )?;
        params.set_item("api_base", facts.url)?;
        for name in ["logger_fn", "litellm_request_debug"] {
            if let Some(value) = kwargs.bind(py).get_item(name)? {
                params.set_item(name, value)?;
            }
        }
        for name in custom_pricing_fields(py)? {
            if let Some(value) = kwargs.bind(py).get_item(&name)?
                && !value.is_none()
            {
                params.set_item(name, value)?;
            }
        }
        update.set_item("litellm_params", params)?;
        update.set_item("custom_llm_provider", facts.custom_llm_provider)?;
        self.object(py)
            .call_method("update_from_kwargs", (), Some(&update))?;
        Ok(())
    }

    fn record_api_call_start(&self, py: Python<'_>) -> PyResult<()> {
        self.object(py).call_method0("record_api_call_start_time")?;
        Ok(())
    }

    fn pre_ocr(
        &self,
        py: Python<'_>,
        api_key: &Option<Py<PyAny>>,
        body: &Bound<'_, PyDict>,
        headers: &Bound<'_, PyDict>,
        url: &str,
    ) -> PyResult<()> {
        let additional = PyDict::new(py);
        additional.set_item("complete_input_dict", body)?;
        additional.set_item("headers", headers)?;
        additional.set_item("api_base", url)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("input", "OCR document processing")?;
        kwargs.set_item("api_key", api_key)?;
        kwargs.set_item("additional_args", &additional)?;
        if self.callbacks_needed(py, "input")? {
            self.object(py).call_method("pre_call", (), Some(&kwargs))?;
        } else {
            self.object(py)
                .call_method("_pre_call", (), Some(&kwargs))?;
            self.record_api_call_start(py)?;
        }
        Ok(())
    }

    fn post_ocr(
        &self,
        py: Python<'_>,
        original_response: &Value,
        body: Option<&Py<PyDict>>,
        headers: Option<&Py<PyDict>>,
    ) -> PyResult<()> {
        let additional = PyDict::new(py);
        additional.set_item("complete_input_dict", body)?;
        additional.set_item("headers", headers)?;
        if self.callbacks_needed(py, "input")? {
            let kwargs = PyDict::new(py);
            kwargs.set_item("original_response", to_py(py, original_response)?)?;
            kwargs.set_item("additional_args", &additional)?;
            self.object(py)
                .call_method("post_call", (), Some(&kwargs))?;
        } else {
            let response = py
                .import("json")?
                .call_method1("dumps", (to_py(py, original_response)?,))?;
            self.object(py).call_method1(
                "record_post_call",
                (response, py.None(), py.None(), additional),
            )?;
        }
        Ok(())
    }
}

fn custom_pricing_fields(py: Python<'_>) -> PyResult<Vec<String>> {
    py.import("litellm.types.utils")?
        .getattr("CustomPricingLiteLLMParams")?
        .getattr("model_fields")?
        .cast_into::<PyDict>()?
        .keys()
        .iter()
        .map(|name| name.extract::<String>())
        .collect()
}

fn redact(
    py: Python<'_>,
    params: &Bound<'_, PyDict>,
    secret_fields: &[&str],
) -> PyResult<Py<PyDict>> {
    let redacted = PyDict::new(py);
    for (name, value) in params {
        let name = name.extract::<String>()?;
        if name == "proxy_server_request" {
            continue;
        }
        if secret_fields.contains(&name.as_str()) {
            redacted.set_item(name, "****")?;
        } else {
            redacted.set_item(name, value)?;
        }
    }
    Ok(redacted.unbind())
}
