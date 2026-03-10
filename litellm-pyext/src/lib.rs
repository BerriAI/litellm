use once_cell::sync::Lazy;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::collections::HashMap;
use std::time::Duration;

static RUNTIME: Lazy<tokio::runtime::Runtime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("Failed to build tokio runtime")
});

static CLIENT: Lazy<reqwest::Client> = Lazy::new(|| {
    reqwest::Client::builder()
        .pool_max_idle_per_host(200)
        .pool_idle_timeout(Duration::from_secs(90))
        .tcp_keepalive(Duration::from_secs(60))
        .tcp_nodelay(true)
        .build()
        .expect("Failed to build reqwest client")
});

#[pyfunction]
#[pyo3(signature = (url, method, headers, body, timeout_secs=300))]
fn forward_request(
    py: Python<'_>,
    url: String,
    method: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
    timeout_secs: u64,
) -> PyResult<(u16, HashMap<String, String>, Py<PyBytes>)> {
    let result = py.allow_threads(|| {
        RUNTIME.block_on(async {
            let req_method = match method.to_uppercase().as_str() {
                "GET" => reqwest::Method::GET,
                "PUT" => reqwest::Method::PUT,
                "PATCH" => reqwest::Method::PATCH,
                "DELETE" => reqwest::Method::DELETE,
                _ => reqwest::Method::POST,
            };

            let mut req = CLIENT.request(req_method.clone(), &url);
            for (k, v) in &headers {
                req = req.header(k.as_str(), v.as_str());
            }
            if req_method != reqwest::Method::GET {
                req = req.body(body);
            }
            req = req.timeout(Duration::from_secs(timeout_secs));

            let resp = req.send().await.map_err(|e| format!("Request failed: {e}"))?;
            let status = resp.status().as_u16();
            let resp_headers: HashMap<String, String> = resp
                .headers()
                .iter()
                .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
                .collect();
            let resp_body = resp.bytes().await.map_err(|e| format!("Body read failed: {e}"))?;
            Ok::<_, String>((status, resp_headers, resp_body.to_vec()))
        })
    });

    match result {
        Ok((status, resp_headers, resp_body)) => {
            let py_bytes = PyBytes::new(py, &resp_body);
            Ok((status, resp_headers, py_bytes.unbind()))
        }
        Err(e) => Err(PyRuntimeError::new_err(e)),
    }
}

#[pyfunction]
fn fast_json_dumps(py: Python<'_>, obj: Bound<'_, PyAny>) -> PyResult<Py<PyBytes>> {
    let value = python_to_json_value(&obj)?;
    let result =
        py.allow_threads(|| serde_json::to_vec(&value).map_err(|e| format!("JSON error: {e}")));
    match result {
        Ok(bytes) => Ok(PyBytes::new(py, &bytes).unbind()),
        Err(e) => Err(PyRuntimeError::new_err(e)),
    }
}

#[pyfunction]
fn fast_json_loads(py: Python<'_>, data: &[u8]) -> PyResult<PyObject> {
    let value: serde_json::Value =
        serde_json::from_slice(data).map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
    json_value_to_python(py, &value)
}

fn python_to_json_value(obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(serde_json::Value::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(serde_json::Value::Number(i.into()));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(serde_json::json!(f));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(serde_json::Value::String(s));
    }
    if let Ok(list) = obj.downcast::<PyList>() {
        let items: Result<Vec<_>, _> = list.iter().map(|item| python_to_json_value(&item)).collect();
        return Ok(serde_json::Value::Array(items?));
    }
    if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict.iter() {
            let key = k.extract::<String>()?;
            map.insert(key, python_to_json_value(&v)?);
        }
        return Ok(serde_json::Value::Object(map));
    }
    let s = obj.str()?.extract::<String>()?;
    Ok(serde_json::Value::String(s))
}

fn json_value_to_python(py: Python<'_>, value: &serde_json::Value) -> PyResult<PyObject> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok((*b).into_pyobject(py).unwrap().to_owned().into_any().unbind()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py).unwrap().to_owned().into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py).unwrap().to_owned().into_any().unbind())
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => Ok(s.clone().into_pyobject(py).unwrap().into_any().unbind()),
        serde_json::Value::Array(arr) => {
            let items: Vec<PyObject> = arr
                .iter()
                .map(|item| json_value_to_python(py, item))
                .collect::<PyResult<Vec<_>>>()?;
            let list = PyList::new(py, items)?;
            Ok(list.into_any().unbind())
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_value_to_python(py, v)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

#[pymodule]
fn litellm_pyext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(forward_request, m)?)?;
    m.add_function(wrap_pyfunction!(fast_json_dumps, m)?)?;
    m.add_function(wrap_pyfunction!(fast_json_loads, m)?)?;
    Ok(())
}
