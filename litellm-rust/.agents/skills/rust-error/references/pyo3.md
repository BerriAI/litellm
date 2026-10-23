# Python bridge

`crates/python-bridge` turns a Rust error into a Python exception in one function per route family: `errors::route_error_to_pyerr` for the inference routes and `routes/ocr/errors.rs::to_pyerr` for OCR. A `#[pyfunction]` returns `Err(PyErr)` and pyo3 raises it, so those two functions are the whole boundary. No route builds a `PyErr` from its own error inline

## Exception classes

| Class | Meaning | Payload |
|---|---|---|
| `RustUpstreamError` | the provider call was issued and failed | `args == (status, body)`, `status` is 0 when there was no HTTP response, plus a `headers` attribute of `(name, value)` pairs |
| `RustBridgeDeclined` | the route stopped before calling the provider, so the host may run its own path | `args == (message,)` |
| `ValueError` | the caller's input is at fault | the message |
| `RuntimeError` | the environment, the wire or the provider's answer is at fault | the message |

The two native classes come from `create_exception!` in `crates/python-bridge/src/errors.rs` and are exported with `#[pymodule_export]`, which is how the [pyo3 guide](https://github.com/PyO3/pyo3/blob/main/guide/src/exception.md) makes an exception importable. Their docstrings are the contract Python reads

Python decodes them once. `litellm/rust_bridge/failures.py` validates `args` and `headers` with a `TypeAdapter` and rebuilds the provider's `httpx.Response`, and `litellm/rust_bridge/runtime.py` catches `RustBridgeDeclined`. Nothing else in Python inspects a native exception, so a new fact goes into `args` or that one `headers` attribute, and the decoder learns to read it in the same change

## Why args, not a typed subclass

The bridge builds with pyo3's `abi3-py310` feature. Under abi3, `#[pyclass(extends = PyException)]` is only allowed on Python 3.12 and newer, so a class with typed fields is not available. Even where it is allowed, pyo3 passes an empty tuple to the native base's `tp_new`, so `.args` and `str()` would need manual overrides. The payload therefore travels in `args`, which a `create_exception!` class fills from `new_err`

```rust
RustUpstreamError::new_err((status, body.to_owned()))
```

OCR still sets `status_code`, `message` and `ocr_request_format_error` on the exception with `setattr` for Python parity. Do not add more attributes that way, nothing on the Python side types them

## Mapping order

```rust
pub(crate) fn route_error_to_pyerr(error: Error) -> PyErr {
    match error.upstream_response() {
        Some((status, body)) => RustUpstreamError::new_err((status, body.to_owned())),
        None => by_fault(error.is_request(), error.to_string()),
    }
}
```

The provider's answer passes through first. Everything else is split by `is_request()` into `ValueError` or `RuntimeError`, with `to_string()` as the message. A missing API key is a `RuntimeError` because `is_request()` answers false for it, and `missing_api_key_stays_a_runtime_error_while_other_auth_failures_are_value_errors` pins that

OCR puts Python parity arms in front of that order, each matching a top-level variant of its own public enum

| Arm | Exception |
|---|---|
| `Provider { status, body, headers }` | `RustUpstreamError` with the headers attached |
| `Http` with a `status()` | `RustUpstreamError`, which also carries a media download status, a known gap |
| `FileRead` whose cause is `NotFound` | `FileNotFoundError` naming the path |
| other `FileRead` | `OSError` with the io message |
| `RequestFormat` | `ValueError` flagged with `ocr_request_format_error` |

OCR also treats every auth failure as the caller's, so a missing key is a `ValueError` there. That is the one place the two bridges disagree

Use the standard class whose Python meaning matches, from `pyo3::exceptions`. `PyFileNotFoundError` for a missing file is right, a `RuntimeError` whose text says the file is missing is not. `import_exception!` exists for an exception defined in Python, none is used today because `failures.py` maps to LiteLLM's public exceptions through `litellm.exception_type`

## Declining

`RustBridgeDeclined` means no provider call was made and the host may retry on its own path. It is raised before dispatch, by the tokenizer, embeddings and token counter routes. Never raise it for a failure after the request went out, the host would repeat a billed call. `route_failures_are_terminal` in `errors.rs` pins that for the inference routes

## Rules

- Never format `Debug` of an error into a `PyErr`
- Never build a `PyErr` at a call site from a string. Return the crate error and let the mapping function classify it
- A `panic!` or `unwrap` inside a `#[pyfunction]` reaches Python as `pyo3_runtime.PanicException`, which is not an `Exception` subclass. Model the failure as a value instead
- When the mapping itself calls Python, as the OCR `setattr` does, a failure there is Python's own exception. Return it as is rather than wrapping it in a Rust message

## Testing

Build the Rust error the way production builds it, map it, and assert the contract Python sees. `Python::initialize()` once, then `Python::attach`

```rust
Python::initialize();
Python::attach(|py| {
    let failure = route_error_to_pyerr(litellm_http::Error::upstream(429, "slow down".into()).into());
    assert!(failure.is_instance_of::<RustUpstreamError>(py));
    let args: (u16, String) = failure.value(py).getattr("args").unwrap().extract().unwrap();
    assert_eq!(args, (429, "slow down".into()));
});
```

One `#[case]` per fault class, asserting `is_instance_of` for the class and `extract` for the payload. The Python side of the contract is tested in `tests/unit/rust_bridge`, through `failures.py`
