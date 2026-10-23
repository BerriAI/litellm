# Python bridge

`crates/python-bridge` turns a Rust error into a Python exception through one mapping function per error type. A `#[pyfunction]` returns `Err(PyErr)` and pyo3 raises it, so those functions are the whole boundary

| Error | Mapping |
|---|---|
| `litellm_inference::Error` from a route | `errors::route_error_to_pyerr` |
| the OCR error | `routes/ocr/errors.rs::to_pyerr` |
| token counter, cache, spend, secrets, argument projection | one `From<Error> for PyErr` or `*_to_pyerr` next to the code that registers the functions |

A route host never builds a `PyErr` from a route error itself. It calls the mapping, and the mapping is the only place that knows the exception classes and their attributes

## Exception classes

| Class | Meaning | Payload |
|---|---|---|
| `RustUpstreamError` | the provider call was issued and the provider answered with a failure | `args == (status, body)` and a `headers` attribute of `(name, value)` pairs, always set |
| `RustBridgeDeclined` | the route stopped before calling the provider, so the host may run its own path | `args == (message,)` |
| a standard class from `pyo3::exceptions` | the caller's input (`ValueError`, `TypeError`), a missing file (`FileNotFoundError`), anything else (`RuntimeError`) | the rendered chain |

The two native classes come from `create_exception!` in `crates/python-bridge/src/errors.rs` and are exported with `#[pymodule_export]`, which is how the [pyo3 guide](https://github.com/PyO3/pyo3/blob/main/guide/src/exception.md) makes an exception importable. Their docstrings are the contract Python reads. `errors::upstream_error(py, status, body, headers)` is the one constructor of `RustUpstreamError`, so the payload cannot differ between routes

Python decodes a `RustUpstreamError` in `litellm/rust_bridge/failures.py`, which validates `args` and `headers` with a `TypeAdapter` and rebuilds the provider's `httpx.Response`. `litellm/rust_bridge/runtime.py` catches `RustBridgeDeclined` and reads its reason. `_raise_upstream` in `runtime.py` still reads `args` a second time by hand for the `attempt` path, which is the one decode to fold into `failures.py`. A new fact goes into `args` or `headers`, and the decoder learns to read it in the same change

Two routes also flag a request fault with a boolean attribute, read by their Python route module: `ocr_request_format_error` in `host/routes/ocr.py` and `messages_request_error` in `host/routes/messages.py`. OCR additionally sets `status_code` and `message` for Python parity, and `native_settings_error`, which only the OCR host reads back to tell a settings failure from the caller's exception. Do not add another attribute, nothing on the Python side types them

## Why args, not a typed subclass

The bridge builds with pyo3's `abi3-py310` feature. Under abi3, `#[pyclass(extends = PyException)]` is only allowed on Python 3.12 and newer, so a class with typed fields is not available. Even where it is allowed, pyo3 passes an empty tuple to the native base's `tp_new`, so `.args` and `str()` would need manual overrides. The payload therefore travels in `args`, which a `create_exception!` class fills from `new_err`

## Mapping order

The provider's answer passes through first as `RustUpstreamError`. Everything else is split by `is_request()` into `ValueError` or `RuntimeError`, with the message rendered by `litellm_error::Report`. A missing API key is a `RuntimeError` because `is_request()` answers false for it, and `missing_api_key_stays_a_runtime_error_while_other_auth_failures_are_value_errors` pins that

OCR puts Python parity arms in front of that order, each matching a top-level variant of its own public enum

| Arm | Exception |
|---|---|
| the provider answered | `RustUpstreamError` with the provider's headers |
| `RequestFormat` | `ValueError` flagged with `ocr_request_format_error` |
| `FileRead` whose cause is `NotFound` | `FileNotFoundError` naming the path |
| other `FileRead` | `OSError` with the io message |

Both routes ask `litellm_auth::Error::is_request()` through their own `is_request()`, so a missing key or API base is a `RuntimeError` on both, and a missing setting the caller can pass, such as a region, is a `ValueError`

Use the standard class whose Python meaning matches. `PyFileNotFoundError` for a missing file is right, a `RuntimeError` whose text says the file is missing is not. `import_exception!` exists for an exception defined in Python, none is used today because `failures.py` maps to LiteLLM's public exceptions through `litellm.exception_type`

## Declining

`RustBridgeDeclined` means no provider call was made and the host may retry on its own path. It is raised before dispatch, by the tokenizer, embeddings and token counter routes. Never raise it for a failure after the request went out, the host would repeat a billed call. `route_failures_are_terminal` in `errors.rs` pins that for the inference routes

## Rules

- Never format `Debug` of an error into a `PyErr`
- Never build a `PyErr` from a string at a call site. A failed argument check is a variant of a typed error, as `ProjectionError` in `coercion.rs` is, mapped once
- A Python exception raised inside the call stays that exception. Return it as is, never wrap it in a Rust message or stringify it into a new `ValueError`
- A `panic!`, `unwrap` or `expect` inside a `#[pyfunction]` or `#[getter]` reaches Python as `pyo3_runtime.PanicException`, which is not an `Exception` subclass. Model the failure as a value instead
- When the mapping itself calls Python, as `setattr` does, propagate its failure with `?` rather than discarding it with `.ok()`

## Testing

Build the Rust error the way production builds it, map it, and assert the contract Python sees. `Python::initialize()` once, then `Python::attach`

```rust
Python::initialize();
Python::attach(|py| {
    let failure = route_error_to_pyerr(py, litellm_http::Error::upstream(429, "slow down".into()).into());
    assert!(failure.is_instance_of::<RustUpstreamError>(py));
    let args: (u16, String) = failure.value(py).getattr("args").unwrap().extract().unwrap();
    assert_eq!(args, (429, "slow down".into()));
});
```

One `#[case]` per fault class, asserting `is_instance_of` for the class and `extract` for the payload. The Python side of the contract is tested in `tests/unit/rust_bridge`, through `failures.py`
