litellm-config is the config-loading boundary. It returns resolved core deployment data and optionally delegates loading to Python.

- Depends on `litellm-core`; may use PyO3 behind the `python` feature
- `load_model_list(path)` calls Python's `litellm.proxy.read_model_list` and parses the JSON into `core::router::Deployment`
- Load-time/startup only, never on a request or stream path
- Returns typed `Error` variants (`PythonLoading`, `Serialization`, `ModelListParsing`)
