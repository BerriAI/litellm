use super::*;

#[pyfunction]
fn snapshot(py: Python<'_>, state: Py<MessagesState>) -> PyResult<Py<PyAny>> {
    let request = take_request(py, &state)?;
    let body: Value = serde_json::from_slice(request.body())
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    litellm_python_interop::to_py(py, &(body, request.headers()))
}

#[test]
fn callback_aliases_survive_snapshot_and_roots_are_collectible() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "messages_test").unwrap();
        module
            .add_function(wrap_pyfunction!(build_request, &module).unwrap())
            .unwrap();
        module
            .add_function(wrap_pyfunction!(pre_call, &module).unwrap())
            .unwrap();
        module
            .add_function(wrap_pyfunction!(snapshot, &module).unwrap())
            .unwrap();
        let globals = PyDict::new(py);
        globals.set_item("native", module).unwrap();
        py.run(
            c"
import gc
import weakref

class Opaque:
    pass

class Logger:
    def pre_call(self, **kwargs):
        view = kwargs['additional_args']
        self.body = view['complete_input_dict']
        self.headers = view['headers']
        assert self.body is body
        assert self.body['messages'][0] is message
        message['content'] = 'changed'
        self.headers['x-hook'] = 'changed'
        view['complete_input_dict'] = {'replacement': True}
        view['headers'] = {'replacement': 'true'}

message = {'role': 'user', 'content': 'original'}
body = {'model': 'model', 'messages': [message], 'max_tokens': 16}
opaque = Opaque()
logger = Logger()
arguments = dict(model='model', body=body, api_key='test',
                 custom_llm_provider='anthropic', opaque=opaque,
                 litellm_logging_obj=logger)
state = native.build_request(arguments, logger)
native.pre_call(state)
wire_body, wire_headers = native.snapshot(state)
assert wire_body['messages'][0]['content'] == 'original'
assert body['messages'][0]['content'] == 'changed'
assert dict(wire_headers)['x-hook'] == 'changed'
try:
    native.snapshot(state)
except RuntimeError:
    pass
else:
    raise AssertionError('request was sent twice')
arguments['cycle'] = state
logger.body['cycle'] = state
logger.headers['cycle'] = state
alive = weakref.ref(opaque)
del arguments, logger, opaque, body
gc.collect()
assert alive() is not None
del state
gc.collect()
assert alive() is None
",
            Some(&globals),
            Some(&globals),
        )
        .unwrap();
    });
}

#[test]
fn decode_state_rejects_invalid_timeouts_without_panicking() {
    Python::initialize();
    Python::attach(|py| {
        for timeout in [-1.0, 0.0, f64::NAN, f64::INFINITY, f64::MAX] {
            let arguments = PyDict::new(py);
            arguments.set_item("model", "model").unwrap();
            arguments.set_item("body", PyDict::new(py)).unwrap();
            arguments.set_item("timeout_seconds", timeout).unwrap();
            let error = match decode_options(&arguments.extract::<MessagesArguments<'_>>().unwrap())
            {
                Ok(_) => panic!("invalid timeout should fail normally"),
                Err(error) => error,
            };
            assert!(error.is_instance_of::<PyValueError>(py));
        }
    });
}
