use super::*;

#[pyfunction]
fn snapshot(py: Python<'_>, state: Py<ChatCompletionsState>) -> PyResult<Py<PyAny>> {
    let request = take_request(py, &state)?;
    let (body, headers) = match &request.readback {
        ChatPreCallReadback::StructuredAtSend { body, headers } => {
            (body.as_bytes(), headers.as_slice())
        }
        ChatPreCallReadback::CapturedAtBuild { headers } => (
            request.request.body.authorized().unwrap().body(),
            headers.as_slice(),
        ),
    };
    to_py(
        py,
        &(serde_json::from_slice::<Value>(body).unwrap(), headers),
    )
}

#[test]
#[ignore = "requires requests on PYTHONPATH"]
fn bedrock_callback_headers_work_without_botocore() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "chat_headers_test").unwrap();
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
import sys
from collections.abc import MutableMapping
from unittest.mock import patch

class Logger:
    def pre_call(self, **kwargs):
        view = kwargs['additional_args']
        headers = view['headers']
        assert isinstance(headers, MutableMapping)
        assert headers is not original_headers
        assert headers['X-ORIGINAL'] == 'original'
        assert headers.get('CONTENT-TYPE') == 'application/json'
        assert 'AUTHORIZATION' in headers
        assert headers['Authorization'].startswith(expected_auth)
        headers.update({'x-original': 'edited', 'X-Added': 'added'})
        assert headers.setdefault('X-ORIGINAL', 'ignored') == 'edited'
        assert sum(key.lower() == 'x-original' for key in headers) == 1
        assert headers.pop('x-ADDED') == 'added'
        del headers['X-DELETE']
        copied = headers.copy()
        copied['x-original'] = 'copy edit'
        assert headers['X-Original'] == 'edited'
        assert isinstance(view['complete_input_dict'], str)
        view['headers'] = {'replacement': 'ignored'}
        view['complete_input_dict'] = 'replacement'

with patch.dict(sys.modules, {'botocore': None, 'botocore.awsrequest': None}):
    for api_key, expected_auth in [('test-token', 'Bearer test-token'), ('', 'AWS4-HMAC-SHA256 ')]:
        original_headers = {'X-Original': 'original', 'x-delete': 'delete'}
        arguments = dict(
            model='anthropic.claude-opus-5',
            messages=[{'role': 'user', 'content': 'original'}],
            optional_params={'maxTokens': 16, 'aws_region_name': 'us-west-2',
                             'aws_access_key_id': 'test-access-key',
                             'aws_secret_access_key': 'test-secret-key'},
            extra_headers=original_headers, api_key=api_key,
            custom_llm_provider='bedrock', api_base=None,
        )
        state = native.build_request(arguments, Logger())
        native.pre_call(state)
        wire_body, wire_headers = native.snapshot(state)
        assert wire_body['messages'][0]['content'][0]['text'] == 'original'
        headers = {key.lower(): value for key, value in wire_headers}
        assert headers['x-original'] == 'edited'
        assert 'x-delete' not in headers
        assert 'x-added' not in headers
        assert 'replacement' not in headers
        assert headers['authorization'].startswith(expected_auth)
        assert original_headers == {'X-Original': 'original', 'x-delete': 'delete'}
",
            Some(&globals),
            Some(&globals),
        )
        .unwrap();
    });
}

#[test]
#[ignore = "requires the Python SDK and its dependencies on PYTHONPATH"]
fn callback_roots_survive_rebinding_and_cycles_are_collected() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "chat_test").unwrap();
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
        assert kwargs['input'] is messages
        assert self.body['messages'] is not messages
        assert self.body['stop_sequences'] is stops
        assert self.headers is headers
        messages[0]['content'] = 'edited'
        self.body['messages'][0]['content'][0]['text'] = 'body edit'
        stops.append('second')
        self.headers['x-hook'] = 'edited'
        view['complete_input_dict'] = {'replacement': True}
        view['headers'] = {'replacement': 'true'}

messages = [{'role': 'user', 'content': 'original'}]
stops = ['first']
headers = {}
opaque = Opaque()
logger = Logger()
arguments = dict(model='claude-opus-5', messages=messages,
                 optional_params={'max_tokens': 16, 'stop_sequences': stops},
                 extra_headers=headers, api_key='test',
                 custom_llm_provider='anthropic', opaque=opaque,
                 litellm_logging_obj=logger)
state = native.build_request(arguments, logger)
native.pre_call(state)
wire_body, wire_headers = native.snapshot(state)
assert wire_body['messages'][0]['content'][0]['text'] == 'body edit'
assert wire_body['stop_sequences'] == ['first', 'second']
assert dict(wire_headers)['x-hook'] == 'edited'
arguments['cycle'] = state
logger.body['cycle'] = state
headers['cycle'] = state
alive = weakref.ref(opaque)
del arguments, logger, opaque, headers
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
