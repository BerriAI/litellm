use super::*;
use litellm_core::integrations::custom_logger::CallbackTiming;
use litellm_core::integrations::types::Usage;
use litellm_core::lifecycle::{RouteProjection, TerminalClassification};
use serde_json::json;

#[test]
#[ignore = "requires the Python SDK and its dependencies on PYTHONPATH"]
fn structured_errors_use_public_sdk_exceptions() {
    Python::initialize();
    Python::attach(|py| {
        let exceptions = py.import("litellm.exceptions").unwrap();
        for (status, class) in [
            (400, "BadRequestError"),
            (401, "AuthenticationError"),
            (403, "PermissionDeniedError"),
            (404, "NotFoundError"),
            (422, "UnprocessableEntityError"),
            (429, "RateLimitError"),
            (500, "InternalServerError"),
            (502, "BadGatewayError"),
            (503, "ServiceUnavailableError"),
            (504, "APIError"),
        ] {
            let error = ocr_error_to_pyerr(
                py,
                CoreError::Http {
                    status,
                    body: "private upstream content".into(),
                },
                "mistral-ocr-latest",
                "mistral",
            );
            let value = error.value(py);
            assert!(
                value
                    .is_instance(&exceptions.getattr(class).unwrap())
                    .unwrap(),
                "HTTP {status}: expected {class}, got {error}"
            );
            assert_eq!(
                value
                    .getattr("status_code")
                    .unwrap()
                    .extract::<u16>()
                    .unwrap(),
                status
            );
            assert_eq!(
                value.getattr("model").unwrap().extract::<String>().unwrap(),
                "mistral-ocr-latest"
            );
            assert_eq!(
                value
                    .getattr("llm_provider")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "mistral"
            );
            assert!(!error.to_string().contains("private upstream content"));
        }
        let error = ocr_error_to_pyerr(
            py,
            CoreError::Auth("private credential".into()),
            "model",
            "mistral",
        );
        assert!(
            error
                .value(py)
                .is_instance(&exceptions.getattr("AuthenticationError").unwrap())
                .unwrap()
        );
        assert!(!error.to_string().contains("private credential"));
    });
}

#[test]
fn unstructured_transport_errors_are_not_guessed_from_strings() {
    Python::initialize();
    Python::attach(|py| {
        for error in [
            CoreError::Network("timeout secret".into()),
            CoreError::Connect("401 secret".into()),
            CoreError::InvalidResponse("429 secret".into()),
        ] {
            let error = ocr_error_to_pyerr(py, error, "model", "mistral");
            assert!(error.is_instance_of::<PyRuntimeError>(py));
            assert!(!error.to_string().contains("secret"));
        }
    });
}

#[test]
fn finish_does_not_modify_the_input_dictionary() {
    Python::initialize();
    Python::attach(|py| {
        let fields = PyDict::new(py);
        fields
            .set_item("provider_native_response", "native")
            .unwrap();
        fields.set_item("model", "model").unwrap();

        let _ = finish(py, fields.clone().unbind());

        assert_eq!(
            fields
                .get_item("provider_native_response")
                .unwrap()
                .unwrap()
                .extract::<String>()
                .unwrap(),
            "native"
        );
    });
}

#[test]
fn terminal_record_exports_core_timing() {
    Python::initialize();
    Python::attach(|py| {
        let state = Py::new(
            py,
            OcrState {
                callback: RetainedCallback::cleared(),
                endpoint: None,
                asynchronous: false,
                terminal: Some(TerminalRecord {
                    call_id: "call-1".into(),
                    trace_id: None,
                    attempt: 1,
                    call_type: "ocr".into(),
                    model: "model".into(),
                    provider: "mistral".into(),
                    timing: CallbackTiming::new(10.25, 12.5),
                    usage: Usage::default(),
                    cost_inputs: Default::default(),
                    classification: TerminalClassification::Success,
                    projection: RouteProjection::Ocr {
                        value: json!({"pages": []}),
                    },
                }),
            },
        )
        .unwrap();

        let record = terminal_record(py, state).unwrap();
        let timing = record.bind(py).get_item("timing").unwrap();
        assert_eq!(
            timing
                .get_item("start_time")
                .unwrap()
                .extract::<f64>()
                .unwrap(),
            10.25
        );
        assert_eq!(
            timing
                .get_item("end_time")
                .unwrap()
                .extract::<f64>()
                .unwrap(),
            12.5
        );
    });
}

#[test]
#[ignore = "requires the Python SDK and its dependencies on PYTHONPATH"]
fn native_send_owns_state_without_the_python_driver() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "ocr_test").unwrap();
        module
            .add_function(wrap_pyfunction!(build_request, &module).unwrap())
            .unwrap();
        module
            .add_function(wrap_pyfunction!(pre_call, &module).unwrap())
            .unwrap();
        module
            .add_function(wrap_pyfunction!(send, &module).unwrap())
            .unwrap();
        let globals = PyDict::new(py);
        globals.set_item("native", module).unwrap();
        py.run(
            cr"
import asyncio
import gc
import weakref

class Logger:
    def update_from_kwargs(self, **values):
        pass
    def pre_call(self, **values):
        pass

async def exercise():
    received = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()

    async def respond(reader, writer):
        await reader.readuntil(b'\r\n\r\n')
        received.set()
        await release.wait()
        writer.close()
        await writer.wait_closed()
        closed.set()

    server = await asyncio.start_server(respond, '127.0.0.1', 0)
    async with server:
        port = server.sockets[0].getsockname()[1]
        logger = Logger()
        alive = weakref.ref(logger)
        state = native.build_request(dict(
            model='mistral/mistral-ocr-latest', api_key='test-key', timeout=5.0,
            api_base=f'http://127.0.0.1:{port}', litellm_logging_obj=logger,
            document={'type': 'document_url', 'document_url': 'https://example.test/doc.pdf'},
        ), logger, True)
        pending = native.send(state)
        del state, logger
        try:
            await asyncio.wait_for(received.wait(), 5)
            gc.collect()
            assert alive() is not None
            pending.cancel()
            try:
                await pending
            except asyncio.CancelledError:
                pass
            for _ in range(500):
                await asyncio.sleep(0.01)
                gc.collect()
                if alive() is None:
                    break
            assert alive() is None
        finally:
            release.set()
            await asyncio.wait_for(closed.wait(), 5)

asyncio.run(exercise())
",
            Some(&globals),
            Some(&globals),
        )
        .unwrap();
    });
}

#[pyfunction]
fn snapshot(py: Python<'_>, state: Py<OcrState>) -> PyResult<Py<PyAny>> {
    let state = state.borrow(py);
    let roots = state.callback.roots(Route::Ocr)?;
    let headers = header_pairs(roots.headers(py).cast::<PyDict>()?)?;
    let body: serde_json::Value = from_py(&roots.body(py))?;
    to_py(py, &(headers, body))
}

#[test]
#[ignore = "requires the Python SDK and its dependencies on PYTHONPATH"]
fn retains_identity_independent_wire_roots_and_collects_cycles() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "ocr_test").unwrap();
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

class Timeout:
    read = 5.0

class Logger:
    def update_from_kwargs(self, **values):
        assert values['kwargs'] is arguments
        assert values['kwargs']['metadata'] is metadata
        assert values['kwargs']['opaque'] is opaque
        assert values['optional_params']['pages'] is pages
        self.calls = ['update']

    def pre_call(self, **values):
        self.calls.append('pre')
        view = values['additional_args']
        self.body = view['complete_input_dict']
        self.headers = view['headers']
        assert self.body['document'] is document
        assert self.body['pages'] is pages
        document['document_url'] = 'https://example.test/changed.pdf'
        pages.append(2)
        self.headers['x-hook'] = 'changed'
        view['complete_input_dict'] = {'replacement': True}
        view['headers'] = {'replacement': 'true'}

document = {'type': 'document_url', 'document_url': 'https://example.test/test.pdf'}
pages = [0]
metadata = {'nested': []}
opaque = Opaque()
logger = Logger()
arguments = dict(model='mistral/mistral-ocr-latest', document=document,
                 api_key='test-key', pages=pages, metadata=metadata,
                 opaque=opaque, litellm_logging_obj=logger, timeout=Timeout())
state = native.build_request(arguments, logger, False)
native.pre_call(state)
assert logger.calls == ['update', 'pre']
roots = gc.get_referents(state)
assert any(root is arguments for root in roots)
assert any(root is logger for root in roots)
assert any(root is logger.body for root in roots)
assert any(root is logger.headers for root in roots)
headers, body = native.snapshot(state)
assert body['document']['document_url'] == 'https://example.test/changed.pdf'
assert body['pages'] == [0, 2]
assert dict(headers)['x-hook'] == 'changed'
assert 'replacement' not in body and 'replacement' not in dict(headers)

arguments['cycle'] = state
logger.cycle = state
logger.body['cycle'] = state
logger.headers['cycle'] = state
alive = weakref.ref(opaque)
del roots, arguments, logger, opaque
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
