use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn native_module(py: Python<'_>) -> Bound<'_, PyModule> {
    pyo3::wrap_pymodule!(_native::_native)(py).into_bound(py)
}

#[test]
fn module_registration_preserves_the_public_surface() {
    Python::initialize();
    Python::attach(|py| {
        let module = native_module(py);

        let expected = [
            "RustBridgeDeclined",
            "RustUpstreamError",
            "RustBridgeDriverError",
            "ocr",
            "aocr",
            "transcription",
            "atranscription",
            "messages",
            "amessages",
            "chat_completions",
            "achat_completions",
            "chat_completions_decline",
            "responses_websocket",
        ];

        let public_names: Vec<String> = module
            .dict()
            .keys()
            .extract::<Vec<String>>()
            .expect("module names should be strings")
            .into_iter()
            .filter(|name| !name.starts_with('_'))
            .collect();
        assert_eq!(public_names, expected);

        #[cfg(not(feature = "trace-parity"))]
        assert!(!module.hasattr("_trace").expect("module lookup should work"));

        #[cfg(feature = "trace-parity")]
        {
            let trace = module
                .getattr("_trace")
                .expect("trace build should expose its diagnostic namespace");
            let trace_names: Vec<String> = trace
                .cast::<PyModule>()
                .expect("trace namespace should be a module")
                .dict()
                .keys()
                .extract::<Vec<String>>()
                .expect("trace names should be strings")
                .into_iter()
                .filter(|name| !name.starts_with("__"))
                .collect();
            assert_eq!(
                trace_names,
                [
                    "ocr",
                    "aocr",
                    "transcription",
                    "atranscription",
                    "messages",
                    "amessages",
                    "chat_completions",
                    "achat_completions",
                    "chat_completions_decline",
                    "gateway_messages",
                ]
            );
        }
    });
}

#[test]
fn sync_and_async_route_signatures_match_the_python_contract() {
    Python::initialize();
    Python::attach(|py| {
        let module = native_module(py);
        let routes = [
            ("ocr", "aocr", "(arguments)"),
            (
                "transcription",
                "atranscription",
                "(model, audio, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None)",
            ),
            ("messages", "amessages", "(arguments)"),
            ("chat_completions", "achat_completions", "(arguments)"),
        ];

        for (sync_name, async_name, expected) in routes {
            let sync_signature: String = module
                .getattr(sync_name)
                .and_then(|function| function.getattr("__text_signature__"))
                .and_then(|signature| signature.extract())
                .expect("sync signature should be available");
            let async_signature: String = module
                .getattr(async_name)
                .and_then(|function| function.getattr("__text_signature__"))
                .and_then(|signature| signature.extract())
                .expect("async signature should be available");

            assert_eq!(sync_signature, expected);
            assert_eq!(async_signature, expected);
        }
    });
}

#[test]
fn sync_and_async_routes_apply_the_same_input_validation() {
    Python::initialize();
    Python::attach(|py| {
        let module = native_module(py);

        let invalid_messages = PyDict::new(py);
        let invalid_chat_arguments = PyDict::new(py);
        invalid_chat_arguments
            .set_item("model", "model")
            .expect("arguments should accept model");
        invalid_chat_arguments
            .set_item("messages", &invalid_messages)
            .expect("arguments should accept messages");
        let sync_chat_error = module
            .getattr("chat_completions")
            .and_then(|function| function.call1((&invalid_chat_arguments,)))
            .expect_err("sync chat should reject a non-list messages value");
        let async_chat_error = module
            .getattr("achat_completions")
            .and_then(|function| function.call1((&invalid_chat_arguments,)))
            .expect_err("async chat should reject a non-list messages value");

        assert_eq!(
            sync_chat_error.to_string(),
            "TypeError: messages must be a list"
        );
        assert_eq!(async_chat_error.to_string(), sync_chat_error.to_string());

        let invalid_body = PyList::empty(py);
        let invalid_arguments = PyDict::new(py);
        invalid_arguments
            .set_item("model", "model")
            .expect("arguments should accept model");
        invalid_arguments
            .set_item("body", &invalid_body)
            .expect("arguments should accept body");
        let sync_messages_error = module
            .getattr("messages")
            .and_then(|function| function.call1((&invalid_arguments,)))
            .expect_err("sync Messages should reject a non-dict body");
        let async_messages_error = module
            .getattr("amessages")
            .and_then(|function| function.call1((&invalid_arguments,)))
            .expect_err("async Messages should reject a non-dict body");

        assert_eq!(
            sync_messages_error.to_string(),
            "TypeError: body must be a dict"
        );
        assert_eq!(
            async_messages_error.to_string(),
            sync_messages_error.to_string()
        );

        let invalid_headers = PyList::empty(py);
        let kwargs = PyDict::new(py);
        kwargs
            .set_item("extra_headers", &invalid_headers)
            .expect("kwargs should accept extra_headers");
        let document = PyDict::new(py);
        document.set_item("data", "AQI=").unwrap();
        document.set_item("format", "wav").unwrap();

        {
            let (sync_name, async_name) = ("transcription", "atranscription");
            let sync_error = module
                .getattr(sync_name)
                .and_then(|function| function.call(("model", &document), Some(&kwargs)))
                .expect_err("sync route should reject non-dict extra_headers");
            let async_error = module
                .getattr(async_name)
                .and_then(|function| function.call(("model", &document), Some(&kwargs)))
                .expect_err("async route should reject non-dict extra_headers");

            assert_eq!(
                sync_error.to_string(),
                "TypeError: extra_headers must be a dict"
            );
            assert_eq!(async_error.to_string(), sync_error.to_string());
        }
    });
}

#[test]
fn route_input_validation_preserves_left_to_right_order() {
    Python::initialize();
    Python::attach(|py| {
        let module = native_module(py);
        let invalid = PyList::empty(py);

        let chat_arguments = PyDict::new(py);
        chat_arguments
            .set_item("model", "model")
            .expect("arguments should accept model");
        chat_arguments
            .set_item("optional_params", &invalid)
            .expect("arguments should accept optional_params");
        chat_arguments
            .set_item("extra_headers", &invalid)
            .expect("arguments should accept extra_headers");
        let invalid_messages = PyDict::new(py);
        chat_arguments
            .set_item("messages", &invalid_messages)
            .expect("arguments should accept messages");
        let error = module
            .getattr("chat_completions")
            .and_then(|function| function.call1((&chat_arguments,)))
            .expect_err("messages should be validated first");
        assert_eq!(error.to_string(), "TypeError: messages must be a list");

        let headers_kwargs = PyDict::new(py);
        headers_kwargs
            .set_item("extra_headers", &invalid)
            .expect("kwargs should accept extra_headers");
        let invalid_body = PyList::empty(py);
        let invalid_arguments = PyDict::new(py);
        invalid_arguments
            .set_item("model", "model")
            .expect("arguments should accept model");
        invalid_arguments
            .set_item("body", &invalid_body)
            .expect("arguments should accept body");
        invalid_arguments
            .set_item("extra_headers", &invalid)
            .expect("arguments should accept extra_headers");
        let error = module
            .getattr("messages")
            .and_then(|function| function.call1((&invalid_arguments,)))
            .expect_err("body should be validated before headers");
        assert_eq!(error.to_string(), "TypeError: body must be a dict");

        let invalid_payload =
            PyModule::new(py, "invalid_payload").expect("invalid payload should be created");
        for name in ["ocr", "transcription"] {
            let error = module
                .getattr(name)
                .and_then(|function| {
                    function.call(("model", &invalid_payload), Some(&headers_kwargs))
                })
                .expect_err("payload should be validated before headers");
            assert!(!error.to_string().contains("extra_headers"));
        }
    });
}

#[test]
#[ignore = "requires the Python SDK and its dependencies on PYTHONPATH"]
fn callback_decline_is_terminal_and_identity_is_reused() {
    Python::initialize();
    Python::attach(|py| {
        let module = native_module(py);
        let globals = PyDict::new(py);
        globals.set_item("native", module).unwrap();
        py.run(
                c"
import asyncio
import contextvars
import threading
from datetime import datetime

marker = contextvars.ContextVar('terminal_marker')

class Logger:
    litellm_call_id = 'supplied-call'
    litellm_trace_id = 'supplied-trace'

    def update_from_kwargs(self, **values):
        assert values['kwargs']['litellm_call_id'] == self.litellm_call_id
        assert values['kwargs']['litellm_trace_id'] == self.litellm_trace_id
        assert threading.get_ident() == self.thread
        marker.set('update')
        raise self.original

    def failure_handler(self, error, trace, start, end):
        assert error is self.original
        assert marker.get() == 'update'
        assert start <= end <= datetime.now()
        self.end = end
        self.calls.append('failure')

    async def async_failure_handler(self, error, trace, start, end):
        await asyncio.sleep(0)
        assert asyncio.current_task() is self.task
        assert marker.get() == 'update'
        assert error is self.original
        assert end is self.end
        self.calls.append('async_failure')

    def _restore_correlation_context(self):
        self.calls.append('restore')

async def exercise():
    for asynchronous in (False, True):
        logger = Logger()
        logger.thread = threading.get_ident()
        logger.task = asyncio.current_task()
        logger.calls = []
        logger.original = NotImplementedError('callback declined, not admission')
        arguments = dict(model='mistral/mistral-ocr-latest', api_key='test-key', timeout=1.0,
                         document={'type': 'document_url', 'document_url': 'https://example.test/doc.pdf'},
                         litellm_logging_obj=logger)
        try:
            if asynchronous:
                await native.aocr(arguments)
            else:
                native.ocr(arguments)
        except NotImplementedError as error:
            assert error is logger.original
        else:
            raise AssertionError('callback exception was lost')
        assert logger.calls == (['failure', 'async_failure', 'restore'] if asynchronous else ['failure', 'restore'])
        assert arguments['litellm_call_id'] == 'supplied-call'
        assert arguments['litellm_trace_id'] == 'supplied-trace'

asyncio.run(exercise())
",
                Some(&globals),
                Some(&globals),
            ).unwrap();
    });
}

#[test]
#[ignore = "requires the Python SDK and its dependencies on PYTHONPATH"]
fn async_callbacks_are_inline_and_unsupported_requests_never_call_them() {
    Python::initialize();
    Python::attach(|py| {
        let module = native_module(py);
        let globals = PyDict::new(py);
        globals.set_item("native", module).unwrap();
        py.run(
            cr"
import asyncio
import contextvars
import gc
import json
import threading
import weakref

marker = contextvars.ContextVar('ocr_marker')

class Opaque:
    pass

class Logger:
    def __init__(self):
        self.calls = []

    def failure_handler(self, error, trace, start, end):
        assert asyncio.current_task() is caller
        assert marker.get() == 'pre'
        self.error = error

    async def async_failure_handler(self, error, trace, start, end):
        await asyncio.sleep(0)
        assert asyncio.current_task() is caller
        assert error is self.error

    def update_from_kwargs(self, **values):
        assert asyncio.current_task() is caller
        assert threading.get_ident() == caller_thread
        assert marker.get() == 'caller'
        assert values['kwargs'] is not arguments
        assert values['kwargs']['opaque'] is arguments['opaque']
        self.calls.append('update')
        marker.set('updated')

    def pre_call(self, **values):
        assert asyncio.current_task() is caller
        assert threading.get_ident() == caller_thread
        assert marker.get() == 'updated'
        self.calls.append('pre')
        values['additional_args']['complete_input_dict']['pages'].append(3)
        marker.set('pre')

async def exercise():
    global arguments, caller, caller_thread
    caller = asyncio.current_task()
    caller_thread = threading.get_ident()
    marker.set('caller')
    logger = Logger()
    document = {'type': 'document_url', 'document_url': 'https://example.test/test.pdf'}
    arguments = dict(model='mistral/mistral-ocr-latest', document=document,
                     api_key='test-key', pages=[0], opaque=Opaque(),
                     litellm_logging_obj=logger)
    alive = weakref.ref(arguments['opaque'])
    received = asyncio.Event()
    errors = []

    async def respond(reader, writer):
        try:
            header = await reader.readuntil(b'\r\n\r\n')
            length = next(int(line.split(b':', 1)[1]) for line in header.split(b'\r\n')
                          if line.lower().startswith(b'content-length:'))
            body = json.loads(await reader.readexactly(length))
            assert body['pages'] == [0, 3]
            assert 'opaque' not in body
            gc.collect()
            assert alive() is not None
            assert logger.calls == ['update', 'pre']
            globals().pop('arguments')
            gc.collect()
            assert alive() is not None
        except BaseException as error:
            errors.append(error)
        finally:
            writer.write(b'HTTP/1.1 200 OK\r\nContent-Length: 1\r\nConnection: close\r\n\r\nx')
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            received.set()

    server = await asyncio.start_server(respond, '127.0.0.1', 0)
    async with server:
        port = server.sockets[0].getsockname()[1]
        arguments['api_base'] = f'http://127.0.0.1:{port}'
        arguments['timeout'] = 5.0
        try:
            await native.aocr(arguments)
        except RuntimeError as error:
            assert error is logger.error
        else:
            raise AssertionError('expected upstream error')
        await asyncio.wait_for(received.wait(), 5)
    assert not errors, errors
    assert marker.get() == 'pre'
    assert logger.calls == ['update', 'pre']

    for model, doc in [
        ('azure_ai/doc-intelligence/prebuilt-read', document),
        ('vertex_ai/ocr', document),
        ('mistral/mistral-ocr-latest', {'type': 'file', 'file': Opaque()}),
    ]:
        logger.calls.clear()
        unsupported = dict(model=model, document=doc, api_key='test-key', timeout=5.0,
                           litellm_logging_obj=logger)
        for asynchronous in (False, True):
            try:
                if asynchronous:
                    await native.aocr(unsupported)
                else:
                    native.ocr(unsupported)
            except NotImplementedError:
                pass
            else:
                raise AssertionError('expected strict unsupported error')
            assert logger.calls == []

asyncio.run(exercise())
",
            Some(&globals),
            Some(&globals),
        )
        .unwrap();
    });
}
