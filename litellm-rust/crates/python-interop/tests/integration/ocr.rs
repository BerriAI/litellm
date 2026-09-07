use litellm_python_interop::{InvocationMode, InvocationOutcome, PreparedCall};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rstest::rstest;
use serial_test::serial;

use crate::support::python::{InitializedPython, initialized_python, item, scope};

fn prepare_pre_call(
    py: Python<'_>,
    logger: &Bound<'_, PyAny>,
    view: &Bound<'_, PyDict>,
) -> PyResult<PreparedCall> {
    let keywords = PyDict::new(py);
    keywords.set_item("input", "OCR document processing")?;
    keywords.set_item("api_key", py.None())?;
    keywords.set_item("additional_args", view)?;
    Ok(PreparedCall::new(
        InvocationMode::Direct,
        logger.getattr("pre_call")?.unbind(),
        PyTuple::empty(py).unbind(),
        Some(keywords.unbind()),
    ))
}

#[rstest]
#[ignore = "requires the repository Python environment and LiteLLM on PYTHONPATH"]
#[serial(python_interpreter)]
fn real_ocr_logging_preserves_execution_roots_and_continues_after_error(
    initialized_python: &InitializedPython,
) -> PyResult<()> {
    let _ = initialized_python;
    Python::attach(|py| {
        let globals = scope(
            py,
            c"
from litellm.integrations.custom_logger import CustomLogger

class Retain(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs):
        order.append('retain')
        self.view = kwargs['additional_args']
        self.headers = self.view['headers']
        self.body = self.view['complete_input_dict']
        self.snapshot = (self.headers['X-Trace'], self.body['document']['value'])
        return {'ignored_replacement': True}

class MutateThenFail(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs):
        order.append('mutate_then_fail')
        view = kwargs['additional_args']
        view['headers']['X-Trace'] = 'mutated'
        view['complete_input_dict']['document']['value'] = 'mutated'
        view['headers'] = {'X-Trace': 'replacement'}
        view['complete_input_dict'] = {'replacement': True}
        raise RuntimeError('expected callback failure')

class Observe(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs):
        order.append('observe')
        self.view = kwargs['additional_args']
        self.snapshot = (
            tuple(sorted(self.view['headers'].items())),
            self.view['complete_input_dict'].get('replacement'),
            'document' in self.view['complete_input_dict'],
        )

",
        )?;
        let order = PyList::empty(py);
        globals.set_item("order", &order)?;
        let first = item(&globals, "Retain").call0()?;
        let last = item(&globals, "Observe").call0()?;
        let document = PyDict::new(py);
        document.set_item("value", "original")?;
        let headers = PyDict::new(py);
        headers.set_item("X-Trace", "original")?;
        let body = PyDict::new(py);
        body.set_item("document", &document)?;
        body.set_item("alias", &document)?;
        let view = PyDict::new(py);
        view.set_item("headers", &headers)?;
        view.set_item("complete_input_dict", &body)?;
        view.set_item("api_base", "https://example.invalid/ocr")?;
        let keywords = PyDict::new(py);
        keywords.set_item("model", "test")?;
        keywords.set_item("messages", PyList::empty(py))?;
        keywords.set_item("stream", false)?;
        keywords.set_item("call_type", "ocr")?;
        keywords.set_item(
            "start_time",
            py.import("datetime")?
                .getattr("datetime")?
                .call_method0("now")?,
        )?;
        keywords.set_item("litellm_call_id", "retained-test")?;
        keywords.set_item("function_id", "retained-test")?;
        keywords.set_item(
            "dynamic_input_callbacks",
            PyList::new(
                py,
                [&first, &item(&globals, "MutateThenFail").call0()?, &last],
            )?,
        )?;
        let logger = py
            .import("litellm.litellm_core_utils.litellm_logging")?
            .getattr("Logging")?
            .call((), Some(&keywords))?;
        let invocation = prepare_pre_call(py, &logger, &view)?;
        match invocation.invoke(py)? {
            InvocationOutcome::Returned(value) => assert!(value.is_none(py)),
            InvocationOutcome::Awaitable(_) => {
                panic!("direct binding produced an awaitable outcome")
            }
        }
        drop(invocation);
        assert!(headers.is(first.getattr("headers")?));
        assert!(body.is(first.getattr("body")?));
        assert!(view.is(last.getattr("view")?));
        assert_eq!(item(&headers, "X-Trace").extract::<String>()?, "mutated");
        assert_eq!(
            order.extract::<Vec<String>>()?,
            ["retain", "mutate_then_fail", "observe"]
        );
        assert_eq!(
            first.getattr("snapshot")?.extract::<(String, String)>()?,
            ("original".to_owned(), "original".to_owned())
        );
        assert_eq!(
            last.getattr("snapshot")?
                .extract::<(Vec<(String, String)>, bool, bool)>()?,
            (
                vec![("X-Trace".to_owned(), "replacement".to_owned())],
                true,
                false
            )
        );
        assert!(first.getattr("view")?.is(last.getattr("view")?));
        assert!(first.getattr("body")?.get_item("document")?.is(&document));
        assert!(first.getattr("body")?.get_item("alias")?.is(&document));
        assert_eq!(item(&document, "value").extract::<String>()?, "mutated");
        assert_eq!(
            last.getattr("view")?
                .get_item("headers")?
                .get_item("X-Trace")?
                .extract::<String>()?,
            "replacement"
        );
        let replacement = PyDict::new(py);
        replacement.set_item("replacement", true)?;
        assert!(
            last.getattr("view")?
                .get_item("complete_input_dict")?
                .eq(replacement)?
        );
        document.set_item("value", "after invocation")?;
        assert_eq!(
            first
                .getattr("body")?
                .get_item("document")?
                .get_item("value")?
                .extract::<String>()?,
            "after invocation"
        );
        drop((headers, body, view));
        assert_eq!(
            first
                .getattr("headers")?
                .get_item("X-Trace")?
                .extract::<String>()?,
            "mutated"
        );
        Ok(())
    })
}
