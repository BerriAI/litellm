use litellm_host::hooks::CallHooks;
use litellm_host_python::{PythonOwned, PythonRuntime};
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyDict, PyTuple},
};

use super::host::ROUTE_HOST_MODULE;

pub(super) struct ModifyParams {
    call: Option<(Py<PyAny>, Py<PyTuple>)>,
}

impl ModifyParams {
    pub(super) fn new(args: &Bound<'_, PyTuple>, kwargs: &Bound<'_, PyDict>) -> PyResult<Self> {
        let model = match args.get_item(0) {
            Ok(model) => model,
            Err(_) => kwargs
                .get_item("model")?
                .unwrap_or_else(|| args.py().None().into_bound(args.py())),
        };
        Ok(Self {
            call: Some((model.unbind(), args.clone().unbind())),
        })
    }
}

impl CallHooks<PythonRuntime> for ModifyParams {
    fn arguments_prepared(&mut self, py: Python<'_>, arguments: &Py<PyDict>) -> PyResult<()> {
        let Some((model, args)) = &self.call else {
            return Ok(());
        };
        py.import(ROUTE_HOST_MODULE)?
            .getattr("modify_max_tokens")?
            .call1((model, args, arguments))?;
        Ok(())
    }
}

impl PythonOwned for ModifyParams {
    fn close(&mut self, _: Python<'_>) {
        self.call = None;
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some((model, args)) = &self.call {
            visit.call(model)?;
            visit.call(args)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use pyo3::{
        prelude::*,
        types::{PyDict, PyTuple},
    };
    use rstest::rstest;

    use super::ModifyParams;

    #[rstest]
    #[case::keyword_model(
        "()",
        "{'model': 'anthropic/claude-sonnet-4-5'}",
        "'anthropic/claude-sonnet-4-5'"
    )]
    #[case::first_positional_wins("(64, [])", "{'model': 'anthropic/claude-sonnet-4-5'}", "64")]
    #[case::no_model("()", "{}", "None")]
    fn reads_the_model_the_client_wrapper_reads(
        #[case] args: &str,
        #[case] kwargs: &str,
        #[case] expected: &str,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let eval = |source: &str| {
                py.eval(&std::ffi::CString::new(source).unwrap(), None, None)
                    .unwrap()
            };
            let args = eval(args).cast_into::<PyTuple>().unwrap();
            let kwargs = eval(kwargs).cast_into::<PyDict>().unwrap();
            let hook = ModifyParams::new(&args, &kwargs).unwrap();
            let (model, captured) = hook.call.as_ref().unwrap();
            assert!(model.bind(py).eq(eval(expected)).unwrap());
            assert!(captured.bind(py).is(&args));
        });
    }
}
