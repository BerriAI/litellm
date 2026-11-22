use litellm_host::hooks::CallHooks;
use litellm_host_python::{PythonOwned, PythonRuntime};
use pyo3::{
    exceptions::PyException,
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
        let clamped = py
            .import(ROUTE_HOST_MODULE)
            .and_then(|module| module.getattr("modify_max_tokens"))
            .and_then(|clamp| clamp.call1((model, args, arguments)));
        match clamped {
            Err(error) if !error.is_instance_of::<PyException>(py) => Err(error),
            _ => Ok(()),
        }
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

    use super::{ModifyParams, ROUTE_HOST_MODULE};
    use litellm_host::hooks::CallHooks;

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

    #[rstest]
    #[case::exception_leaves_max_tokens("ValueError", true)]
    #[case::base_exception_propagates("KeyboardInterrupt", false)]
    fn a_failing_clamp_behaves_like_the_client_wrappers_swallowed_try(
        #[case] raised: &str,
        #[case] swallowed: bool,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("name", ROUTE_HOST_MODULE).unwrap();
            locals.set_item("raised", raised).unwrap();
            py.run(
                c"
import builtins, sys, types
for parent in ('litellm', 'litellm.rust_bridge', 'litellm.rust_bridge.host', 'litellm.rust_bridge.host.routes'):
    sys.modules.setdefault(parent, types.ModuleType(parent))
saved = sys.modules.get(name)
stub = types.ModuleType(name)
def modify_max_tokens(model, args, kwargs):
    kwargs['max_tokens'] = 1
    raise getattr(builtins, raised)()
stub.modify_max_tokens = modify_max_tokens
sys.modules[name] = stub
",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let args = PyTuple::empty(py);
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("model", "anthropic/claude-sonnet-4-5")
                .unwrap();
            let mut hook = ModifyParams::new(&args, &kwargs).unwrap();
            let result = hook.arguments_prepared(py, &kwargs.clone().unbind());
            py.run(
                c"
if saved is None:
    sys.modules.pop(name, None)
else:
    sys.modules[name] = saved
",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            assert_eq!(result.is_ok(), swallowed);
            if let Err(error) = result {
                assert!(error.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py));
            }
        });
    }
}
