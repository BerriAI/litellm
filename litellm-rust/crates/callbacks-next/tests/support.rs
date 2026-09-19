use std::ffi::CStr;

use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) const PYTHON_CONTRACT: &str = include_str!("../python_contract.json");

const STUBS: &CStr = c"
import inspect
import json
import sys
import types

for name in ('litellm', 'litellm.rust_bridge', 'litellm.rust_bridge.callbacks_next'):
    sys.modules.setdefault(name, types.ModuleType(name))

module = sys.modules['litellm.rust_bridge.callbacks_next']
contract = json.loads(python_contract)

def contracted(name, fake):
    signature = inspect.Signature([
        inspect.Parameter(parameter, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for parameter in contract[name]
    ])
    def checked(*args, **kwargs):
        signature.bind(*args, **kwargs)
        return fake(*args, **kwargs)
    return checked

if not hasattr(module, 'reports'):
    module.reports = []

def project_response(response):
    if getattr(response, 'project_error', False):
        raise ValueError('projection failed')
    return response

functions = {
    'snapshot': lambda: (),
    'project_response': project_response,
    'new_call_id': lambda: 'generated-call-id',
    'report': lambda name, event, error: module.reports.append((name, event, error)),
}
for name, function in functions.items():
    setattr(module, name, contracted(name, function))
";

pub(crate) fn namespace<'py>(py: Python<'py>, script: &CStr) -> Bound<'py, PyDict> {
    let locals = PyDict::new(py);
    locals.set_item("python_contract", PYTHON_CONTRACT).unwrap();
    py.run(STUBS, Some(&locals), Some(&locals)).unwrap();
    py.run(script, Some(&locals), Some(&locals)).unwrap();
    locals
}

pub(crate) fn local<'py>(locals: &Bound<'py, PyDict>, name: &str) -> Bound<'py, PyAny> {
    locals.get_item(name).unwrap().unwrap()
}
