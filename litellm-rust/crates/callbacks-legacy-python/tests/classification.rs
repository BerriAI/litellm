use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::CString,
};

use litellm_callbacks_legacy_python::{Dispatch, callback_mappings};
use pyo3::{prelude::*, types::PyDict};

fn home(dispatch: Dispatch) -> &'static str {
    match dispatch {
        Dispatch::Gateway(_) => "gateway",
        Dispatch::Router(_) => "router",
        Dispatch::Inference(_) => "inference",
        Dispatch::Management(_) => "management",
        Dispatch::HandlerTrait => "handler trait",
    }
}

const GUARDRAIL_HOOKS: [&str; 1] = ["apply_guardrail"];

fn custom_logger_methods() -> BTreeSet<String> {
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item(
                "source",
                include_str!("../../../../litellm/integrations/custom_logger.py"),
            )
            .unwrap();
        py.run(
            &CString::new(
                "import ast\n\
                 methods = {method.name for node in ast.parse(source).body \
                 if isinstance(node, ast.ClassDef) and node.name == 'CustomLogger' \
                 for method in node.body if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                 and not method.name.startswith('_')}",
            )
            .unwrap(),
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
        locals
            .get_item("methods")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap()
    })
}

#[test]
fn every_callback_method_has_exactly_one_home() {
    let methods: BTreeSet<String> = custom_logger_methods()
        .into_iter()
        .chain(GUARDRAIL_HOOKS.map(str::to_owned))
        .collect();
    let mut homes: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for entry in callback_mappings() {
        homes
            .entry(entry.callback)
            .or_default()
            .insert(home(entry.dispatch));
    }
    let classified: BTreeSet<String> = homes.keys().map(|name| (*name).to_owned()).collect();
    let split: Vec<_> = homes.iter().filter(|(_, home)| home.len() > 1).collect();

    assert_eq!(classified, methods);
    assert!(split.is_empty(), "{split:?}");
}
