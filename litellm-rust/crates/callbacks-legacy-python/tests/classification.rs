use std::{collections::BTreeSet, ffi::CString};

use litellm_callbacks_legacy_python::{Dispatch, callback_mappings};
use pyo3::{prelude::*, types::PyDict};
use rstest::rstest;

fn public_methods(source: &str, class: &str) -> BTreeSet<String> {
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals.set_item("source", source).unwrap();
        locals.set_item("class_name", class).unwrap();
        py.run(
            &CString::new(
                "import ast\n\
                 tree = ast.parse(source)\n\
                 methods = {method.name for node in tree.body \
                 if isinstance(node, ast.ClassDef) and node.name == class_name \
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

fn unclassified(methods: &BTreeSet<String>) -> BTreeSet<String> {
    let classified: BTreeSet<String> = callback_mappings()
        .map(|entry| entry.callback.to_owned())
        .collect();
    methods.difference(&classified).cloned().collect()
}

#[rstest]
fn every_public_method_has_exactly_one_home() {
    let methods: BTreeSet<String> = public_methods(
        include_str!("../../../../litellm/integrations/custom_logger.py"),
        "CustomLogger",
    )
    .union(&public_methods(
        include_str!("../../../../litellm/integrations/custom_guardrail.py"),
        "CustomGuardrail",
    ))
    .cloned()
    .collect();
    assert!(
        unclassified(&methods).is_empty(),
        "{:?}",
        unclassified(&methods)
    );
    let entries: Vec<_> = callback_mappings().collect();
    assert_eq!(entries.len(), methods.len());
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.callback)
            .collect::<BTreeSet<_>>()
            .len(),
        entries.len()
    );
}

#[rstest]
fn a_new_public_method_requires_classification() {
    let source = format!(
        "{}\n    def new_unclassified_hook(self): pass\n",
        include_str!("../../../../litellm/integrations/custom_logger.py"),
    );
    assert_eq!(
        unclassified(&public_methods(&source, "CustomLogger")),
        BTreeSet::from(["new_unclassified_hook".to_owned()]),
    );
}

#[rstest]
#[case::gateway(|dispatch| matches!(dispatch, Dispatch::Gateway(_)))]
#[case::router(|dispatch| matches!(dispatch, Dispatch::Router(_)))]
#[case::inference(|dispatch| matches!(dispatch, Dispatch::Inference(_)))]
fn each_request_layer_has_methods(#[case] belongs: fn(Dispatch) -> bool) {
    assert!(callback_mappings().any(|entry| belongs(entry.dispatch)));
}
