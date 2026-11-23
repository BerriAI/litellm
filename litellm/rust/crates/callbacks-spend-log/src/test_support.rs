//! Readers for the helper goldens under `tests/goldens/helpers/`, which `scripts/record_goldens.py`
//! records from the Python helper each module ports.

use std::sync::Arc;

use litellm_python_compat::{Value, json};
use litellm_redaction::TokenRedactor;

use crate::py::get;

/// One recorded call: the field of the record named `name`.
pub(crate) fn field<'a>(record: &'a Value, name: &str) -> &'a Value {
    get(record, name).unwrap_or_else(|| panic!("record has no {name}"))
}

/// What the recorder writes after the first character of each token-shaped string, so that no
/// committed golden matches a secret scanner. Readers remove it before parsing a line.
pub(crate) const TOKEN_SPLIT: &str = "#~#";

/// The recorded calls of one helper, from its JSON lines file.
pub(crate) fn helper_records(text: &str) -> Vec<Value> {
    let records: Vec<Value> = text
        .lines()
        .map(|line| json::loads(&line.replace(TOKEN_SPLIT, "")).expect("golden parses"))
        .collect();
    assert!(!records.is_empty(), "no recorded calls");
    records
}

/// The positional argument at `index` of a recorded call.
pub(crate) fn argument(record: &Value, index: usize) -> &Value {
    match field(record, "args") {
        Value::List(args) => &args[index],
        other => panic!("args is {other:?}"),
    }
}

/// The token redactor the proxy builds, from the recorded word ranges.
pub(crate) fn token_redactor() -> Arc<TokenRedactor> {
    let ranges = match json::loads(include_str!("../tests/goldens/token_word_ranges.json"))
        .expect("word ranges parse")
    {
        Value::List(pairs) => pairs
            .iter()
            .map(|pair| match pair {
                Value::List(bounds) => match bounds.as_slice() {
                    [Value::Int(start), Value::Int(end)] => (
                        u32::try_from(start).expect("range start fits"),
                        u32::try_from(end).expect("range end fits"),
                    ),
                    _ => panic!("malformed word range"),
                },
                _ => panic!("malformed word range"),
            })
            .collect(),
        _ => panic!("word ranges are a list"),
    };
    Arc::new(TokenRedactor::new(16, ranges))
}

/// The case name of a recorded call, for failure messages.
pub(crate) fn case(record: &Value) -> String {
    litellm_python_compat::repr::to_str(field(record, "case"))
}
