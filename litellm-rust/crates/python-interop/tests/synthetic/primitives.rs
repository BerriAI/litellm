use rstest::rstest;
use serde_json::{Value, json};
use serial_test::parallel;

use litellm_python_interop::{from_py, to_py};

use crate::support::python::{InitializedPython, initialized_python};

#[rstest]
#[parallel(python_interpreter)]
fn serde_values_round_trip_through_python(#[from(initialized_python)] python: &InitializedPython) {
    python.attach(|py| {
        let expected = json!({"model": "test", "items": [1, true, null]});
        let python_value = to_py(py, &expected).expect("value should convert to Python");
        let actual: Value =
            from_py(python_value.bind(py)).expect("Python value should convert to serde");

        assert_eq!(actual, expected);
    });
}
