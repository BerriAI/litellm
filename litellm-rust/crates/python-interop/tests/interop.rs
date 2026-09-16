use pyo3::Python;
use rstest::{fixture, rstest};
use serde_json::{Value, json};

use litellm_python_interop::{from_py, release_count, release_gil, to_py};

struct InitializedPython;

impl InitializedPython {
    fn attach<F, R>(&self, f: F) -> R
    where
        F: for<'py> FnOnce(Python<'py>) -> R,
    {
        Python::attach(f)
    }
}

#[fixture]
#[once]
fn initialized_python() -> InitializedPython {
    Python::initialize();
    InitializedPython
}

#[rstest]
fn serde_values_round_trip_through_python(#[from(initialized_python)] python: &InitializedPython) {
    python.attach(|py| {
        let expected = json!({"model": "test", "items": [1, true, null]});
        let python_value = to_py(py, &expected).expect("value should convert to Python");
        let actual: Value =
            from_py(python_value.bind(py)).expect("Python value should convert to serde");

        assert_eq!(actual, expected);
    });
}

#[rstest]
fn release_gil_runs_work_and_records_it(#[from(initialized_python)] python: &InitializedPython) {
    let before = release_count();
    let result = python.attach(|py| release_gil(py, || 42));

    assert_eq!(result, 42);
    assert_eq!(release_count(), before + 1);
}
