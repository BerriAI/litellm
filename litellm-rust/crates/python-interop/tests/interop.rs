use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyList, PyString, PyTuple};
use rstest::{fixture, rstest};
use serde_json::{Value, json};

use litellm_python_interop::{
    array_from_py, from_py, release_count, release_gil, to_py, value_to_py,
};

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
fn value_converter_matches_pythonize_and_round_trips(
    #[from(initialized_python)] python: &InitializedPython,
) {
    python.attach(|py| {
        let payload = json!({
            "null": null,
            "bools": [true, false],
            "numbers": [0, -9223372036854775808i64, 9223372036854775807i64, 18446744073709551615u64],
            "floats": [0.0, -2.5, 1024.75],
            "unicode": "héllo 🌍 中文",
            "empty_list": [],
            "empty_dict": {},
            "nested": {"a": [{"b": [null, {"c": "deep", "d": [1, 2, 3]}]}]},
        });
        let converted = value_to_py(py, &payload).expect("value should convert to Python");
        let reference = to_py(py, &payload).expect("pythonize should convert the same value");
        let equal: bool = converted
            .bind(py)
            .eq(reference.bind(py))
            .expect("converted values should compare in Python");
        assert!(equal, "value converter diverged from pythonize");

        let round_tripped: Value = from_py(converted.bind(py)).expect("Python value should convert back");
        assert_eq!(round_tripped, payload);
    });
}

#[rstest]
fn value_converter_shares_repeated_keys_within_one_call(
    #[from(initialized_python)] python: &InitializedPython,
) {
    python.attach(|py| {
        let payload = json!({
            "type": "message",
            "content": [
                {"type": "text", "text": "one", "index": 0},
                {"type": "text", "text": "two", "index": 1},
                {"type": "text", "text": "three", "index": 2},
            ],
        });
        let first = value_to_py(py, &payload).expect("first payload should convert");
        let second = value_to_py(py, &payload).expect("second payload should convert");
        let globals = PyDict::new(py);
        globals
            .set_item("first", &first)
            .expect("first payload should enter Python locals");
        globals
            .set_item("second", &second)
            .expect("second payload should enter Python locals");
        py.run(
            c"
first_keys = [k for block in first['content'] for k in block]
second_keys = [k for block in second['content'] for k in block]
shared = [k for k in first_keys if k == 'type']
assert all(k is shared[0] for k in shared)
assert all(k is not second_keys[0] for k in first_keys)
",
            Some(&globals),
            None,
        )
        .expect("repeated keys should resolve to one object per conversion");
    });
}

#[rstest]
fn array_from_py_validates_shape_before_depythonizing(
    #[from(initialized_python)] python: &InitializedPython,
) {
    python.attach(|py| {
        for rejected in [
            PyDict::new(py).into_any(),
            PyString::new(py, "text").into_any(),
            py.None().into_bound(py),
        ] {
            let error = array_from_py("messages", &rejected)
                .expect_err("non-array input should be rejected before depythonization");
            assert_eq!(error.to_string(), "ValueError: messages must be a list");
        }

        let list = PyList::new(py, [1, 2]).expect("list should be created");
        let converted = array_from_py("messages", &list)
            .expect("list input should depythonize after the shape check");
        assert_eq!(converted, json!([1, 2]));

        let tuple = PyTuple::new(py, ["a"]).expect("tuple should be created");
        let converted = array_from_py("messages", &tuple)
            .expect("tuple input should depythonize after the shape check");
        assert_eq!(converted, json!(["a"]));
    });
}

#[rstest]
fn release_gil_runs_work_and_records_it(#[from(initialized_python)] python: &InitializedPython) {
    let before = release_count();
    let result = python.attach(|py| release_gil(py, || 42));

    assert_eq!(result, 42);
    assert_eq!(release_count(), before + 1);
}

#[rstest]
fn immutable_payloads_share_storage_and_mutable_payloads_take_snapshots(
    #[from(initialized_python)] python: &InitializedPython,
) {
    use litellm_python_interop::{bytes_from_py, text_bytes_from_py};
    use pyo3::prelude::*;
    use pyo3::types::{PyByteArray, PyBytes, PyBytesMethods, PyString, PyStringMethods};

    let (raw, text) = python.attach(|py| {
        let source = PyBytes::new(py, b"large immutable media");
        let references = refcount(&source);
        let raw = bytes_from_py(source.as_any()).unwrap();
        assert_eq!(raw.as_ptr(), source.as_bytes().as_ptr());
        assert_eq!(refcount(&source), references + 1);
        let replay = raw.clone().slice(6..15);
        assert_eq!(replay.as_ptr(), source.as_bytes()[6..].as_ptr());
        drop(replay);
        assert_eq!(refcount(&source), references + 1);
        let text = PyString::new(py, "QUJDREVGR0g=");
        let shared_text = text_bytes_from_py(text.as_any()).unwrap();
        assert_eq!(shared_text.as_ptr(), text.to_str().unwrap().as_ptr());
        let mutable = PyByteArray::new(py, b"original");
        let snapshot = bytes_from_py(mutable.as_any()).unwrap();
        mutable.set_item(0, b'X').unwrap();
        assert_eq!(snapshot.as_ref(), b"original");
        (raw, shared_text)
    });
    std::thread::spawn(move || {
        assert_eq!(raw.as_ref(), b"large immutable media");
        assert_eq!(text.slice(4..).as_ref(), b"REVGR0g=");
    })
    .join()
    .unwrap();
}

#[rstest]
fn owners_are_released_on_success_error_and_cancelled_suspension(
    #[from(initialized_python)] python: &InitializedPython,
) {
    use litellm_python_interop::bytes_from_py;
    use pyo3::types::PyBytes;
    use std::future::Future;
    use std::task::{Context, Poll, Waker};

    let source = python.attach(|py| PyBytes::new(py, b"owner lifetime").unbind());
    let baseline = python.attach(|py| refcount(source.bind(py)));
    for fail in [false, true] {
        let bytes = python.attach(|py| bytes_from_py(source.bind(py).as_any()).unwrap());
        let result = std::thread::spawn(move || {
            assert_eq!(bytes.as_ref(), b"owner lifetime");
            if fail { Err(()) } else { Ok(()) }
        })
        .join()
        .unwrap();
        assert_eq!(result.is_err(), fail);
        python.attach(|py| assert_eq!(refcount(source.bind(py)), baseline));
    }
    let bytes = python.attach(|py| bytes_from_py(source.bind(py).as_any()).unwrap());
    let mut future = Box::pin(async move {
        std::future::pending::<()>().await;
        assert_eq!(bytes.as_ref(), b"owner lifetime");
    });
    assert_eq!(
        future
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop())),
        Poll::Pending
    );
    python.attach(|py| assert_eq!(refcount(source.bind(py)), baseline + 1));
    drop(future);
    python.attach(|py| assert_eq!(refcount(source.bind(py)), baseline));
}

fn refcount<T>(value: &pyo3::Bound<'_, T>) -> isize {
    unsafe { pyo3::ffi::Py_REFCNT(value.as_ptr()) }
}
