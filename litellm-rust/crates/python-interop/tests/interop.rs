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
