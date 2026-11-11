use std::{cell::Cell, rc::Rc};

use litellm_host_python::{FromPythonCache, Pythonized, ToPythonCache};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use rstest::{fixture, rstest};

#[fixture]
fn python() {
    Python::initialize();
}

#[rstest]
fn rust_identity_reuses_python_objects_without_merging_equal_values(#[from(python)] _python: ()) {
    Python::attach(|py| {
        let original = Rc::new(vec![1, 2]);
        let cloned = original.clone();
        let equal = Rc::new(vec![1, 2]);
        let mut cache = ToPythonCache::default();
        let first = cache
            .get_or_try_insert_with(original.as_ref(), |value| {
                Pythonized(value).into_pyobject(py)
            })
            .unwrap()
            .clone();
        let second = cache
            .get_or_try_insert_with(cloned.as_ref(), |_| panic!("must reuse conversion"))
            .unwrap()
            .clone();
        let third = cache
            .get_or_try_insert_with(equal.as_ref(), |value| Pythonized(value).into_pyobject(py))
            .unwrap();
        assert!(first.is(&second));
        assert!(!first.is(third));
        assert!(first.eq(third).unwrap());
    });
}

#[rstest]
fn python_identity_reuses_rust_values_without_merging_equal_objects(#[from(python)] _python: ()) {
    Python::attach(|py| {
        let original = PyDict::new(py);
        original.set_item("value", 1).unwrap();
        let equal = original.copy().unwrap();
        let calls = Cell::new(0);
        let mut cache = FromPythonCache::default();
        let convert = |value: &Bound<'_, PyAny>| {
            calls.set(calls.get() + 1);
            value.get_item("value")?.extract::<i32>().map(Rc::new)
        };
        let first = cache
            .get_or_try_insert_with(original.as_any(), convert)
            .unwrap()
            .clone();
        let second = cache
            .get_or_try_insert_with(original.as_any(), convert)
            .unwrap()
            .clone();
        let third = cache
            .get_or_try_insert_with(equal.as_any(), convert)
            .unwrap();
        assert!(Rc::ptr_eq(&first, &second));
        assert!(!Rc::ptr_eq(&first, third));
        assert_eq!(&first, third);
        assert_eq!(calls.get(), 2);
    });
}

#[rstest]
fn python_sources_stay_alive_until_the_cache_is_dropped(#[from(python)] _python: ()) {
    Python::attach(|py| {
        let value = py
            .eval(pyo3::ffi::c_str!("type('Tracked', (), {})()"), None, None)
            .unwrap();
        let weak = py
            .import("weakref")
            .unwrap()
            .call_method1("ref", (&value,))
            .unwrap();
        let mut cache = FromPythonCache::default();
        cache.get_or_try_insert_with(&value, |_| Ok(42)).unwrap();
        drop(value);
        assert!(!weak.call0().unwrap().is_none());
        drop(cache);
        assert!(weak.call0().unwrap().is_none());
    });
}

#[rstest]
#[case::to_python(true)]
#[case::from_python(false)]
fn failed_conversions_preserve_exceptions_and_can_be_retried(
    #[from(python)] _python: (),
    #[case] to_python: bool,
) {
    Python::attach(|py| {
        let failure = PyValueError::new_err("conversion failed");
        if to_python {
            let source = vec![1, 2];
            let mut cache = ToPythonCache::default();
            let error = cache
                .get_or_try_insert_with(&source, |_| Err(failure.clone_ref(py)))
                .unwrap_err();
            assert!(error.value(py).is(failure.value(py)));
            let result = cache
                .get_or_try_insert_with(&source, |value| Pythonized(value).into_pyobject(py))
                .unwrap();
            assert_eq!(result.extract::<Vec<i32>>().unwrap(), source);
        } else {
            let source = PyDict::new(py).into_any();
            let mut cache = FromPythonCache::default();
            let error = cache
                .get_or_try_insert_with(&source, |_| Err(failure.clone_ref(py)))
                .unwrap_err();
            assert!(error.value(py).is(failure.value(py)));
            assert_eq!(
                *cache.get_or_try_insert_with(&source, |_| Ok(42)).unwrap(),
                42
            );
        }
    });
}
