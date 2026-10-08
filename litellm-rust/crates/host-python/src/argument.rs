use pyo3::{prelude::*, types::PyDict};

/// What `original_function(*args, **kwargs)` sees: the signature base with the keyword
/// dict laid over it, so a rewritten keyword wins and a deleted keyword falls back to the
/// signature default.
pub fn effective<'py>(
    base: &Bound<'py, PyDict>,
    kwargs: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let merged = base.copy()?;
    merged.update(kwargs.as_mapping())?;
    Ok(merged)
}

pub fn present<'py>(
    arguments: &Bound<'py, PyDict>,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    Ok(arguments.get_item(name)?.filter(|value| !value.is_none()))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn dict<'py>(py: Python<'py>, source: &str) -> Bound<'py, PyDict> {
        py.eval(&std::ffi::CString::new(source).unwrap(), None, None)
            .unwrap()
            .cast_into::<PyDict>()
            .unwrap()
    }

    #[rstest]
    #[case::keyword_wins("{'api_key': 'base'}", "{'api_key': 'keyword'}", Some(Some("keyword")))]
    #[case::explicit_none_wins("{'api_key': 'base'}", "{'api_key': None}", Some(None))]
    #[case::base_default("{'api_key': 'base'}", "{}", Some(Some("base")))]
    #[case::keyword_only("{}", "{'api_key': 'keyword'}", Some(Some("keyword")))]
    #[case::missing("{}", "{}", None)]
    fn effective_lays_the_keywords_over_the_base(
        #[case] base: &str,
        #[case] kwargs: &str,
        #[case] expected: Option<Option<&str>>,
    ) {
        crate::initialize_python();
        Python::attach(|py| {
            let merged = effective(&dict(py, base), &dict(py, kwargs)).unwrap();
            let value = merged
                .get_item("api_key")
                .unwrap()
                .map(|value| value.extract::<Option<String>>().unwrap());
            assert_eq!(value, expected.map(|value| value.map(str::to_owned)));
        });
    }

    #[rstest]
    fn effective_leaves_both_inputs_untouched_and_keeps_object_identity() {
        crate::initialize_python();
        Python::attach(|py| {
            let document = PyDict::new(py);
            let base = dict(py, "{'model': 'base', 'pages': None}");
            let kwargs = PyDict::new(py);
            kwargs.set_item("document", &document).unwrap();
            let merged = effective(&base, &kwargs).unwrap();
            merged.set_item("model", "merged").unwrap();
            assert_eq!(
                base.get_item("model")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "base"
            );
            assert!(!kwargs.contains("model").unwrap());
            assert!(merged.get_item("document").unwrap().unwrap().is(&document));
        });
    }

    #[rstest]
    #[case::explicit_none_is_unset("{'api_key': None}", None)]
    #[case::value("{'api_key': 'set'}", Some("set"))]
    #[case::missing("{}", None)]
    fn present_treats_none_as_unset(#[case] arguments: &str, #[case] expected: Option<&str>) {
        crate::initialize_python();
        Python::attach(|py| {
            let value = present(&dict(py, arguments), "api_key")
                .unwrap()
                .map(|value| value.extract::<String>().unwrap());
            assert_eq!(value.as_deref(), expected);
        });
    }
}
