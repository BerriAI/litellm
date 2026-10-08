use pyo3::{prelude::*, types::PyDict};

pub fn lookup<'py>(
    kwargs: &Bound<'py, PyDict>,
    bound: &Bound<'py, PyDict>,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match kwargs.get_item(name)? {
        Some(value) => Ok(Some(value)),
        None => bound.get_item(name),
    }
}

pub fn present<'py>(
    kwargs: &Bound<'py, PyDict>,
    bound: &Bound<'py, PyDict>,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    Ok(lookup(kwargs, bound, name)?.filter(|value| !value.is_none()))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn dicts<'py>(
        py: Python<'py>,
        kwargs: &str,
        bound: &str,
    ) -> (Bound<'py, PyDict>, Bound<'py, PyDict>) {
        let eval = |source: &str| {
            py.eval(&std::ffi::CString::new(source).unwrap(), None, None)
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap()
        };
        (eval(kwargs), eval(bound))
    }

    #[rstest]
    #[case::keyword_wins(
        "{'api_key': 'keyword'}",
        "{'api_key': 'bound'}",
        Some(Some("keyword"))
    )]
    #[case::explicit_none_wins("{'api_key': None}", "{'api_key': 'bound'}", Some(None))]
    #[case::bound_fallback("{}", "{'api_key': 'bound'}", Some(Some("bound")))]
    #[case::missing("{}", "{}", None)]
    fn lookup_prefers_the_keyword_and_falls_back_to_bound(
        #[case] kwargs: &str,
        #[case] bound: &str,
        #[case] expected: Option<Option<&str>>,
    ) {
        crate::initialize_python();
        Python::attach(|py| {
            let (kwargs, bound) = dicts(py, kwargs, bound);
            let value = lookup(&kwargs, &bound, "api_key")
                .unwrap()
                .map(|value| value.extract::<Option<String>>().unwrap());
            assert_eq!(value, expected.map(|value| value.map(str::to_owned)));
        });
    }

    #[rstest]
    #[case::explicit_none_hides_bound("{'api_key': None}", "{'api_key': 'bound'}", None)]
    #[case::bound_none("{}", "{'api_key': None}", None)]
    #[case::bound_value("{}", "{'api_key': 'bound'}", Some("bound"))]
    fn present_treats_none_as_unset(
        #[case] kwargs: &str,
        #[case] bound: &str,
        #[case] expected: Option<&str>,
    ) {
        crate::initialize_python();
        Python::attach(|py| {
            let (kwargs, bound) = dicts(py, kwargs, bound);
            let value = present(&kwargs, &bound, "api_key")
                .unwrap()
                .map(|value| value.extract::<String>().unwrap());
            assert_eq!(value.as_deref(), expected);
        });
    }

    #[rstest]
    fn lookup_returns_the_callers_object() {
        crate::initialize_python();
        Python::attach(|py| {
            let document = PyDict::new(py);
            let bound = PyDict::new(py);
            bound.set_item("document", &document).unwrap();
            let found = lookup(&PyDict::new(py), &bound, "document")
                .unwrap()
                .unwrap();
            assert!(found.is(&document));
        });
    }
}
