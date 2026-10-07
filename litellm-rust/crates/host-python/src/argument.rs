use pyo3::{prelude::*, types::PyDict};

pub fn lookup<'py>(
    kwargs: &Bound<'py, PyDict>,
    request: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if let Some(value) = kwargs.get_item(name)? {
        return Ok(Some(value));
    }
    if let Ok(bound) = request.cast::<PyDict>() {
        return bound.get_item(name);
    }
    request.getattr_opt(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_prefers_the_keyword_even_when_none_and_falls_back_to_the_request() {
        crate::initialize_python();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"
key = object()
document = {'type': 'document_url'}
class Request:
    api_key = 'from-request'
    api_base = 'from-request'
    document = document
request = Request()
kwargs = {'api_key': key, 'api_base': None}
",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let item = |name: &str| locals.get_item(name).unwrap().unwrap();
            let kwargs = item("kwargs").cast_into::<PyDict>().unwrap();
            let request = item("request");
            let find = |name: &str| lookup(&kwargs, &request, name).unwrap();
            assert!(find("api_key").unwrap().is(item("key")));
            assert!(find("api_base").unwrap().is_none());
            assert!(find("document").unwrap().is(item("document")));
            assert!(find("model").is_none());
        });
    }

    #[rstest::rstest]
    #[case::prepared_value("{'api_key': 'replacement'}", Some("replacement"))]
    #[case::explicit_none("{'api_key': None}", None)]
    #[case::bound_fallback("{}", Some("original"))]
    fn prepared_mapping_overrides_bound_values(
        #[case] source: &str,
        #[case] expected: Option<&str>,
    ) {
        crate::initialize_python();
        Python::attach(|py| {
            let bound = PyDict::new(py);
            bound.set_item("api_key", "original").unwrap();
            let source = std::ffi::CString::new(source).unwrap();
            let prepared = py
                .eval(&source, None, None)
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            let value = lookup(&prepared, bound.as_any(), "api_key")
                .unwrap()
                .unwrap();
            assert_eq!(
                value.extract::<Option<String>>().unwrap().as_deref(),
                expected
            );
        });
    }
}
