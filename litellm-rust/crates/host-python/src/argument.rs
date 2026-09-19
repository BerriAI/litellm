use pyo3::{prelude::*, types::PyDict};

/// The caller's own object for a public argument: the keyword if given, even an explicit
/// `None`, else the bound request's attribute. Every reader of a public Python call uses
/// this rule, so the callbacks and the provider see one object per argument.
pub fn lookup<'py>(
    kwargs: &Bound<'py, PyDict>,
    request: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if let Some(value) = kwargs.get_item(name)? {
        return Ok(Some(value));
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
}
