use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

pub(crate) struct Signature {
    pub name: &'static str,
    pub parameters: &'static [&'static str],
    pub required: usize,
}

#[derive(Debug)]
pub(crate) struct BoundArguments<'py> {
    kwargs: Bound<'py, PyDict>,
}

impl Signature {
    pub(crate) fn bind<'py>(
        &self,
        args: &Bound<'py, PyTuple>,
        kwargs: &Bound<'py, PyDict>,
    ) -> PyResult<BoundArguments<'py>> {
        if args.len() > self.parameters.len() {
            return Err(PyTypeError::new_err(format!(
                "{}() takes {} positional arguments but {} were given",
                self.name,
                self.parameters.len(),
                args.len()
            )));
        }
        let bound = kwargs.copy()?;
        for (name, value) in self.parameters.iter().zip(args.iter()) {
            if kwargs.contains(name)? {
                return Err(PyTypeError::new_err(format!(
                    "{}() got multiple values for argument '{name}'",
                    self.name
                )));
            }
            bound.set_item(name, value)?;
        }
        let missing: Vec<&str> = self.parameters[..self.required]
            .iter()
            .copied()
            .filter(|name| !bound.contains(name).unwrap_or(false))
            .collect();
        match missing.as_slice() {
            [] => {}
            [name] => {
                return Err(PyTypeError::new_err(format!(
                    "{}() missing 1 required positional argument: '{name}'",
                    self.name
                )));
            }
            names => {
                let quoted: Vec<String> = names.iter().map(|name| format!("'{name}'")).collect();
                let (last, rest) = quoted.split_last().expect("at least two names");
                return Err(PyTypeError::new_err(format!(
                    "{}() missing {} required positional arguments: {} and {last}",
                    self.name,
                    names.len(),
                    rest.join(", ")
                )));
            }
        }
        Ok(BoundArguments { kwargs: bound })
    }
}

impl<'py> BoundArguments<'py> {
    pub(crate) fn get(&self, name: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.kwargs.get_item(name)
    }

    pub(crate) fn required(&self, name: &str) -> PyResult<Bound<'py, PyAny>> {
        self.get(name)?
            .ok_or_else(|| PyTypeError::new_err(format!("missing required argument '{name}'")))
    }

    pub(crate) fn extract<T>(&self, name: &str) -> PyResult<T>
    where
        T: for<'a> FromPyObject<'a, 'py>,
        for<'a> <T as FromPyObject<'a, 'py>>::Error: Into<PyErr>,
    {
        self.required(name)?.extract().map_err(Into::into)
    }

    pub(crate) fn optional<T>(&self, name: &str) -> PyResult<Option<T>>
    where
        T: for<'a> FromPyObject<'a, 'py>,
        for<'a> <T as FromPyObject<'a, 'py>>::Error: Into<PyErr>,
    {
        self.get(name)?
            .filter(|value| !value.is_none())
            .map(|value| value.extract().map_err(Into::into))
            .transpose()
    }

    pub(crate) fn kwargs(&self) -> &Bound<'py, PyDict> {
        &self.kwargs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OCR: Signature = Signature {
        name: "ocr",
        parameters: &["model", "document", "api_key"],
        required: 2,
    };

    fn call<'py>(
        py: Python<'py>,
        args: &[&str],
        kwargs: &[(&str, &str)],
    ) -> PyResult<BoundArguments<'py>> {
        let args = PyTuple::new(py, args).unwrap();
        let dict = PyDict::new(py);
        for (name, value) in kwargs {
            dict.set_item(name, value).unwrap();
        }
        OCR.bind(&args, &dict)
    }

    #[test]
    fn positional_and_keyword_arguments_bind_like_python() {
        Python::initialize();
        Python::attach(|py| {
            let bound = call(py, &["m", "d"], &[("api_key", "k"), ("extra", "x")]).unwrap();
            assert_eq!(bound.extract::<String>("model").unwrap(), "m");
            assert_eq!(bound.extract::<String>("document").unwrap(), "d");
            assert_eq!(
                bound.optional::<String>("api_key").unwrap().as_deref(),
                Some("k")
            );
            assert_eq!(bound.kwargs().len(), 4);

            let bound = call(py, &[], &[("document", "d"), ("model", "m")]).unwrap();
            assert_eq!(bound.extract::<String>("model").unwrap(), "m");
        });
    }

    #[test]
    fn binding_errors_match_python_messages() {
        Python::initialize();
        Python::attach(|py| {
            let error = call(py, &["m", "d"], &[("model", "dup")]).unwrap_err();
            assert_eq!(
                error.to_string(),
                "TypeError: ocr() got multiple values for argument 'model'"
            );

            let error = call(py, &["m"], &[]).unwrap_err();
            assert_eq!(
                error.to_string(),
                "TypeError: ocr() missing 1 required positional argument: 'document'"
            );

            let error = call(py, &[], &[]).unwrap_err();
            assert_eq!(
                error.to_string(),
                "TypeError: ocr() missing 2 required positional arguments: 'model' and 'document'"
            );

            let error = call(py, &["m", "d", "k", "extra"], &[]).unwrap_err();
            assert_eq!(
                error.to_string(),
                "TypeError: ocr() takes 3 positional arguments but 4 were given"
            );
        });
    }

    #[test]
    fn explicit_none_is_absent_for_optional_and_present_for_required() {
        Python::initialize();
        Python::attach(|py| {
            let args = PyTuple::new(py, ["m"]).unwrap();
            let dict = PyDict::new(py);
            dict.set_item("document", py.None()).unwrap();
            dict.set_item("api_key", py.None()).unwrap();
            let bound = OCR.bind(&args, &dict).unwrap();
            assert!(bound.required("document").unwrap().is_none());
            assert_eq!(bound.optional::<String>("api_key").unwrap(), None);
        });
    }
}
