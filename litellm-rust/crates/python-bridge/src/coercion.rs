use litellm_core_utils::serde_compat::parse_str_bool;
use pyo3::{
    exceptions::{PyAttributeError, PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBool, PyString},
};

#[derive(Debug)]
pub(crate) enum ProjectionError {
    Python(PyErr),
    InvalidConfiguration(String),
    UnsupportedLiveObject(String),
    InternalSchemaFailure(String),
}

impl From<PyErr> for ProjectionError {
    fn from(error: PyErr) -> Self {
        Self::Python(error)
    }
}

impl From<ProjectionError> for PyErr {
    fn from(error: ProjectionError) -> Self {
        match error {
            ProjectionError::Python(error) => error,
            ProjectionError::InvalidConfiguration(message)
            | ProjectionError::UnsupportedLiveObject(message) => PyValueError::new_err(message),
            ProjectionError::InternalSchemaFailure(message) => PyRuntimeError::new_err(message),
        }
    }
}

pub(crate) struct FieldSpec<T> {
    name: &'static str,
    decode: fn(&Field<'_>) -> Result<T, ProjectionError>,
}

impl<T> FieldSpec<T> {
    pub(crate) const fn new(
        name: &'static str,
        decode: fn(&Field<'_>) -> Result<T, ProjectionError>,
    ) -> Self {
        Self { name, decode }
    }

    pub(crate) fn read(
        &self,
        snapshot: &Bound<'_, PyAny>,
        group: &'static str,
    ) -> Result<T, ProjectionError> {
        (self.decode)(&Field::read(snapshot, group, self.name)?)
    }
}

pub(crate) struct Field<'py> {
    group: &'static str,
    name: &'static str,
    value: Bound<'py, PyAny>,
}

impl<'py> Field<'py> {
    pub(crate) fn new(group: &'static str, name: &'static str, value: Bound<'py, PyAny>) -> Self {
        Self { group, name, value }
    }

    /// Reads `snapshot.<name>`, distinguishing a field the accessor never declared from a
    /// descriptor that raised `AttributeError`.
    pub(crate) fn read(
        snapshot: &Bound<'py, PyAny>,
        group: &'static str,
        name: &'static str,
    ) -> Result<Self, ProjectionError> {
        match snapshot.getattr(name) {
            Ok(value) => Ok(Self::new(group, name, value)),
            Err(error) if error.is_instance_of::<PyAttributeError>(snapshot.py()) => {
                match Self::missing_field(snapshot, name) {
                    Ok(true) => Err(ProjectionError::InternalSchemaFailure(format!(
                        "{group}.{name}: missing snapshot field"
                    ))),
                    _ => Err(error.into()),
                }
            }
            Err(error) => Err(error.into()),
        }
    }

    fn missing_field(snapshot: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
        let py = snapshot.py();
        let object = py.import("builtins")?.getattr("object")?;
        let missing = object.call0()?;
        let lookup = py.import("inspect")?.getattr("getattr_static")?;
        let declared = lookup.call1((snapshot, name, &missing))?;
        let fallback = lookup.call1((snapshot.get_type(), "__getattr__", &missing))?;
        let getter = lookup.call1((snapshot.get_type(), "__getattribute__"))?;
        Ok(declared.is(&missing)
            && fallback.is(&missing)
            && getter.is(object.getattr("__getattribute__")?))
    }

    pub(crate) fn path(&self) -> String {
        format!("{}.{}", self.group, self.name)
    }

    /// A member of this field's collection, reported under the same path.
    pub(crate) fn member(&self, value: Bound<'py, PyAny>) -> Self {
        Self::new(self.group, self.name, value)
    }

    pub(crate) fn expected(&self, expected: &str) -> Result<String, ProjectionError> {
        Ok(format!(
            "{}: expected {expected}, got {}",
            self.path(),
            self.value.get_type().name()?
        ))
    }

    pub(crate) fn invalid(&self, expected: &str) -> ProjectionError {
        match self.expected(expected) {
            Ok(message) => ProjectionError::InvalidConfiguration(message),
            Err(error) => error,
        }
    }

    pub(crate) fn value(&self) -> &Bound<'py, PyAny> {
        &self.value
    }

    pub(crate) fn truthy(&self) -> Result<bool, ProjectionError> {
        Ok(self.value.is_truthy()?)
    }

    pub(crate) fn exact_true(&self) -> bool {
        self.value.is(PyBool::new(self.value.py(), true))
    }

    pub(crate) fn schema_bool(&self) -> Result<bool, ProjectionError> {
        if !self.value.is_instance_of::<PyBool>() {
            return Err(ProjectionError::InternalSchemaFailure(
                self.expected("a Boolean")?,
            ));
        }
        Ok(self.exact_true())
    }

    pub(crate) fn strict_string(&self) -> Result<String, ProjectionError> {
        let value = self
            .value
            .cast::<PyString>()
            .map_err(|_| self.invalid("a string"))?;
        Ok(value.to_str()?.to_owned())
    }

    pub(crate) fn schema_string(&self) -> Result<String, ProjectionError> {
        if !self.value.is_instance_of::<PyString>() {
            return Err(ProjectionError::InternalSchemaFailure(
                self.expected("a string")?,
            ));
        }
        self.strict_string()
    }

    pub(crate) fn str_bool(&self) -> Result<Option<bool>, ProjectionError> {
        if self.value.is_none() {
            return Ok(None);
        }
        Ok(parse_str_bool(&self.strict_string()?))
    }

    pub(crate) fn optional_strict_string(&self) -> Result<Option<String>, ProjectionError> {
        if self.value.is_none() {
            return Ok(None);
        }
        self.strict_string().map(Some)
    }

    pub(crate) fn falsy_optional_string(&self) -> Result<Option<String>, ProjectionError> {
        if !self.truthy()? {
            return Ok(None);
        }
        self.strict_string().map(Some)
    }

    pub(crate) fn tuning_string(&self) -> Result<Option<String>, ProjectionError> {
        if !self.truthy()? || !self.value.is_instance_of::<PyString>() {
            return Ok(None);
        }
        self.strict_string().map(Some)
    }

    pub(crate) fn string_collection(&self) -> Result<Vec<String>, ProjectionError> {
        if !self.truthy()? {
            return Ok(Vec::new());
        }
        if self.value.is_instance_of::<PyString>() {
            return self.strict_string().map(|value| vec![value]);
        }
        self.value
            .try_iter()?
            .filter_map(|item| {
                let member = match item {
                    Ok(value) => self.member(value),
                    Err(error) => return Some(Err(error.into())),
                };
                match member.truthy() {
                    Ok(false) => None,
                    Ok(true) => Some(member.strict_string()),
                    Err(error) => Some(Err(error)),
                }
            })
            .collect()
    }

    pub(crate) fn optional_string_collection(
        &self,
    ) -> Result<Option<Vec<String>>, ProjectionError> {
        if self.value.is_none() {
            return Ok(None);
        }
        self.string_collection().map(Some)
    }

    pub(crate) fn python_binding(&self) -> Option<Py<PyAny>> {
        (!self.value.is_none()).then(|| self.value.clone().unbind())
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use pyo3::{
        exceptions::{PyLookupError, PyRuntimeError, PyValueError},
        types::PyDict,
    };
    use rstest::rstest;

    use super::*;

    fn evaluate<'py>(py: Python<'py>, source: &str) -> Bound<'py, PyAny> {
        py.eval(&CString::new(source).unwrap(), None, None).unwrap()
    }

    #[rstest]
    #[case("None", false, false)]
    #[case("False", false, false)]
    #[case("True", true, true)]
    #[case("0", false, false)]
    #[case("1", true, false)]
    #[case("''", false, false)]
    #[case("'false'", true, false)]
    #[case("[]", false, false)]
    #[case("[0]", true, false)]
    #[case("{}", false, false)]
    #[case("object()", true, false)]
    fn boolean_operations_have_distinct_python_semantics(
        #[case] source: &str,
        #[case] truth: bool,
        #[case] exact: bool,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let value = evaluate(py, source);
            let field = Field::new("test", "flag", value.clone());
            assert_eq!(field.truthy().unwrap(), truth);
            assert_eq!(field.exact_true(), exact);
            assert_eq!(
                field.truthy().unwrap(),
                py.import("builtins")
                    .unwrap()
                    .getattr("bool")
                    .unwrap()
                    .call1((value,))
                    .unwrap()
                    .extract::<bool>()
                    .unwrap()
            );
        });
    }

    #[rstest]
    #[case("None", Ok(None), Ok(None), Ok(None))]
    #[case("''", Ok(Some("")), Ok(None), Ok(None))]
    #[case(
        "' value '",
        Ok(Some(" value ")),
        Ok(Some(" value ")),
        Ok(Some(" value "))
    )]
    #[case("[]", Err(()), Ok(None), Ok(None))]
    #[case("0", Err(()), Ok(None), Ok(None))]
    #[case("1", Err(()), Err(()), Ok(None))]
    #[case("object()", Err(()), Err(()), Ok(None))]
    fn string_operations_do_not_conflate_absence_and_type_checks(
        #[case] source: &str,
        #[case] strict: Result<Option<&str>, ()>,
        #[case] fallback: Result<Option<&str>, ()>,
        #[case] tuning: Result<Option<&str>, ()>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let field = Field::new("test", "string", evaluate(py, source));
            let owned =
                |expected: Result<Option<&str>, ()>| expected.map(|value| value.map(str::to_owned));
            assert_eq!(
                field.optional_strict_string().map_err(|_| ()),
                owned(strict)
            );
            assert_eq!(
                field.falsy_optional_string().map_err(|_| ()),
                owned(fallback)
            );
            assert_eq!(field.tuning_string().map_err(|_| ()), owned(tuning));
        });
    }

    #[rstest]
    #[case("None", None)]
    #[case("' True '", Some(true))]
    #[case("' fAlSe '", Some(false))]
    #[case("'yes'", None)]
    #[case("'1'", None)]
    #[case("'unknown'", None)]
    fn string_boolean_tokens_remain_separate_from_truthiness(
        #[case] source: &str,
        #[case] expected: Option<bool>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            assert_eq!(
                Field::new("test", "flag", evaluate(py, source))
                    .str_bool()
                    .unwrap(),
                expected
            );
        });
    }

    #[test]
    fn protocol_errors_preserve_exception_identity_traceback_cause_and_context() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"
failure = LookupError('protocol failed')
cause = ValueError('cause')
context = RuntimeError('context')
def fail():
    try:
        raise context
    except RuntimeError:
        raise failure from cause
class Bool:
    def __bool__(self): return fail()
class Length:
    def __len__(self): return fail()
class Iter:
    def __iter__(self): return fail()
class Next:
    def __iter__(self): return self
    def __next__(self): return fail()
class Descriptor:
    @property
    def flag(self): return fail()
values = (Bool(), Length(), Iter(), Next(), [Bool()])
descriptor = Descriptor()
",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let values = locals.get_item("values").unwrap().unwrap();
            for value in values.try_iter().unwrap() {
                let error = Field::new("test", "flag", value.unwrap())
                    .string_collection()
                    .err()
                    .unwrap();
                let error = PyErr::from(error);
                assert!(
                    error
                        .value(py)
                        .is(locals.get_item("failure").unwrap().unwrap())
                );
                assert!(error.is_instance_of::<PyLookupError>(py));
                assert!(error.traceback(py).is_some());
                assert!(
                    error
                        .value(py)
                        .getattr("__cause__")
                        .unwrap()
                        .is(locals.get_item("cause").unwrap().unwrap())
                );
                assert!(
                    error
                        .value(py)
                        .getattr("__context__")
                        .unwrap()
                        .is(locals.get_item("context").unwrap().unwrap())
                );
            }
            let error = Field::read(
                &locals.get_item("descriptor").unwrap().unwrap(),
                "test",
                "flag",
            )
            .err()
            .unwrap();
            assert!(
                PyErr::from(error)
                    .value(py)
                    .is(locals.get_item("failure").unwrap().unwrap())
            );
        });
    }

    #[test]
    fn identity_and_string_contents_do_not_invoke_unrelated_protocols() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"
class Hostile:
    def __bool__(self): raise AssertionError('bool called')
    def __eq__(self, other): raise AssertionError('eq called')
    def __str__(self): raise AssertionError('str called')
class Text(str):
    def __str__(self): raise AssertionError('str called')
    def strip(self): raise AssertionError('strip called')
    def lower(self): raise AssertionError('lower called')
hostile = Hostile()
text = Text(' False ')
",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let hostile = Field::new("test", "flag", locals.get_item("hostile").unwrap().unwrap());
            assert!(!hostile.exact_true());
            assert!(matches!(
                hostile.strict_string(),
                Err(ProjectionError::InvalidConfiguration(_))
            ));
            let text = Field::new("test", "flag", locals.get_item("text").unwrap().unwrap());
            assert_eq!(text.strict_string().unwrap(), " False ");
            assert_eq!(text.str_bool().unwrap(), Some(false));
        });
    }

    #[test]
    fn missing_snapshot_fields_and_descriptor_attribute_errors_are_distinct() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"
failure = AttributeError('descriptor failed')
class Snapshot:
    @property
    def flag(self): raise failure
snapshot = Snapshot()
class Dynamic:
    def __getattr__(self, name): raise failure
class Intercepted:
    def __getattribute__(self, name): raise failure
dynamic = Dynamic()
intercepted = Intercepted()
",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let snapshot = locals.get_item("snapshot").unwrap().unwrap();
            let descriptor = PyErr::from(Field::read(&snapshot, "test", "flag").err().unwrap());
            assert!(
                descriptor
                    .value(py)
                    .is(locals.get_item("failure").unwrap().unwrap())
            );
            for name in ["dynamic", "intercepted"] {
                let value = locals.get_item(name).unwrap().unwrap();
                let error = PyErr::from(Field::read(&value, "test", "flag").err().unwrap());
                assert!(
                    error
                        .value(py)
                        .is(locals.get_item("failure").unwrap().unwrap())
                );
            }
            let missing = PyErr::from(Field::read(&snapshot, "test", "missing").err().unwrap());
            assert!(missing.is_instance_of::<PyRuntimeError>(py));
            assert!(missing.to_string().contains("test.missing"));
        });
    }

    #[test]
    fn configuration_errors_name_fields_without_exposing_values() {
        Python::initialize();
        Python::attach(|py| {
            for source in [
                "{'secret': 'do-not-print'}",
                "['host.test', {'secret': 'do-not-print'}]",
            ] {
                let field = Field::new("test", "setting", evaluate(py, source));
                let error = PyErr::from(field.falsy_optional_string().err().unwrap());
                assert!(error.is_instance_of::<PyValueError>(py));
                assert!(error.to_string().contains("test.setting"));
                assert!(!error.to_string().contains("do-not-print"));
            }
            let hosts = Field::new(
                "url_policy",
                "user_url_allowed_hosts",
                evaluate(py, "['host.test', 1]"),
            );
            assert!(matches!(
                hosts.string_collection(),
                Err(ProjectionError::InvalidConfiguration(_))
            ));
            assert!(matches!(
                Field::new("test", "flag", evaluate(py, "1")).str_bool(),
                Err(ProjectionError::InvalidConfiguration(_))
            ));
        });
    }
}
