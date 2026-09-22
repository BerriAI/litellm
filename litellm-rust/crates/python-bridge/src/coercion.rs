use std::collections::BTreeSet;

use litellm_core_utils::serde_compat::{parse_redis_bool, parse_str_bool};
use litellm_http::SslVerify;
use pyo3::{
    exceptions::{PyAttributeError, PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBool, PyDict, PyInt, PyString},
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

pub(crate) struct Truthy(pub bool);
pub(crate) struct ExactTrue(pub bool);
pub(crate) struct StrBool(pub Option<bool>);
pub(crate) struct OptionalStrictString(pub Option<String>);
pub(crate) struct FalsyOptionalString(pub Option<String>);
pub(crate) struct TuningString(pub Option<String>);
pub(crate) struct StringCollection(pub Vec<String>);
pub(crate) struct SslVerifyInput(pub Option<SslVerify>);
/// `redis-py` Boolean coercion: string tokens use the Redis parser, everything else `bool(value)`.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "cache configuration projection adopts this adapter in a later commit"
    )
)]
pub(crate) struct OptionalRedisBool(pub Option<bool>);
/// `None` stays absent, an empty string also stays absent, another type is invalid.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "cache configuration projection adopts this adapter in a later commit"
    )
)]
pub(crate) struct NonEmptyOptionalString(pub Option<String>);

/// `ssl_cert_reqs` as `redis-py` accepts it: an `ssl` constant or a case-insensitive token.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "cache configuration projection adopts this adapter in a later commit"
    )
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CertificateRequirement {
    None,
    Optional,
    Required,
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

    /// Reads `values[key]`; an absent key is `None`, a present `None` value is a field.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "cache configuration projection adopts this adapter in a later commit"
        )
    )]
    pub(crate) fn item(
        values: &Bound<'py, PyDict>,
        group: &'static str,
        key: &'static str,
    ) -> Result<Option<Self>, ProjectionError> {
        Ok(values
            .get_item(key)?
            .map(|value| Self::new(group, key, value)))
    }

    /// Reads `values[key]` for a key the consumer requires.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "cache configuration projection adopts this adapter in a later commit"
        )
    )]
    pub(crate) fn required_item(
        values: &Bound<'py, PyDict>,
        group: &'static str,
        key: &'static str,
    ) -> Result<Self, ProjectionError> {
        Self::item(values, group, key)?.ok_or_else(|| {
            ProjectionError::InvalidConfiguration(format!("{group}.{key}: missing required value"))
        })
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

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "cache configuration projection adopts this adapter in a later commit"
        )
    )]
    pub(crate) fn value(&self) -> &Bound<'py, PyAny> {
        &self.value
    }

    /// A member of this field's collection, reported under the same path.
    pub(crate) fn member(&self, value: Bound<'py, PyAny>) -> Self {
        Self::new(self.group, self.name, value)
    }

    fn expected(&self, expected: &str) -> Result<String, ProjectionError> {
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

    pub(crate) fn truthy(&self) -> Result<Truthy, ProjectionError> {
        Ok(Truthy(self.value.is_truthy()?))
    }

    pub(crate) fn exact_true(&self) -> ExactTrue {
        ExactTrue(self.value.is(PyBool::new(self.value.py(), true)))
    }

    #[cfg(test)]
    pub(crate) fn schema_bool(&self) -> Result<bool, ProjectionError> {
        if !self.value.is_instance_of::<PyBool>() {
            return Err(ProjectionError::InternalSchemaFailure(
                self.expected("a Boolean")?,
            ));
        }
        Ok(self.exact_true().0)
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

    pub(crate) fn str_bool(&self) -> Result<StrBool, ProjectionError> {
        if self.value.is_none() {
            return Ok(StrBool(None));
        }
        Ok(StrBool(parse_str_bool(&self.strict_string()?)))
    }

    pub(crate) fn optional_strict_string(&self) -> Result<OptionalStrictString, ProjectionError> {
        if self.value.is_none() {
            return Ok(OptionalStrictString(None));
        }
        self.strict_string().map(Some).map(OptionalStrictString)
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "cache configuration projection adopts this adapter in a later commit"
        )
    )]
    pub(crate) fn non_empty_string(&self) -> Result<NonEmptyOptionalString, ProjectionError> {
        Ok(NonEmptyOptionalString(
            self.optional_strict_string()?
                .0
                .filter(|value| !value.is_empty()),
        ))
    }

    pub(crate) fn falsy_optional_string(&self) -> Result<FalsyOptionalString, ProjectionError> {
        if !self.truthy()?.0 {
            return Ok(FalsyOptionalString(None));
        }
        self.strict_string().map(Some).map(FalsyOptionalString)
    }

    pub(crate) fn tuning_string(&self) -> Result<TuningString, ProjectionError> {
        if !self.truthy()?.0 || !self.value.is_instance_of::<PyString>() {
            return Ok(TuningString(None));
        }
        self.strict_string().map(Some).map(TuningString)
    }

    pub(crate) fn string_collection(&self) -> Result<StringCollection, ProjectionError> {
        if !self.truthy()?.0 {
            return Ok(StringCollection(Vec::new()));
        }
        if self.value.is_instance_of::<PyString>() {
            return self
                .strict_string()
                .map(|value| StringCollection(vec![value]));
        }
        let values = self
            .value
            .try_iter()?
            .filter_map(|item| {
                let member = match item {
                    Ok(value) => self.member(value),
                    Err(error) => return Some(Err(error.into())),
                };
                match member.truthy() {
                    Ok(Truthy(false)) => None,
                    Ok(Truthy(true)) => Some(member.strict_string()),
                    Err(error) => Some(Err(error)),
                }
            })
            .collect::<Result<Vec<_>, ProjectionError>>()?;
        Ok(StringCollection(values))
    }

    pub(crate) fn optional_string_collection(
        &self,
    ) -> Result<Option<StringCollection>, ProjectionError> {
        if self.value.is_none() {
            return Ok(None);
        }
        self.string_collection().map(Some)
    }

    pub(crate) fn host_collection(&self) -> Result<StringCollection, ProjectionError> {
        let values = self
            .string_collection()?
            .0
            .into_iter()
            .map(|host| litellm_http::media::normalize_host(&host))
            .collect::<BTreeSet<_>>();
        Ok(StringCollection(values.into_iter().collect()))
    }

    pub(crate) fn ssl_verify(&self) -> Result<SslVerifyInput, ProjectionError> {
        if self.value.is_none() {
            return Ok(SslVerifyInput(None));
        }
        if self.value.is_instance_of::<PyBool>() {
            return Ok(SslVerifyInput(Some(if self.exact_true().0 {
                SslVerify::Enabled
            } else {
                SslVerify::Disabled
            })));
        }
        if self.value.is_instance_of::<PyString>() {
            let parsed = match self.str_bool()?.0 {
                Some(true) => SslVerify::Enabled,
                Some(false) => SslVerify::Disabled,
                None => SslVerify::CaBundle(self.strict_string()?.into()),
            };
            return Ok(SslVerifyInput(Some(parsed)));
        }
        let context = self.value.py().import("ssl")?.getattr("SSLContext")?;
        if self.value.is_instance(&context)? {
            return Err(ProjectionError::UnsupportedLiveObject(self.expected(
                "a Boolean, Boolean string, CA path, or None; live SSLContext is unsupported",
            )?));
        }
        Err(self.invalid("a Boolean, Boolean string, CA path, or None"))
    }

    /// A live Python object whose behavior stays in Python; `None` means no binding.
    pub(crate) fn python_binding(&self) -> Option<Py<PyAny>> {
        (!self.value.is_none()).then(|| self.value.clone().unbind())
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "cache configuration projection adopts this adapter in a later commit"
        )
    )]
    pub(crate) fn optional_redis_bool(&self) -> Result<OptionalRedisBool, ProjectionError> {
        if self.value.is_none() {
            return Ok(OptionalRedisBool(None));
        }
        if self.value.is_instance_of::<PyString>() {
            return Ok(OptionalRedisBool(Some(parse_redis_bool(
                &self.strict_string()?,
            ))));
        }
        Ok(OptionalRedisBool(Some(self.truthy()?.0)))
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "cache configuration projection adopts this adapter in a later commit"
        )
    )]
    pub(crate) fn redis_cert_reqs(&self) -> Result<CertificateRequirement, ProjectionError> {
        if self.value.is_none() {
            return Ok(CertificateRequirement::Required);
        }
        if self.value.is_instance_of::<PyInt>() {
            return match self.value.extract::<i64>()? {
                0 => Ok(CertificateRequirement::None),
                1 => Ok(CertificateRequirement::Optional),
                2 => Ok(CertificateRequirement::Required),
                _ => Err(self.invalid("ssl.CERT_NONE, ssl.CERT_OPTIONAL, or ssl.CERT_REQUIRED")),
            };
        }
        let text = self.value.str()?;
        let text = text.to_str()?;
        if text.eq_ignore_ascii_case("none") || text.eq_ignore_ascii_case("cert_none") {
            return Ok(CertificateRequirement::None);
        }
        if text.eq_ignore_ascii_case("optional") || text.eq_ignore_ascii_case("cert_optional") {
            return Ok(CertificateRequirement::Optional);
        }
        if text.eq_ignore_ascii_case("required") || text.eq_ignore_ascii_case("cert_required") {
            return Ok(CertificateRequirement::Required);
        }
        Err(self.invalid("a certificate requirement token"))
    }

    /// `None` stays absent; another value must be a Python `bool`.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "cache configuration projection adopts this adapter in a later commit"
        )
    )]
    pub(crate) fn optional_bool(&self) -> Result<Option<bool>, ProjectionError> {
        if self.value.is_none() {
            return Ok(None);
        }
        if !self.value.is_instance_of::<PyBool>() {
            return Err(self.invalid("a Boolean"));
        }
        Ok(Some(self.exact_true().0))
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "cache configuration projection adopts this adapter in a later commit"
        )
    )]
    pub(crate) fn optional_i64(&self) -> Result<Option<i64>, ProjectionError> {
        if self.value.is_none() {
            return Ok(None);
        }
        self.value
            .extract::<i64>()
            .map(Some)
            .map_err(|_| self.invalid("an integer"))
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "cache configuration projection adopts this adapter in a later commit"
        )
    )]
    pub(crate) fn required_i64(&self) -> Result<i64, ProjectionError> {
        self.value
            .extract::<i64>()
            .map_err(|_| self.invalid("an integer"))
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "cache configuration projection adopts this adapter in a later commit"
        )
    )]
    pub(crate) fn optional_f64(&self) -> Result<Option<f64>, ProjectionError> {
        if self.value.is_none() {
            return Ok(None);
        }
        self.value
            .extract::<f64>()
            .map(Some)
            .map_err(|_| self.invalid("a number"))
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
            assert_eq!(field.truthy().unwrap().0, truth);
            assert_eq!(field.exact_true().0, exact);
            assert_eq!(
                field.truthy().unwrap().0,
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
                field
                    .optional_strict_string()
                    .map(|value| value.0)
                    .map_err(|_| ()),
                owned(strict)
            );
            assert_eq!(
                field
                    .falsy_optional_string()
                    .map(|value| value.0)
                    .map_err(|_| ()),
                owned(fallback)
            );
            assert_eq!(
                field.tuning_string().map(|value| value.0).map_err(|_| ()),
                owned(tuning)
            );
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
                    .unwrap()
                    .0,
                expected
            );
        });
    }

    #[rstest]
    #[case("'EXAMPLE.TEST.'", vec!["example.test"])]
    #[case("['B.test', '', None, 0, [], 'A.test.', 'b.test']", vec!["a.test", "b.test"])]
    #[case("('B.test', 'a.test')", vec!["a.test", "b.test"])]
    #[case("{'B.test', 'a.test'}", vec!["a.test", "b.test"])]
    #[case("(host for host in ['B.test', 'a.test'])", vec!["a.test", "b.test"])]
    #[case("None", vec![])]
    #[case("False", vec![])]
    fn host_collection_is_owned_normalized_and_deterministic(
        #[case] source: &str,
        #[case] expected: Vec<&str>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            assert_eq!(
                Field::new("url_policy", "user_url_allowed_hosts", evaluate(py, source))
                    .host_collection()
                    .unwrap()
                    .0,
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
                    .host_collection()
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
            assert!(!hostile.exact_true().0);
            assert!(matches!(
                hostile.strict_string(),
                Err(ProjectionError::InvalidConfiguration(_))
            ));
            let text = Field::new("test", "flag", locals.get_item("text").unwrap().unwrap());
            assert_eq!(text.strict_string().unwrap(), " False ");
            assert_eq!(text.str_bool().unwrap().0, Some(false));
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
                hosts.host_collection(),
                Err(ProjectionError::InvalidConfiguration(_))
            ));
            assert!(matches!(
                Field::new("test", "flag", evaluate(py, "1")).str_bool(),
                Err(ProjectionError::InvalidConfiguration(_))
            ));
        });
    }

    #[test]
    fn projection_releases_the_source_collection() {
        Python::initialize();
        Python::attach(|py| {
            let source = evaluate(py, "['A.test']");
            let projected = Field::new("test", "hosts", source.clone())
                .host_collection()
                .unwrap()
                .0;
            source.call_method1("append", ("b.test",)).unwrap();
            assert_eq!(projected, ["a.test"]);
            assert_eq!(
                Field::new("test", "hosts", source)
                    .host_collection()
                    .unwrap()
                    .0,
                ["a.test", "b.test"]
            );
        });
    }

    #[rstest]
    #[case("True", Some(true))]
    #[case("False", Some(false))]
    #[case("1", None)]
    #[case("None", None)]
    #[case("[]", None)]
    fn accessor_booleans_are_strict_schema_values(
        #[case] source: &str,
        #[case] expected: Option<bool>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let result =
                Field::new("secret_manager", "readable", evaluate(py, source)).schema_bool();
            match expected {
                Some(expected) => assert_eq!(result.unwrap(), expected),
                None => {
                    let error = PyErr::from(result.unwrap_err());
                    assert!(error.is_instance_of::<PyRuntimeError>(py));
                    assert!(error.to_string().contains("secret_manager.readable"));
                }
            }
        });
    }

    #[rstest]
    #[case("None", None)]
    #[case("'true'", Some(true))]
    #[case("'YES'", Some(true))]
    #[case("'1'", Some(true))]
    #[case("' true '", Some(false))]
    #[case("'false'", Some(false))]
    #[case("'unknown'", Some(false))]
    #[case("'0'", Some(false))]
    #[case("True", Some(true))]
    #[case("1", Some(true))]
    #[case("0", Some(false))]
    #[case("[]", Some(false))]
    #[case("[0]", Some(true))]
    fn redis_booleans_parse_string_tokens_and_fall_back_to_truthiness(
        #[case] source: &str,
        #[case] expected: Option<bool>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            assert_eq!(
                Field::new("cache", "retry_on_timeout", evaluate(py, source))
                    .optional_redis_bool()
                    .unwrap()
                    .0,
                expected
            );
        });
    }

    #[rstest]
    #[case("None", Some(CertificateRequirement::Required))]
    #[case("0", Some(CertificateRequirement::None))]
    #[case("1", Some(CertificateRequirement::Optional))]
    #[case("2", Some(CertificateRequirement::Required))]
    #[case("True", Some(CertificateRequirement::Optional))]
    #[case("3", None)]
    #[case("'none'", Some(CertificateRequirement::None))]
    #[case("'CERT_OPTIONAL'", Some(CertificateRequirement::Optional))]
    #[case("'Required'", Some(CertificateRequirement::Required))]
    #[case("'bogus'", None)]
    #[case("__import__('ssl').CERT_NONE", Some(CertificateRequirement::None))]
    fn certificate_requirements_accept_constants_and_tokens(
        #[case] source: &str,
        #[case] expected: Option<CertificateRequirement>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let result =
                Field::new("cache", "ssl_cert_reqs", evaluate(py, source)).redis_cert_reqs();
            match expected {
                Some(expected) => assert_eq!(result.unwrap(), expected),
                None => {
                    let error = PyErr::from(result.unwrap_err());
                    assert!(error.is_instance_of::<PyValueError>(py));
                    assert!(error.to_string().contains("cache.ssl_cert_reqs"));
                }
            }
        });
    }

    #[test]
    fn certificate_requirement_stringification_preserves_the_python_exception() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"
failure = LookupError('str failed')
class Requirement:
    def __str__(self): raise failure
value = Requirement()
",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let value = locals.get_item("value").unwrap().unwrap();
            let error = PyErr::from(
                Field::new("cache", "ssl_cert_reqs", value)
                    .redis_cert_reqs()
                    .unwrap_err(),
            );
            assert!(
                error
                    .value(py)
                    .is(locals.get_item("failure").unwrap().unwrap())
            );
        });
    }

    #[rstest]
    #[case("None", Ok(None))]
    #[case("''", Ok(None))]
    #[case("'bucket'", Ok(Some("bucket")))]
    #[case("1", Err(()))]
    #[case("[]", Err(()))]
    fn non_empty_strings_treat_empty_as_absent_and_reject_other_types(
        #[case] source: &str,
        #[case] expected: Result<Option<&str>, ()>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            assert_eq!(
                Field::new("cache", "bucket_name", evaluate(py, source))
                    .non_empty_string()
                    .map(|value| value.0)
                    .map_err(|_| ()),
                expected.map(|value| value.map(str::to_owned))
            );
        });
    }

    #[rstest]
    #[case("None", Ok(None))]
    #[case("True", Ok(Some(true)))]
    #[case("False", Ok(Some(false)))]
    #[case("1", Err(()))]
    #[case("'true'", Err(()))]
    fn optional_booleans_are_strict(
        #[case] source: &str,
        #[case] expected: Result<Option<bool>, ()>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            assert_eq!(
                Field::new("cache", "ssl", evaluate(py, source))
                    .optional_bool()
                    .map_err(|_| ()),
                expected
            );
        });
    }

    #[test]
    fn dictionary_items_distinguish_absent_keys_from_none_values() {
        Python::initialize();
        Python::attach(|py| {
            let values = evaluate(py, "{'present': None, 'port': 6379}")
                .cast_into::<PyDict>()
                .unwrap();
            assert!(Field::item(&values, "cache", "absent").unwrap().is_none());
            let present = Field::item(&values, "cache", "present").unwrap().unwrap();
            assert!(present.value().is_none());
            assert_eq!(present.optional_i64().unwrap(), None);
            assert_eq!(present.optional_f64().unwrap(), None);
            assert_eq!(
                Field::required_item(&values, "cache", "port")
                    .unwrap()
                    .optional_f64()
                    .unwrap(),
                Some(6379.0)
            );
            assert_eq!(
                Field::required_item(&values, "cache", "port")
                    .unwrap()
                    .required_i64()
                    .unwrap(),
                6379
            );
            let missing = match Field::required_item(&values, "cache", "host") {
                Ok(_) => panic!("missing key must be an error"),
                Err(error) => PyErr::from(error),
            };
            assert!(missing.is_instance_of::<PyValueError>(py));
            assert!(missing.to_string().contains("cache.host"));
        });
    }
}
