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
        let field = Field::new("test.flag", value.clone());
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
        let field = Field::new("test.string", evaluate(py, source));
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
            Field::new("test.flag", evaluate(py, source))
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
            Field::new("url_policy.user_url_allowed_hosts", evaluate(py, source))
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
            let error = Field::new("test.flag", value.unwrap())
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
            "test.flag",
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
        let hostile = Field::new("test.flag", locals.get_item("hostile").unwrap().unwrap());
        assert!(!hostile.exact_true().0);
        assert!(matches!(
            hostile.strict_string(),
            Err(ProjectionError::InvalidConfiguration(_))
        ));
        let text = Field::new("test.flag", locals.get_item("text").unwrap().unwrap());
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
        let descriptor = PyErr::from(Field::read(&snapshot, "test.flag").err().unwrap());
        assert!(
            descriptor
                .value(py)
                .is(locals.get_item("failure").unwrap().unwrap())
        );
        for name in ["dynamic", "intercepted"] {
            let value = locals.get_item(name).unwrap().unwrap();
            let error = PyErr::from(Field::read(&value, "test.flag").err().unwrap());
            assert!(
                error
                    .value(py)
                    .is(locals.get_item("failure").unwrap().unwrap())
            );
        }
        let missing = PyErr::from(Field::read(&snapshot, "test.missing").err().unwrap());
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
            let field = Field::new("test.setting", evaluate(py, source));
            let error = PyErr::from(field.falsy_optional_string().err().unwrap());
            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(error.to_string().contains("test.setting"));
            assert!(!error.to_string().contains("do-not-print"));
        }
        let hosts = Field::new(
            "url_policy.user_url_allowed_hosts",
            evaluate(py, "['host.test', 1]"),
        );
        assert!(matches!(
            hosts.host_collection(),
            Err(ProjectionError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            Field::new("test.flag", evaluate(py, "1")).str_bool(),
            Err(ProjectionError::InvalidConfiguration(_))
        ));
    });
}

#[test]
fn projection_releases_the_source_collection() {
    Python::initialize();
    Python::attach(|py| {
        let source = evaluate(py, "['A.test']");
        let projected = Field::new("test.hosts", source.clone())
            .host_collection()
            .unwrap()
            .0;
        source.call_method1("append", ("b.test",)).unwrap();
        assert_eq!(projected, ["a.test"]);
        assert_eq!(
            Field::new("test.hosts", source)
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
        let result = Field::new("secret_manager.readable", evaluate(py, source)).schema_bool();
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
