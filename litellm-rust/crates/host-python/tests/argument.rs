use std::ffi::CStr;

use litellm_host_python::present;
use pyo3::{prelude::*, types::PyDict};
use rstest::rstest;

#[rstest]
#[case::missing(None, false)]
#[case::none(Some(c"None"), false)]
#[case::false_value(Some(c"False"), true)]
#[case::zero(Some(c"0"), true)]
#[case::empty_string(Some(c"''"), true)]
#[case::empty_list(Some(c"[]"), true)]
#[case::object(Some(c"object()"), true)]
fn present_preserves_every_supplied_value_except_none(
    #[case] expression: Option<&CStr>,
    #[case] expected: bool,
) {
    Python::initialize();
    Python::attach(|py| {
        let arguments = PyDict::new(py);
        if let Some(expression) = expression {
            arguments
                .set_item("value", py.eval(expression, None, None).unwrap())
                .unwrap();
        }

        let result = present(&arguments, "value").unwrap();

        assert_eq!(result.is_some(), expected);
        if let Some(value) = result {
            assert!(value.is(arguments.get_item("value").unwrap().unwrap()));
        }
    });
}
