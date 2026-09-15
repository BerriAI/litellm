use std::collections::HashSet;

use pyo3::prelude::*;
use pyo3::types::PyList;

pub(crate) struct RequestParamPolicy {
    sdk_reserved_param_names: HashSet<String>,
}

pub(crate) struct RouteParamSpec<'a> {
    pub bound: &'a [&'a str],
    pub consumed: &'a [&'a str],
}

impl RequestParamPolicy {
    pub(crate) fn extract(names: &Bound<'_, PyList>) -> PyResult<Self> {
        let names: Vec<String> = names.extract()?;
        Ok(Self {
            sdk_reserved_param_names: names.into_iter().collect(),
        })
    }

    pub(crate) fn includes(&self, name: &str, route: &RouteParamSpec<'_>) -> bool {
        route.consumed.contains(&name)
            || (!self.sdk_reserved_param_names.contains(name) && !route.bound.contains(&name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marshal::project_optional_fields;
    use pyo3::exceptions::PyTypeError;
    use pyo3::types::PyDict;
    use rstest::rstest;
    use serde_json::json;

    #[rstest]
    #[case::unknown("future", &[], true)]
    #[case::sdk_control("callbacks", &[], false)]
    #[case::bound_argument("document", &[], false)]
    #[case::consumed_control("callbacks", &["callbacks"], true)]
    #[case::consumed_bound_argument("document", &["document"], true)]
    #[case::overrides("extra_body", &[], true)]
    #[case::new_sdk_control("new_control", &[], false)]
    fn selection(#[case] name: &str, #[case] consumed: &[&str], #[case] expected: bool) {
        let policy = RequestParamPolicy {
            sdk_reserved_param_names: ["callbacks", "new_control", "callbacks"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
        };
        assert_eq!(
            policy.includes(
                name,
                &RouteParamSpec {
                    bound: &["document"],
                    consumed
                }
            ),
            expected
        );
    }

    #[test]
    fn registry_extraction_rejects_non_strings() {
        Python::initialize();
        Python::attach(|py| {
            let names = PyList::new(py, [42]).unwrap();
            let error = RequestParamPolicy::extract(&names).err().unwrap();
            assert!(error.is_instance_of::<PyTypeError>(py));
        });
    }

    #[rstest]
    #[case::ocr("document", &["document"], false)]
    #[case::chat("messages", &["messages"], false)]
    #[case::transcription("audio", &["audio"], false)]
    #[case::no_implicit_ocr_rule("document", &["messages"], true)]
    #[case::no_implicit_chat_rule("messages", &["audio"], true)]
    fn route_owns_bound_argument_names(
        #[case] name: &str,
        #[case] bound: &[&str],
        #[case] expected: bool,
    ) {
        let policy = RequestParamPolicy {
            sdk_reserved_param_names: HashSet::new(),
        };
        assert_eq!(
            policy.includes(
                name,
                &RouteParamSpec {
                    bound,
                    consumed: &[]
                }
            ),
            expected
        );
    }

    #[test]
    fn registry_is_read_at_projection_not_capture() {
        Python::initialize();
        Python::attach(|py| {
            let names = PyList::empty(py);
            let retained = names.clone().unbind();
            names.append("future").unwrap();
            let policy = RequestParamPolicy::extract(retained.bind(py)).unwrap();
            assert!(!policy.includes(
                "future",
                &RouteParamSpec {
                    bound: &[],
                    consumed: &[]
                }
            ));
        });
    }

    #[test]
    fn projection_skips_host_objects_and_preserves_unknown_values() {
        Python::initialize();
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            let host = py.eval(c"object()", None, None).unwrap();
            kwargs.set_item("callbacks", &host).unwrap();
            kwargs.set_item("future", py.None()).unwrap();
            kwargs.set_item("enabled", false).unwrap();
            let policy =
                RequestParamPolicy::extract(&PyList::new(py, ["callbacks"]).unwrap()).unwrap();
            assert_eq!(
                project_optional_fields(
                    &kwargs,
                    &RouteParamSpec {
                        bound: &[],
                        consumed: &[]
                    },
                    &policy
                )
                .unwrap(),
                json!({"future":null,"enabled":false})
                    .as_object()
                    .unwrap()
                    .clone()
            );
            assert!(kwargs.get_item("callbacks").unwrap().unwrap().is(&host));
            assert_eq!(kwargs.len(), 3);
            kwargs.set_item("unknown", &host).unwrap();
            let error = project_optional_fields(
                &kwargs,
                &RouteParamSpec {
                    bound: &[],
                    consumed: &[],
                },
                &policy,
            )
            .unwrap_err();
            assert!(error.is_instance_of::<PyTypeError>(py));
        });
    }
}
