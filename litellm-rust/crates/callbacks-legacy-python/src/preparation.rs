use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};

use crate::python::Wrapper;

struct CredentialEntry<'py>(Bound<'py, PyAny>);

impl<'py> CredentialEntry<'py> {
    fn name(&self) -> PyResult<String> {
        self.0.getattr("credential_name")?.extract()
    }

    fn values(&self) -> PyResult<Bound<'py, PyDict>> {
        Ok(self.0.getattr("credential_values")?.cast_into::<PyDict>()?)
    }
}

pub fn prepare<'py>(
    py: Python<'py>,
    kwargs: &Bound<'py, PyDict>,
    logger: &crate::PythonLogger,
) -> PyResult<Bound<'py, PyDict>> {
    let arguments = kwargs.copy()?;
    arguments.set_item("litellm_logging_obj", logger.object(py))?;
    inherit_credentials(py, &arguments, || {
        Ok(Wrapper::CredentialList
            .call(py, ())?
            .cast_into::<PyList>()?)
    })?;
    check_limits(py, &arguments)?;
    Ok(arguments)
}

fn check_limits(py: Python<'_>, kwargs: &Bound<'_, PyDict>) -> PyResult<()> {
    let sdk = py.import("litellm")?;
    check_limits_with_settings(py, kwargs, sdk.as_any())
}

fn check_limits_with_settings(
    py: Python<'_>,
    kwargs: &Bound<'_, PyDict>,
    sdk: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let max_budget = sdk.getattr("max_budget")?.extract::<Option<f64>>()?;
    if max_budget.is_some_and(|limit| limit != 0.0) {
        let current_cost = sdk.getattr("_current_cost")?.extract::<f64>()?;
        if let Err(error) = litellm_core_utils::budget::check_budget(current_cost, max_budget) {
            let arguments = PyDict::new(py);
            arguments.set_item("current_cost", error.current_cost)?;
            arguments.set_item("max_budget", error.max_budget)?;
            return Err(PyErr::from_value(
                sdk.getattr("BudgetExceededError")?
                    .call((), Some(&arguments))?,
            ));
        }
    }
    let limit = sdk.getattr("num_retries_per_request")?;
    if limit.is_none() {
        return Ok(());
    }
    let key = if kwargs.contains("litellm_metadata")? {
        "litellm_metadata"
    } else {
        "metadata"
    };
    let Some(metadata) = kwargs.get_item(key)? else {
        return Ok(());
    };
    if !metadata.is_instance(&py.import("collections.abc")?.getattr("Mapping")?)? {
        return Ok(());
    }
    let retry = metadata.call_method1("get", ("request_retry_count",))?;
    if retry.get_type().is(&py.get_type::<pyo3::types::PyInt>())
        && retry.gt(0)?
        && retry.ge(limit)?
    {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Max retries per request hit!",
        ));
    }
    Ok(())
}

fn inherit_credentials<'py>(
    py: Python<'py>,
    arguments: &Bound<'py, PyDict>,
    credential_list: impl FnOnce() -> PyResult<Bound<'py, PyList>>,
) -> PyResult<()> {
    let Some(requested) = arguments
        .get_item("litellm_credential_name")?
        .filter(|value| !value.is_none())
    else {
        return Ok(());
    };
    if !requested.is_truthy()? {
        return Ok(());
    }
    let requested: String = requested.extract()?;
    let credentials = credential_list()?;
    let names = credentials
        .iter()
        .map(|credential| CredentialEntry(credential).name())
        .collect::<PyResult<Vec<_>>>()?;
    let Some(index) = names.iter().position(|name| *name == requested) else {
        Wrapper::WarnUnknownCredential.call(py, (requested, names.len()))?;
        return Ok(());
    };
    let selected = CredentialEntry(credentials.get_item(index)?);
    let values = selected.values()?;
    let supplied: Vec<String> = arguments.keys().extract()?;
    let fields: Vec<String> = values.keys().extract()?;
    for name in fields.iter().filter(|name| !supplied.contains(name)) {
        if let Some(value) = values.get_item(name.as_str())? {
            arguments.set_item(name.as_str(), value)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_limits_preserve_budget_and_retry_boundaries() {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"
import types
class BudgetExceededError(Exception):
    def __init__(self, current_cost, max_budget):
        self.current_cost = current_cost
        self.max_budget = max_budget
settings = types.SimpleNamespace(max_budget=2.0, _current_cost=2.0, num_retries_per_request=5,
    BudgetExceededError=BudgetExceededError)
",
            );
            let settings = locals.get_item("settings").unwrap().unwrap();
            let arguments = PyDict::new(py);
            assert!(check_limits_with_settings(py, &arguments, &settings).is_ok());
            settings.setattr("_current_cost", 3.0).unwrap();
            let error = check_limits_with_settings(py, &arguments, &settings).unwrap_err();
            assert_eq!(
                error
                    .value(py)
                    .getattr("current_cost")
                    .unwrap()
                    .extract::<f64>()
                    .unwrap(),
                3.0
            );
            settings.setattr("max_budget", 0.0).unwrap();
            for key in ["metadata", "litellm_metadata"] {
                for (count, refused) in [(0, false), (4, false), (5, true), (6, true)] {
                    let metadata = PyDict::new(py);
                    metadata.set_item("request_retry_count", count).unwrap();
                    arguments.set_item(key, metadata).unwrap();
                    assert_eq!(
                        check_limits_with_settings(py, &arguments, &settings).is_err(),
                        refused
                    );
                }
                let metadata = PyDict::new(py);
                metadata.set_item("request_retry_count", true).unwrap();
                settings.setattr("num_retries_per_request", 0).unwrap();
                arguments.set_item(key, metadata).unwrap();
                assert!(check_limits_with_settings(py, &arguments, &settings).is_ok());
                arguments.del_item(key).unwrap();
                settings.setattr("num_retries_per_request", 5).unwrap();
            }
        });
    }

    fn eval<'py>(py: Python<'py>, source: &std::ffi::CStr) -> Bound<'py, PyDict> {
        let locals = PyDict::new(py);
        py.run(source, Some(&locals), Some(&locals)).unwrap();
        locals
    }

    fn inherit(py: Python<'_>, locals: &Bound<'_, PyDict>) -> PyResult<()> {
        inherit_credentials(
            py,
            &locals
                .get_item("arguments")
                .unwrap()
                .unwrap()
                .cast_into::<PyDict>()?,
            || {
                Ok(locals
                    .get_item("credentials")?
                    .unwrap()
                    .cast_into::<PyList>()?)
            },
        )
    }

    #[test]
    fn duplicate_names_select_the_first_entry_without_reading_other_values() {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"
accesses = []
class Credential:
    def __init__(self, name, values):
        self._name = name
        self._values = values
    @property
    def credential_name(self):
        accesses.append(('name', self._name))
        return self._name
    @property
    def credential_values(self):
        accesses.append(('values', self._name))
        return self._values
credentials = [
    Credential('ocr-test', {'api_key': 'first'}),
    Credential('other', {'api_key': 'unused'}),
    Credential('ocr-test', {'api_key': 'later'}),
]
arguments = {'litellm_credential_name': 'ocr-test'}
",
            );
            inherit(py, &locals).unwrap();
            let arguments = locals
                .get_item("arguments")
                .unwrap()
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            assert_eq!(
                arguments
                    .get_item("api_key")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "first"
            );
            let accesses: Vec<(String, String)> = locals
                .get_item("accesses")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(
                accesses,
                [
                    ("name".into(), "ocr-test".into()),
                    ("name".into(), "other".into()),
                    ("name".into(), "ocr-test".into()),
                    ("values".into(), "ocr-test".into()),
                ]
            );
        });
    }

    #[test]
    fn later_invalid_name_still_fails_after_an_earlier_match() {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"
failure = LookupError('later name')
class Good:
    credential_name = 'ocr-test'
    credential_values = {'api_key': 'first'}
class Bad:
    @property
    def credential_name(self):
        raise failure
credentials = [Good(), Bad()]
arguments = {'litellm_credential_name': 'ocr-test'}
",
            );
            let error = inherit(py, &locals).unwrap_err();
            assert!(
                error
                    .value(py)
                    .is(locals.get_item("failure").unwrap().unwrap())
            );
        });
    }

    #[test]
    fn selected_values_must_be_a_dictionary_and_property_errors_keep_identity() {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"
class Listed:
    credential_name = 'ocr-test'
    credential_values = ['not-a-dict']
credentials = [Listed()]
arguments = {'litellm_credential_name': 'ocr-test'}
",
            );
            assert!(
                inherit(py, &locals)
                    .unwrap_err()
                    .is_instance_of::<pyo3::exceptions::PyTypeError>(py)
            );

            let locals = eval(
                py,
                c"
failure = RuntimeError('values failed')
class Broken:
    credential_name = 'ocr-test'
    @property
    def credential_values(self):
        raise failure
credentials = [Broken()]
arguments = {'litellm_credential_name': 'ocr-test'}
",
            );
            let error = inherit(py, &locals).unwrap_err();
            assert!(
                error
                    .value(py)
                    .is(locals.get_item("failure").unwrap().unwrap())
            );
        });
    }

    #[test]
    fn explicit_none_is_not_overwritten_and_inherited_objects_keep_identity() {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"
opaque = object()
class Credential:
    credential_name = 'ocr-test'
    credential_values = {'api_key': 'credential-key', 'opaque': opaque}
credentials = [Credential()]
arguments = {'litellm_credential_name': 'ocr-test', 'api_key': None}
",
            );
            inherit(py, &locals).unwrap();
            let arguments = locals
                .get_item("arguments")
                .unwrap()
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            assert!(arguments.get_item("api_key").unwrap().unwrap().is_none());
            assert!(
                arguments
                    .get_item("opaque")
                    .unwrap()
                    .unwrap()
                    .is(locals.get_item("opaque").unwrap().unwrap())
            );
        });
    }

    #[test]
    fn selection_rereads_the_list_after_name_properties_run() {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"
class First:
    @property
    def credential_name(self):
        credentials[0] = Second()
        return 'ocr-test'
    credential_values = {'api_key': 'first'}
class Second:
    credential_name = 'ocr-test'
    credential_values = {'api_key': 'replaced'}
credentials = [First()]
arguments = {'litellm_credential_name': 'ocr-test'}
",
            );
            inherit(py, &locals).unwrap();
            let arguments = locals
                .get_item("arguments")
                .unwrap()
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            assert_eq!(
                arguments
                    .get_item("api_key")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "replaced"
            );
        });
    }

    #[test]
    fn falsy_credential_names_return_before_loading_credentials() {
        Python::initialize();
        Python::attach(|py| {
            for name in [py.None(), py.eval(c"''", None, None).unwrap().unbind()] {
                let arguments = PyDict::new(py);
                arguments.set_item("litellm_credential_name", name).unwrap();
                inherit_credentials(py, &arguments, || panic!("credentials must not be loaded"))
                    .unwrap();
            }
        });
    }
}
