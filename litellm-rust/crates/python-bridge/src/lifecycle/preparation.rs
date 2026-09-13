use litellm_core::auth::{credential_default_fields, credential_index};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

struct CredentialEntry<'py>(Bound<'py, PyAny>);

impl<'py> CredentialEntry<'py> {
    fn name(&self) -> PyResult<String> {
        self.0.getattr("credential_name")?.extract()
    }

    fn values(&self) -> PyResult<Bound<'py, PyDict>> {
        Ok(self.0.getattr("credential_values")?.cast_into::<PyDict>()?)
    }
}

pub(super) fn prepare<'py>(
    py: Python<'py>,
    kwargs: &Bound<'py, PyDict>,
    logger: &super::PythonLogger,
) -> PyResult<Bound<'py, PyDict>> {
    let arguments = kwargs.copy()?;
    arguments.set_item("litellm_logging_obj", logger.object(py))?;
    let litellm = py.import("litellm")?;
    inherit_credentials(py, &litellm, &arguments)?;
    py.import("litellm.rust_bridge.lifecycle")?
        .getattr("check_limits")?
        .call1((&arguments,))?;
    Ok(arguments)
}

fn inherit_credentials(
    py: Python<'_>,
    litellm: &Bound<'_, PyModule>,
    arguments: &Bound<'_, PyDict>,
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
    let credentials = litellm.getattr("credential_list")?.cast_into::<PyList>()?;
    let names = credentials
        .iter()
        .map(|credential| CredentialEntry(credential).name())
        .collect::<PyResult<Vec<_>>>()?;
    let Some(index) = credential_index(&requested, &names) else {
        py.import("litellm._logging")?.getattr("verbose_logger")?.call_method1(
            "warning",
            ("litellm_credential_name=%s matched none of the %d loaded credentials; the request runs without it", requested, names.len()),
        )?;
        return Ok(());
    };
    let selected = CredentialEntry(credentials.get_item(index)?);
    let values = selected.values()?;
    let supplied: Vec<String> = arguments.keys().extract()?;
    let fields: Vec<String> = values.keys().extract()?;
    for name in credential_default_fields(&supplied, &fields) {
        if let Some(value) = values.get_item(name)? {
            arguments.set_item(name, value)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval<'py>(py: Python<'py>, source: &std::ffi::CStr) -> Bound<'py, PyDict> {
        let locals = PyDict::new(py);
        py.run(source, Some(&locals), Some(&locals)).unwrap();
        locals
    }

    fn inherit(py: Python<'_>, locals: &Bound<'_, PyDict>) -> PyResult<()> {
        let litellm = PyModule::new(py, "credential_host")?;
        litellm.setattr(
            "credential_list",
            locals.get_item("credentials").unwrap().unwrap(),
        )?;
        inherit_credentials(
            py,
            &litellm,
            &locals
                .get_item("arguments")
                .unwrap()
                .unwrap()
                .cast_into::<PyDict>()?,
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
            let litellm = PyModule::new(py, "credential_host").unwrap();
            for name in [py.None(), py.eval(c"''", None, None).unwrap().unbind()] {
                let arguments = PyDict::new(py);
                arguments.set_item("litellm_credential_name", name).unwrap();
                inherit_credentials(py, &litellm, &arguments).unwrap();
            }
        });
    }
}
