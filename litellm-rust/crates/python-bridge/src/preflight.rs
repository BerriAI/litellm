//! The SDK's request policy the driver runs on every route's keyword view before the host
//! projects from it: credential-name inheritance from `litellm.credential_list`, then the
//! budget and retry-count limits. It is the `@client` prologue after `function_setup` and the
//! deployment hook, and belongs to no callback contract.

use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};
use strum::{IntoStaticStr, VariantArray};

const MODULE: &str = "litellm.rust_bridge.preflight";

/// The litellm globals the preflight still reads through Python. `preflight_contract.json`
/// pins each function's parameters on both sides.
#[derive(Clone, Copy, Debug, IntoStaticStr, PartialEq, Eq, VariantArray)]
pub(crate) enum PythonPreflight {
    #[strum(serialize = "credential_list")]
    CredentialList,
    #[strum(serialize = "warn_unknown_credential")]
    WarnUnknownCredential,
    #[strum(serialize = "check_limits")]
    CheckLimits,
}

impl PythonPreflight {
    fn call<'py, A>(self, py: Python<'py>, args: A) -> PyResult<Bound<'py, PyAny>>
    where
        A: pyo3::call::PyCallArgs<'py>,
    {
        py.import(MODULE)?.getattr(<&str>::from(self))?.call1(args)
    }
}

#[cfg(test)]
pub(crate) const PYTHON_CONTRACT: &str = include_str!("../preflight_contract.json");

/// Rewrites `arguments` in place, in the order the Python wrapper runs: credentials first,
/// so the limits see the same view the provider request is built from.
pub(crate) fn sdk_preflight(py: Python<'_>, arguments: &Bound<'_, PyDict>) -> PyResult<()> {
    inherit_credentials(py, arguments, || {
        Ok(PythonPreflight::CredentialList
            .call(py, ())?
            .cast_into::<PyList>()?)
    })?;
    PythonPreflight::CheckLimits.call(py, (arguments,))?;
    Ok(())
}

struct CredentialEntry<'py>(Bound<'py, PyAny>);

impl<'py> CredentialEntry<'py> {
    fn name(&self) -> PyResult<String> {
        self.0.getattr("credential_name")?.extract()
    }

    fn values(&self) -> PyResult<Bound<'py, PyDict>> {
        Ok(self.0.getattr("credential_values")?.cast_into::<PyDict>()?)
    }
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
        PythonPreflight::WarnUnknownCredential.call(py, (requested, names.len()))?;
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
    use std::collections::BTreeSet;
    use std::sync::Mutex;

    use super::*;
    use strum::VariantArray;

    /// Tests share one interpreter, and the stub module below is global state, so the
    /// tests that install it run one at a time.
    static PREFLIGHT_MODULE: Mutex<()> = Mutex::new(());

    /// A fresh stand-in for `litellm.rust_bridge.preflight` that records every call, then
    /// `script` run against it with the module bound as `preflight`.
    fn preflight_stubs<'py>(py: Python<'py>, script: &std::ffi::CStr) -> Bound<'py, PyDict> {
        let locals = PyDict::new(py);
        py.run(
            c"
import sys
import types

for name in ('litellm', 'litellm.rust_bridge'):
    sys.modules.setdefault(name, types.ModuleType(name))
preflight = types.ModuleType('litellm.rust_bridge.preflight')
preflight.warnings = []
preflight.checked = []
preflight.credential_list = lambda: []
preflight.warn_unknown_credential = lambda name, loaded: preflight.warnings.append((name, loaded))
preflight.check_limits = lambda kwargs: preflight.checked.append(kwargs)
sys.modules['litellm.rust_bridge.preflight'] = preflight
",
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
        py.run(script, Some(&locals), Some(&locals)).unwrap();
        locals
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

    #[test]
    fn every_borrowed_function_is_in_the_python_contract() {
        Python::initialize();
        Python::attach(|py| {
            let contract = litellm_host_python::json_loads(py, PYTHON_CONTRACT.as_bytes()).unwrap();
            let declared: BTreeSet<String> = contract
                .bind(py)
                .cast::<PyDict>()
                .unwrap()
                .keys()
                .extract()
                .map(|names: Vec<String>| names.into_iter().collect())
                .unwrap();
            let called: BTreeSet<String> = PythonPreflight::VARIANTS
                .iter()
                .map(|&function| <&str>::from(function).to_owned())
                .collect();
            assert_eq!(
                called.len(),
                PythonPreflight::VARIANTS.len(),
                "a function is borrowed twice"
            );
            assert_eq!(called, declared);
        });
    }

    #[test]
    fn an_unknown_name_is_reported_with_the_loaded_count_and_leaves_the_arguments_alone() {
        let _guard = PREFLIGHT_MODULE
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        Python::initialize();
        Python::attach(|py| {
            let locals = preflight_stubs(
                py,
                c"
class Credential:
    credential_name = 'listed'
    credential_values = {'api_key': 'listed-key'}
preflight.credential_list = lambda: [Credential(), Credential()]
arguments = {'litellm_credential_name': 'missing'}
",
            );
            sdk_preflight(py, &argument_dict(&locals)).unwrap();
            py.run(
                c"
assert arguments == {'litellm_credential_name': 'missing'}, arguments
assert preflight.warnings == [('missing', 2)], preflight.warnings
assert preflight.checked == [arguments]
",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
        });
    }

    #[test]
    fn limits_are_checked_on_the_arguments_after_credentials_are_inherited() {
        let _guard = PREFLIGHT_MODULE
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        Python::initialize();
        Python::attach(|py| {
            let locals = preflight_stubs(
                py,
                c"
class Credential:
    credential_name = 'ocr-test'
    credential_values = {'api_key': 'inherited'}
preflight.credential_list = lambda: [Credential()]
rejection = RuntimeError('Max retries per request hit!')
def check_limits(arguments):
    preflight.checked.append(dict(arguments))
    raise rejection
preflight.check_limits = check_limits
arguments = {'litellm_credential_name': 'ocr-test'}
",
            );
            let error = sdk_preflight(py, &argument_dict(&locals)).unwrap_err();
            assert!(
                error
                    .value(py)
                    .is(locals.get_item("rejection").unwrap().unwrap())
            );
            py.run(
                c"
assert preflight.checked == [{'litellm_credential_name': 'ocr-test', 'api_key': 'inherited'}], preflight.checked
assert arguments['api_key'] == 'inherited'
",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
        });
    }

    fn argument_dict<'py>(locals: &Bound<'py, PyDict>) -> Bound<'py, PyDict> {
        locals
            .get_item("arguments")
            .unwrap()
            .unwrap()
            .cast_into::<PyDict>()
            .unwrap()
    }
}
