use std::sync::Arc;

use litellm_secrets::{SecretManager, SecretManagerState};
use litellm_secrets_types::{AccessMode, KeyManagementSettings, KeyManagementSystem, SecretValue};
use pyo3::prelude::*;
use serde_json::Value;

use litellm_host_python::PythonContext;

use super::callback::PythonSecretManager;
use crate::{
    coercion::{Field, FieldSpec, ProjectionError},
    python_settings::{PythonSettings, Snapshot},
};

const SYSTEM: FieldSpec<Option<KeyManagementSystem>> =
    FieldSpec::new("system", parse_optional_system);
const ACCESS_MODE: FieldSpec<AccessMode> = FieldSpec::new("access_mode", parse_access_mode);
const HOSTED_KEYS: FieldSpec<Option<Vec<String>>> =
    FieldSpec::new("hosted_keys", |field| field.optional_string_collection());
const STORE_VIRTUAL_KEYS: FieldSpec<bool> =
    FieldSpec::new("store_virtual_keys", |field| field.truthy());
const PREFIX_FOR_STORED_VIRTUAL_KEYS: FieldSpec<String> =
    FieldSpec::new("prefix_for_stored_virtual_keys", |field| {
        field.strict_string()
    });
const PRIMARY_SECRET_NAME: FieldSpec<Option<String>> =
    FieldSpec::new("primary_secret_name", |field| field.falsy_optional_string());
const KMS_KEY_ID: FieldSpec<Option<String>> =
    FieldSpec::new("kms_key_id", |field| field.falsy_optional_string());
const CUSTOM_SECRET_MANAGER: FieldSpec<Option<String>> =
    FieldSpec::new("custom_secret_manager", |field| {
        field.falsy_optional_string()
    });
const AWS_REGION_NAME: FieldSpec<Option<String>> =
    FieldSpec::new("aws_region_name", |field| field.falsy_optional_string());
const AWS_ROLE_NAME: FieldSpec<Option<String>> =
    FieldSpec::new("aws_role_name", |field| field.falsy_optional_string());
const AWS_SESSION_NAME: FieldSpec<Option<String>> =
    FieldSpec::new("aws_session_name", |field| field.falsy_optional_string());
const AWS_EXTERNAL_ID: FieldSpec<Option<String>> =
    FieldSpec::new("aws_external_id", |field| field.falsy_optional_string());
const AWS_PROFILE_NAME: FieldSpec<Option<String>> =
    FieldSpec::new("aws_profile_name", |field| field.falsy_optional_string());
const AWS_WEB_IDENTITY_TOKEN: FieldSpec<Option<String>> =
    FieldSpec::new("aws_web_identity_token", |field| {
        field.falsy_optional_string()
    });
const AWS_STS_ENDPOINT: FieldSpec<Option<String>> =
    FieldSpec::new("aws_sts_endpoint", |field| field.falsy_optional_string());
const REPLICA_REGIONS: FieldSpec<Option<Vec<String>>> =
    FieldSpec::new("replica_regions", |field| {
        field.optional_string_collection()
    });
const CLIENT: FieldSpec<Option<Py<PyAny>>> =
    FieldSpec::new("client", |field| Ok(field.python_binding()));
const SETTINGS_OBJECT: FieldSpec<Option<Py<PyAny>>> =
    FieldSpec::new("settings_object", |field| Ok(field.python_binding()));

/// `litellm.secret_manager_client` as the bridge classifies it.
pub(crate) enum SecretManagerClient {
    /// `None`: reads come from the process environment.
    Local,
    /// A custom manager, legacy compatible client, or manually assigned SDK client that keeps
    /// executing in Python.
    PythonCallback(Py<PyAny>),
    Native(Box<SecretManager>),
}

impl std::fmt::Debug for SecretManagerClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Local => "Local",
            Self::Native(_) => "Native",
            Self::PythonCallback(_) => "PythonCallback",
        })
    }
}

/// One operation-local capture of the secret manager globals, taken while attached to Python.
#[derive(Debug)]
pub(crate) struct SecretManagerSnapshot {
    pub(crate) client: SecretManagerClient,
    pub(crate) system: Option<KeyManagementSystem>,
    /// Typed settings that drive native routing: access mode and hosted keys.
    pub(crate) settings: KeyManagementSettings,
    /// The original `KeyManagementSettings` object, handed back to Python callbacks unchanged.
    pub(crate) settings_object: Option<Py<PyAny>>,
}

impl SecretManagerSnapshot {
    pub(crate) fn into_state(self, context: PythonContext) -> Arc<SecretManagerState> {
        match self.client {
            SecretManagerClient::Native(backend) => {
                Arc::new(SecretManagerState::new(*backend, self.settings))
            }
            SecretManagerClient::Local => Arc::new(SecretManagerState::default()),
            SecretManagerClient::PythonCallback(client) => Arc::new(SecretManagerState::new(
                SecretManager::External(Arc::new(PythonSecretManager::new(
                    client,
                    self.system,
                    self.settings_object,
                    context,
                ))),
                self.settings,
            )),
        }
    }
}

/// Reads and projects the secret manager settings group in one attached operation.
pub(crate) fn read(py: Python<'_>) -> PyResult<SecretManagerSnapshot> {
    let snapshot = project(&PythonSettings::SecretManagerBinding.read(py)?)?;
    let SecretManagerClient::PythonCallback(client) = &snapshot.client else {
        return Ok(snapshot);
    };
    if matches!(
        snapshot.system,
        Some(KeyManagementSystem::Custom | KeyManagementSystem::Local)
    ) {
        return Ok(snapshot);
    }
    let Some(native) = super::runtime::NativeSecretManager::from_client(client.bind(py))? else {
        return Ok(snapshot);
    };
    let backend = native.borrow(py).backend()?;
    if snapshot
        .system
        .is_some_and(|system| system != backend.system())
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "native secret manager system does not match configuration",
        ));
    }
    Ok(SecretManagerSnapshot {
        client: SecretManagerClient::Native(Box::new(backend)),
        ..snapshot
    })
}

pub(crate) fn project(snapshot: &Snapshot<'_>) -> Result<SecretManagerSnapshot, ProjectionError> {
    let system = snapshot.read(&SYSTEM)?;
    let access_mode = snapshot.read(&ACCESS_MODE)?;
    let settings = KeyManagementSettings {
        hosted_keys: snapshot.read(&HOSTED_KEYS)?,
        store_virtual_keys: Some(snapshot.read(&STORE_VIRTUAL_KEYS)?),
        prefix_for_stored_virtual_keys: snapshot.read(&PREFIX_FOR_STORED_VIRTUAL_KEYS)?,
        access_mode,
        primary_secret_name: snapshot.read(&PRIMARY_SECRET_NAME)?,
        kms_key_id: snapshot.read(&KMS_KEY_ID)?,
        custom_secret_manager: snapshot.read(&CUSTOM_SECRET_MANAGER)?,
        aws_region_name: snapshot.read(&AWS_REGION_NAME)?,
        aws_role_name: snapshot.read(&AWS_ROLE_NAME)?,
        aws_session_name: snapshot.read(&AWS_SESSION_NAME)?,
        aws_external_id: snapshot.read(&AWS_EXTERNAL_ID)?.map(SecretValue::new),
        aws_profile_name: snapshot.read(&AWS_PROFILE_NAME)?,
        aws_web_identity_token: snapshot
            .read(&AWS_WEB_IDENTITY_TOKEN)?
            .map(SecretValue::new),
        aws_sts_endpoint: snapshot.read(&AWS_STS_ENDPOINT)?,
        replica_regions: snapshot.read(&REPLICA_REGIONS)?,
        ..KeyManagementSettings::default()
    };
    let client = match snapshot.read(&CLIENT)? {
        None => SecretManagerClient::Local,
        Some(client) => SecretManagerClient::PythonCallback(client),
    };
    Ok(SecretManagerSnapshot {
        client,
        system,
        settings,
        settings_object: snapshot.read(&SETTINGS_OBJECT)?,
    })
}

fn parse_optional_system(
    field: &Field<'_>,
) -> Result<Option<KeyManagementSystem>, ProjectionError> {
    let Some(value) = field.falsy_optional_string()? else {
        return Ok(None);
    };
    serde_json::from_value(Value::String(value))
        .map(Some)
        .map_err(|error| {
            ProjectionError::InvalidConfiguration(format!("secret manager system: {error}"))
        })
}

fn parse_access_mode(field: &Field<'_>) -> Result<AccessMode, ProjectionError> {
    let value = field.strict_string()?;
    serde_json::from_value(Value::String(value)).map_err(|error| {
        ProjectionError::InvalidConfiguration(format!("secret manager access mode: {error}"))
    })
}

#[cfg(test)]
mod tests {
    use pyo3::{
        prelude::*,
        types::{PyDict, PyTuple},
    };

    use super::{SecretManagerClient, project};
    use crate::python_settings::PythonSettings;

    fn snapshot<'py>(
        py: Python<'py>,
        system: &str,
        access_mode: &str,
        store_virtual_keys: Bound<'py, PyAny>,
        hosted_keys: Bound<'py, PyAny>,
    ) -> crate::python_settings::Snapshot<'py> {
        snapshot_with_client(
            py,
            system,
            access_mode,
            store_virtual_keys,
            hosted_keys,
            py.None().into_bound(py),
        )
    }

    fn snapshot_with_client<'py>(
        py: Python<'py>,
        system: &str,
        access_mode: &str,
        store_virtual_keys: Bound<'py, PyAny>,
        hosted_keys: Bound<'py, PyAny>,
        client: Bound<'py, PyAny>,
    ) -> crate::python_settings::Snapshot<'py> {
        let locals = PyDict::new(py);
        locals.set_item("client", client).unwrap();
        locals.set_item("system", system).unwrap();
        locals.set_item("access_mode", access_mode).unwrap();
        locals
            .set_item("store_virtual_keys", store_virtual_keys)
            .unwrap();
        locals.set_item("hosted_keys", hosted_keys).unwrap();
        py.run(
            cr#"
from dataclasses import dataclass
from types import SimpleNamespace

@dataclass(frozen=True, slots=True)
class SecretManager:
    system: object
    access_mode: object
    hosted_keys: object
    primary_secret_name: object
    store_virtual_keys: object
    prefix_for_stored_virtual_keys: object
    kms_key_id: object
    custom_secret_manager: object
    aws_region_name: object
    aws_role_name: object
    aws_session_name: object
    aws_external_id: object
    aws_profile_name: object
    aws_web_identity_token: object
    aws_sts_endpoint: object
    replica_regions: object
    client: object
    settings_object: object

root = SimpleNamespace(secret_manager=SecretManager(
    system=system,
    access_mode=access_mode,
    hosted_keys=hosted_keys,
    primary_secret_name=None,
    store_virtual_keys=store_virtual_keys,
    prefix_for_stored_virtual_keys="litellm/",
    kms_key_id=None,
    custom_secret_manager=None,
    aws_region_name=None,
    aws_role_name=None,
    aws_session_name=None,
    aws_external_id=None,
    aws_profile_name=None,
    aws_web_identity_token=None,
    aws_sts_endpoint=None,
    replica_regions=None,
    client=client,
    settings_object=None,
))
"#,
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
        PythonSettings::SecretManagerBinding.snapshot(
            locals
                .get_item("root")
                .unwrap()
                .unwrap()
                .getattr("secret_manager")
                .unwrap(),
        )
    }

    #[rstest::rstest]
    #[case::string_true(Some("true"), false, true)]
    #[case::string_one(Some("1"), false, true)]
    #[case::true_value(None, true, true)]
    #[case::false_value(None, false, false)]
    #[case::string_false(Some("false"), false, true)]
    fn python_compatible_boolean_coercion(
        #[case] string_value: Option<&str>,
        #[case] bool_value: bool,
        #[case] expected: bool,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let store_virtual_keys = match string_value {
                Some(value) => value.into_pyobject(py).unwrap().into_any(),
                None => bool_value.into_pyobject(py).unwrap().to_owned().into_any(),
            };
            let hosted_keys = PyTuple::new(py, ["ONE"]).unwrap().into_any();
            let projected = project(&snapshot(
                py,
                "local",
                "read_only",
                store_virtual_keys,
                hosted_keys,
            ))
            .unwrap();
            assert_eq!(projected.settings.store_virtual_keys, Some(expected));
        });
    }

    #[test]
    fn unknown_system_is_rejected() {
        Python::initialize();
        Python::attach(|py| {
            let error = project(&snapshot(
                py,
                "unknown",
                "read_only",
                false.into_pyobject(py).unwrap().to_owned().into_any(),
                PyTuple::empty(py).into_any(),
            ))
            .unwrap_err();
            let error: PyErr = error.into();
            assert!(error.is_instance_of::<pyo3::exceptions::PyValueError>(py));
        });
    }

    #[test]
    fn client_identity_selects_local_or_python_callback() {
        Python::initialize();
        Python::attach(|py| {
            let falsy = false.into_pyobject(py).unwrap().to_owned().into_any();
            let local = project(&snapshot(
                py,
                "local",
                "read_only",
                falsy.clone(),
                PyTuple::empty(py).into_any(),
            ))
            .unwrap();
            assert!(matches!(local.client, SecretManagerClient::Local));
            assert!(local.settings_object.is_none());

            let manager = py.eval(c"object()", None, None).unwrap();
            let custom = project(&snapshot_with_client(
                py,
                "custom",
                "read_only",
                falsy,
                PyTuple::empty(py).into_any(),
                manager.clone(),
            ))
            .unwrap();
            let SecretManagerClient::PythonCallback(client) = custom.client else {
                panic!("a live client must stay a Python callback");
            };
            assert!(client.bind(py).is(&manager));
        });
    }
}
