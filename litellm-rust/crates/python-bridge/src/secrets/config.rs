use std::sync::Arc;

use litellm_secrets::{SecretManager, SecretManagerState};
use litellm_secrets_types::{AccessMode, KeyManagementSettings, KeyManagementSystem, SecretValue};
use pyo3::prelude::*;
use serde_json::Value;

use super::callback::PythonSecretManager;
use crate::{
    coercion::{Field, ProjectionError},
    python_settings::{Adapter, PythonSettings, SettingSpec, Snapshot},
};

const fn secret_manager(name: &'static str, adapter: Adapter) -> SettingSpec {
    SettingSpec::new(PythonSettings::SecretManager, name, adapter)
}

const fn aws(name: &'static str) -> SettingSpec {
    secret_manager(name, Adapter::FalsyOptionalString)
}

pub(crate) const SYSTEM: SettingSpec = secret_manager("system", Adapter::FalsyOptionalString);
pub(crate) const ACCESS_MODE: SettingSpec = secret_manager("access_mode", Adapter::StrictString);
pub(crate) const HOSTED_KEYS: SettingSpec =
    secret_manager("hosted_keys", Adapter::StringCollection).shapes(&["none", "list"]);
pub(crate) const PRIMARY_SECRET_NAME: SettingSpec =
    secret_manager("primary_secret_name", Adapter::FalsyOptionalString);
pub(crate) const STORE_VIRTUAL_KEYS: SettingSpec =
    secret_manager("store_virtual_keys", Adapter::Truthy);
pub(crate) const PREFIX_FOR_STORED_VIRTUAL_KEYS: SettingSpec =
    secret_manager("prefix_for_stored_virtual_keys", Adapter::StrictString);
pub(crate) const KMS_KEY_ID: SettingSpec =
    secret_manager("kms_key_id", Adapter::FalsyOptionalString);
pub(crate) const CUSTOM_SECRET_MANAGER: SettingSpec =
    secret_manager("custom_secret_manager", Adapter::FalsyOptionalString);
pub(crate) const AWS_REGION_NAME: SettingSpec = aws("aws_region_name");
pub(crate) const AWS_ROLE_NAME: SettingSpec = aws("aws_role_name");
pub(crate) const AWS_SESSION_NAME: SettingSpec = aws("aws_session_name");
pub(crate) const AWS_EXTERNAL_ID: SettingSpec = aws("aws_external_id").sensitive();
pub(crate) const AWS_PROFILE_NAME: SettingSpec = aws("aws_profile_name");
pub(crate) const AWS_WEB_IDENTITY_TOKEN: SettingSpec = aws("aws_web_identity_token").sensitive();
pub(crate) const AWS_STS_ENDPOINT: SettingSpec = aws("aws_sts_endpoint");
pub(crate) const REPLICA_REGIONS: SettingSpec =
    secret_manager("replica_regions", Adapter::StringCollection).shapes(&["none", "list"]);
pub(crate) const CLIENT: SettingSpec =
    secret_manager("client", Adapter::PythonBinding).shapes(&["none", "object"]);
pub(crate) const SETTINGS_OBJECT: SettingSpec =
    secret_manager("settings_object", Adapter::PythonBinding).shapes(&["none", "object"]);

#[cfg(test)]
pub(crate) const SECRET_MANAGER_SPECS: &[SettingSpec] = &[
    SYSTEM,
    ACCESS_MODE,
    HOSTED_KEYS,
    PRIMARY_SECRET_NAME,
    STORE_VIRTUAL_KEYS,
    PREFIX_FOR_STORED_VIRTUAL_KEYS,
    KMS_KEY_ID,
    CUSTOM_SECRET_MANAGER,
    AWS_REGION_NAME,
    AWS_ROLE_NAME,
    AWS_SESSION_NAME,
    AWS_EXTERNAL_ID,
    AWS_PROFILE_NAME,
    AWS_WEB_IDENTITY_TOKEN,
    AWS_STS_ENDPOINT,
    REPLICA_REGIONS,
    CLIENT,
    SETTINGS_OBJECT,
];

/// `litellm.secret_manager_client` as the bridge classifies it.
#[derive(Debug)]
pub(crate) enum SecretManagerClient {
    /// `None`: reads come from the process environment.
    Local,
    /// A custom manager, legacy compatible client, or manually assigned SDK client that keeps
    /// executing in Python.
    PythonCallback(Py<PyAny>),
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
    pub(crate) fn into_state(self) -> Arc<SecretManagerState> {
        match self.client {
            SecretManagerClient::Local => Arc::new(SecretManagerState::default()),
            SecretManagerClient::PythonCallback(client) => Arc::new(SecretManagerState::new(
                SecretManager::External(Arc::new(PythonSecretManager::new(
                    client,
                    self.system,
                    self.settings_object,
                ))),
                self.settings,
            )),
        }
    }
}

/// Reads and projects the secret manager settings group in one attached operation.
pub(crate) fn read(py: Python<'_>) -> PyResult<SecretManagerSnapshot> {
    Ok(project(&PythonSettings::SecretManager.read(py)?)?)
}

pub(crate) fn project(snapshot: &Snapshot<'_>) -> Result<SecretManagerSnapshot, ProjectionError> {
    let string = |spec: &SettingSpec| -> Result<Option<String>, ProjectionError> {
        Ok(snapshot.field(spec)?.falsy_optional_string()?.0)
    };
    let collection = |spec: &SettingSpec| -> Result<Option<Vec<String>>, ProjectionError> {
        Ok(snapshot
            .field(spec)?
            .optional_string_collection()?
            .map(|collection| collection.0))
    };
    let system = parse_optional_system(&snapshot.field(&SYSTEM)?)?;
    let access_mode = parse_access_mode(&snapshot.field(&ACCESS_MODE)?)?;
    let settings = KeyManagementSettings {
        hosted_keys: collection(&HOSTED_KEYS)?,
        store_virtual_keys: Some(snapshot.field(&STORE_VIRTUAL_KEYS)?.truthy()?.0),
        prefix_for_stored_virtual_keys: snapshot
            .field(&PREFIX_FOR_STORED_VIRTUAL_KEYS)?
            .strict_string()?,
        access_mode,
        primary_secret_name: string(&PRIMARY_SECRET_NAME)?,
        kms_key_id: string(&KMS_KEY_ID)?,
        custom_secret_manager: string(&CUSTOM_SECRET_MANAGER)?,
        aws_region_name: string(&AWS_REGION_NAME)?,
        aws_role_name: string(&AWS_ROLE_NAME)?,
        aws_session_name: string(&AWS_SESSION_NAME)?,
        aws_external_id: string(&AWS_EXTERNAL_ID)?.map(SecretValue::new),
        aws_profile_name: string(&AWS_PROFILE_NAME)?,
        aws_web_identity_token: string(&AWS_WEB_IDENTITY_TOKEN)?.map(SecretValue::new),
        aws_sts_endpoint: string(&AWS_STS_ENDPOINT)?,
        replica_regions: collection(&REPLICA_REGIONS)?,
        ..KeyManagementSettings::default()
    };
    let client = match snapshot.field(&CLIENT)?.python_binding() {
        None => SecretManagerClient::Local,
        Some(client) => SecretManagerClient::PythonCallback(client),
    };
    Ok(SecretManagerSnapshot {
        client,
        system,
        settings,
        settings_object: snapshot.field(&SETTINGS_OBJECT)?.python_binding(),
    })
}

fn parse_optional_system(
    field: &Field<'_>,
) -> Result<Option<KeyManagementSystem>, ProjectionError> {
    let Some(value) = field.falsy_optional_string()?.0 else {
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
        PythonSettings::SecretManager.snapshot(
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
