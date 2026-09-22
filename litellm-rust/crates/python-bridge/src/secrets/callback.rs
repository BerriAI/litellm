use std::{fmt, future::Future, pin::Pin};

use litellm_core_utils::settings::Lookup;
use litellm_secrets::{
    Error, ExternalSecretManager, KeyManagementSettings, KeyManagementSystem, Secret, SecretValue,
};
use pyo3::{exceptions::PyBaseException, prelude::*, types::PyDict};

const HANDLER_MODULE: &str = "litellm.secret_managers.secret_manager_handler";

struct PythonSecretError(Py<PyBaseException>);

impl fmt::Debug for PythonSecretError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PythonSecretError")
    }
}

impl fmt::Display for PythonSecretError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Python secret manager failed")
    }
}

impl std::error::Error for PythonSecretError {}

pub(crate) fn python_error(py: Python<'_>, error: &Error) -> Option<PyErr> {
    let Error::ExternalManager(source) = error else {
        return None;
    };
    source
        .downcast_ref::<PythonSecretError>()
        .map(|error| PyErr::from_value(error.0.clone_ref(py).into_bound(py).into_any()))
}

/// A secret manager whose reads execute in Python: a custom manager, a legacy compatible
/// client, or a manually assigned SDK client.
pub(crate) struct PythonSecretManager {
    client: Py<PyAny>,
    system: Option<KeyManagementSystem>,
    /// The `key_manager` name Python's handler dispatches on.
    key_manager: &'static str,
    settings: Option<Py<PyAny>>,
}

impl PythonSecretManager {
    pub(crate) fn new(
        client: Py<PyAny>,
        system: Option<KeyManagementSystem>,
        settings: Option<Py<PyAny>>,
    ) -> Self {
        Self {
            client,
            system,
            key_manager: system.map_or("local", python_name),
            settings,
        }
    }

    fn read(&self, py: Python<'_>, name: &str) -> PyResult<Option<String>> {
        let client = self.client.bind(py);
        if self.system == Some(KeyManagementSystem::Custom)
            || (self.system.is_none() && client.hasattr("sync_read_secret")?)
        {
            let kwargs = PyDict::new(py);
            kwargs.set_item("secret_name", name)?;
            if self.system == Some(KeyManagementSystem::Custom) {
                let optional_params = self
                    .settings
                    .as_ref()
                    .map(|settings| settings.bind(py).call_method0("model_dump"))
                    .transpose()?;
                kwargs.set_item("optional_params", optional_params)?;
            }
            return client
                .call_method("sync_read_secret", (), Some(&kwargs))?
                .extract();
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("client", client)?;
        kwargs.set_item("key_manager", self.key_manager)?;
        kwargs.set_item("secret_name", name)?;
        kwargs.set_item(
            "key_management_settings",
            self.settings
                .as_ref()
                .map_or_else(|| py.None(), |settings| settings.clone_ref(py)),
        )?;
        py.import(HANDLER_MODULE)?
            .getattr("get_secret_from_manager")?
            .call((), Some(&kwargs))?
            .extract()
    }
}

/// The `KeyManagementSystem` value as Python spells it.
fn python_name(system: KeyManagementSystem) -> &'static str {
    match system {
        KeyManagementSystem::GoogleKms => "google_kms",
        KeyManagementSystem::AzureKeyVault => "azure_key_vault",
        KeyManagementSystem::AwsSecretManager => "aws_secret_manager",
        KeyManagementSystem::GoogleSecretManager => "google_secret_manager",
        KeyManagementSystem::HashicorpVault => "hashicorp_vault",
        KeyManagementSystem::Cyberark => "cyberark",
        KeyManagementSystem::Local => "local",
        KeyManagementSystem::AwsKms => "aws_kms",
        KeyManagementSystem::Custom => "custom",
    }
}

impl ExternalSecretManager for PythonSecretManager {
    fn system(&self) -> KeyManagementSystem {
        self.system.unwrap_or(KeyManagementSystem::Custom)
    }

    fn read_secret<'a>(
        &'a self,
        name: &'a str,
        _settings: &'a KeyManagementSettings,
        _environment: &'a (dyn Lookup + Send + Sync),
    ) -> Pin<Box<dyn Future<Output = Result<Option<Secret>, Error>> + Send + 'a>> {
        Box::pin(async move {
            Python::attach(|py| {
                self.read(py, name)
                    .map(|value| value.map(SecretValue::new).map(Secret::String))
                    .map_err(|error| {
                        Error::ExternalManager(Box::new(PythonSecretError(error.into_value(py))))
                    })
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use litellm_secrets::{
        FailurePolicy, KeyManagementSettings, KeyManagementSystem, OidcResolver, SecretManager,
        SecretManagerState, SecretResolver,
    };
    use pyo3::{prelude::*, types::PyDict};

    use super::{HANDLER_MODULE, PythonSecretManager, python_error, python_name};

    #[tokio::test]
    async fn callback_failures_preserve_python_exceptions_even_with_environment_fallback() {
        Python::initialize();
        for failure_type in ["ValueError", "asyncio.CancelledError"] {
            for fallback in [None, Some("environment-key")] {
                let (reader, locals) = Python::attach(|py| {
                    let locals = PyDict::new(py);
                    locals.set_item("failure_type", failure_type).unwrap();
                    py.run(
                        c"
import asyncio
failure = eval(failure_type)('secret manager failed')
cause = RuntimeError('original cause')
context = RuntimeError('original context')
failure.__cause__ = cause
failure.__context__ = context
class Manager:
    def sync_read_secret(self, secret_name):
        raise failure
manager = Manager()
",
                        Some(&locals),
                        Some(&locals),
                    )
                    .unwrap();
                    let reader = PythonSecretManager::new(
                        locals.get_item("manager").unwrap().unwrap().unbind(),
                        None,
                        None,
                    );
                    (reader, locals.unbind())
                });
                let resolver = SecretResolver::new(
                    Arc::new(SecretManagerState::new(
                        SecretManager::External(Arc::new(reader)),
                        KeyManagementSettings::default(),
                    )),
                    Arc::new(move |_: &str| fallback.map(str::to_owned)),
                    OidcResolver::default(),
                )
                .with_failure_policy(FailurePolicy::EnvironmentFallback);
                let error = resolver.get_secret("API_KEY", None).await.unwrap_err();
                Python::attach(|py| {
                    let original = python_error(py, &error).unwrap();
                    let locals = locals.bind(py);
                    assert!(
                        original
                            .value(py)
                            .is(locals.get_item("failure").unwrap().unwrap())
                    );
                    for (attribute, name) in [("__cause__", "cause"), ("__context__", "context")] {
                        assert!(
                            original
                                .value(py)
                                .getattr(attribute)
                                .unwrap()
                                .is(locals.get_item(name).unwrap().unwrap())
                        );
                    }
                    assert!(original.traceback(py).is_some());
                });
            }
        }
    }

    /// Installs a fake `get_secret_from_manager` that records its kwargs, runs `body`, and
    /// removes the fake modules again.
    fn with_fake_handler<'py>(py: Python<'py>, body: impl FnOnce(&Bound<'py, PyDict>)) {
        let locals = PyDict::new(py);
        py.run(
            c"
import sys, types
calls = []
def get_secret_from_manager(**kwargs):
    calls.append(kwargs)
    return 'handled-' + kwargs['secret_name']
handler = types.ModuleType('litellm.secret_managers.secret_manager_handler')
handler.get_secret_from_manager = get_secret_from_manager
installed = {}
for name in ('litellm', 'litellm.secret_managers'):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
        installed[name] = True
sys.modules['litellm.secret_managers.secret_manager_handler'] = handler
",
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
        body(&locals);
        py.run(
            c"
sys.modules.pop('litellm.secret_managers.secret_manager_handler', None)
for name in installed:
    sys.modules.pop(name, None)
",
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
    }

    #[test]
    fn python_names_round_trip_through_serde() {
        for system in [
            KeyManagementSystem::GoogleKms,
            KeyManagementSystem::AzureKeyVault,
            KeyManagementSystem::AwsSecretManager,
            KeyManagementSystem::GoogleSecretManager,
            KeyManagementSystem::HashicorpVault,
            KeyManagementSystem::Cyberark,
            KeyManagementSystem::Local,
            KeyManagementSystem::AwsKms,
            KeyManagementSystem::Custom,
        ] {
            assert_eq!(
                serde_json::to_value(system).unwrap(),
                serde_json::Value::String(python_name(system).to_owned())
            );
        }
    }

    #[test]
    fn custom_readers_without_a_system_are_called_directly() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"
class Manager:
    def __init__(self):
        self.names = []
    def sync_read_secret(self, secret_name, optional_params=None, timeout=None):
        self.names.append(secret_name)
        return 'direct-' + secret_name
manager = Manager()
",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let manager = locals.get_item("manager").unwrap().unwrap();
            let reader = PythonSecretManager::new(manager.clone().unbind(), None, None);
            assert_eq!(
                reader.read(py, "API_KEY").unwrap().as_deref(),
                Some("direct-API_KEY")
            );
            assert_eq!(
                manager
                    .getattr("names")
                    .unwrap()
                    .extract::<Vec<String>>()
                    .unwrap(),
                ["API_KEY"]
            );
        });
    }

    #[test]
    fn configured_systems_dispatch_through_the_python_handler_with_the_original_settings() {
        Python::initialize();
        Python::attach(|py| {
            with_fake_handler(py, |locals| {
                let client = py.eval(c"object()", None, None).unwrap();
                let settings = py.eval(c"object()", None, None).unwrap();
                let reader = PythonSecretManager::new(
                    client.clone().unbind(),
                    Some(KeyManagementSystem::AzureKeyVault),
                    Some(settings.clone().unbind()),
                );
                assert_eq!(
                    reader.read(py, "API_KEY").unwrap().as_deref(),
                    Some("handled-API_KEY")
                );
                assert!(py.import(HANDLER_MODULE).is_ok());
                let calls = locals.get_item("calls").unwrap().unwrap();
                let call = calls.get_item(0).unwrap().cast_into::<PyDict>().unwrap();
                assert!(call.get_item("client").unwrap().unwrap().is(&client));
                assert!(
                    call.get_item("key_management_settings")
                        .unwrap()
                        .unwrap()
                        .is(&settings)
                );
                assert_eq!(
                    call.get_item("key_manager")
                        .unwrap()
                        .unwrap()
                        .extract::<String>()
                        .unwrap(),
                    "azure_key_vault"
                );
                assert_eq!(
                    call.get_item("secret_name")
                        .unwrap()
                        .unwrap()
                        .extract::<String>()
                        .unwrap(),
                    "API_KEY"
                );
            });
        });
    }
}
