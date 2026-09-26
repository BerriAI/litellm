use std::{future::Future, pin::Pin, sync::Arc};

use litellm_core_utils::settings::Lookup;
use litellm_host_python::{PythonContext, attach_blocking};
use litellm_secrets::{
    Error, ExternalSecretManager, KeyManagementSettings, KeyManagementSystem, Secret, SecretValue,
};
use pyo3::{
    exceptions::PyException,
    prelude::*,
    types::{PyDict, PyString},
};

use super::error::{external_error, read_error};

const HANDLER_MODULE: &str = "litellm.secret_managers.secret_manager_handler";
const ENVIRONMENT_FALLBACK_LOG: &str =
    "Defaulting to os.environ value for key=%s. An exception occurred - %s.\n\n%s";

/// A secret manager whose reads execute in Python: a custom manager, a legacy compatible
/// client, or a manually assigned SDK client.
pub(crate) struct PythonSecretManager {
    client: Arc<PythonClient>,
    context: PythonContext,
}

struct PythonClient {
    client: Py<PyAny>,
    system: Option<KeyManagementSystem>,
    settings: Option<Py<PyAny>>,
}

impl PythonSecretManager {
    pub(crate) fn new(
        client: Py<PyAny>,
        system: Option<KeyManagementSystem>,
        settings: Option<Py<PyAny>>,
        context: PythonContext,
    ) -> Self {
        Self {
            client: Arc::new(PythonClient {
                client,
                system,
                settings,
            }),
            context,
        }
    }
}

impl PythonClient {
    fn read(&self, py: Python<'_>, name: &str) -> PyResult<Option<String>> {
        let client = self.client.bind(py);
        let kwargs = PyDict::new(py);
        kwargs.set_item("client", client)?;
        kwargs.set_item("key_manager", self.system.map_or("local", python_name))?;
        kwargs.set_item("secret_name", name)?;
        match &self.settings {
            Some(settings) => kwargs.set_item("key_management_settings", settings.bind(py))?,
            None => kwargs.set_item("key_management_settings", py.None())?,
        }
        let result = py
            .import(HANDLER_MODULE)?
            .getattr("get_secret_from_manager")?
            .call((), Some(&kwargs))?;
        if result.is_instance_of::<PyString>() {
            result.extract().map(Some)
        } else {
            Ok(None)
        }
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
        self.client.system.unwrap_or(KeyManagementSystem::Custom)
    }

    fn read_secret<'a>(
        &'a self,
        name: &'a str,
        _settings: &'a KeyManagementSettings,
        _environment: &'a (dyn Lookup + Send + Sync),
    ) -> Pin<Box<dyn Future<Output = Result<Option<Secret>, Error>> + Send + 'a>> {
        let client = Arc::clone(&self.client);
        let context = self.context.clone();
        let name = name.to_owned();
        Box::pin(async move {
            match attach_blocking(context, move |py| match client.read(py, &name) {
                Ok(value) => Ok(value.map(SecretValue::new).map(Secret::String)),
                // `get_secret` answers a failed manager read from the process environment, but
                // only for `Exception`: cancellation and other `BaseException`s propagate.
                Err(error) if error.is_instance_of::<PyException>(py) => {
                    log_environment_fallback(py, &name, &error)
                        .map_err(|error| external_error(py, error))?;
                    Err(read_error(py, error))
                }
                Err(error) => Err(external_error(py, error)),
            })
            .await
            {
                Ok(result) => result,
                Err(error) => Python::attach(|py| Err(external_error(py, error))),
            }
        })
    }
}

fn log_environment_fallback(py: Python<'_>, name: &str, error: &PyErr) -> PyResult<()> {
    let traceback = py
        .import("traceback")?
        .call_method1("format_exception", (error.value(py),))?;
    let traceback = "".into_pyobject(py)?.call_method1("join", (traceback,))?;
    py.import("litellm._logging")?
        .getattr("verbose_logger")?
        .call_method1(
            "error",
            (ENVIRONMENT_FALLBACK_LOG, name, error.value(py), traceback),
        )?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::await_holding_lock)]
mod tests {
    use std::sync::{Arc, Mutex, MutexGuard};

    use litellm_secrets::{
        FailurePolicy, KeyManagementSettings, KeyManagementSystem, OidcResolver, SecretManager,
        SecretManagerState, SecretResolver,
    };
    use pyo3::{prelude::*, types::PyDict};
    use rstest::rstest;

    use litellm_host_python::PythonContext;

    use super::{HANDLER_MODULE, PythonSecretManager, python_name};
    use crate::secrets::python_error;

    /// `sys.modules` is interpreter-global, so tests that install or rely on the handler module
    /// cannot overlap with any other test on this list.
    static HANDLER_LOCK: Mutex<()> = Mutex::new(());

    fn handler_guard() -> MutexGuard<'static, ()> {
        HANDLER_LOCK.lock().expect("handler lock poisoned")
    }

    /// A resolver over a Python manager whose reads raise `failure_type`, with the chained
    /// exceptions Python attaches, and `fallback` as the process environment. The returned guard
    /// keeps other module-mutating tests out for the lifetime of the returned resolver.
    fn failing_resolver(
        failure_type: &str,
        fallback: Option<&'static str>,
    ) -> (SecretResolver, Py<PyDict>, MutexGuard<'static, ()>) {
        let handler = handler_guard();
        Python::initialize();
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
import sys, types
for name in ('litellm', 'litellm.secret_managers'):
    sys.modules.setdefault(name, types.ModuleType(name))
handler = sys.modules.setdefault('litellm.secret_managers.secret_manager_handler', types.ModuleType('litellm.secret_managers.secret_manager_handler'))
def get_secret_from_manager(**kwargs):
    return kwargs['client'].sync_read_secret(kwargs['secret_name'])
handler.get_secret_from_manager = get_secret_from_manager
",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let reader = PythonSecretManager::new(
                locals.get_item("manager").unwrap().unwrap().unbind(),
                None,
                None,
                PythonContext::capture(py).unwrap(),
            );
            (reader, locals.unbind())
        });
        let resolver = SecretResolver::new_python_compatible(
            Arc::new(SecretManagerState::new(
                SecretManager::External(Arc::new(reader)),
                KeyManagementSettings::default(),
            )),
            Arc::new(move |_: &str| fallback.map(str::to_owned)),
            OidcResolver::new(litellm_http::Client::plain_for_test()),
        )
        .with_failure_policy(FailurePolicy::EnvironmentFallback);
        (resolver, locals, handler)
    }

    #[rstest]
    #[case::cancelled("asyncio.CancelledError", None)]
    #[case::cancelled_with_fallback("asyncio.CancelledError", Some("environment-key"))]
    #[case::keyboard_interrupt("KeyboardInterrupt", Some("environment-key"))]
    #[tokio::test]
    async fn base_exceptions_propagate_unchanged_even_with_environment_fallback(
        #[case] failure_type: &str,
        #[case] fallback: Option<&'static str>,
    ) {
        let (resolver, locals, _handler) = failing_resolver(failure_type, fallback);
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

    /// Installs a persistent `litellm._logging` stub whose `verbose_logger.error` records its
    /// arguments, and returns those recorded for `name`.
    fn logged_errors<'py>(py: Python<'py>, name: &str) -> Vec<Bound<'py, PyAny>> {
        py.run(
            c"
import sys, types
class Logger:
    calls = []
    def error(self, *args):
        self.calls.append(args)
logging = types.ModuleType('litellm._logging')
logging.verbose_logger = Logger()
sys.modules.setdefault('litellm', types.ModuleType('litellm'))
sys.modules.setdefault('litellm._logging', logging)
",
            None,
            None,
        )
        .unwrap();
        py.import("litellm._logging")
            .unwrap()
            .getattr("verbose_logger")
            .unwrap()
            .getattr("calls")
            .unwrap()
            .try_iter()
            .unwrap()
            .map(Result::unwrap)
            .filter(|call| call.get_item(1).unwrap().extract::<String>().unwrap() == name)
            .collect()
    }

    #[rstest]
    #[case::value_error("ValueError", None, "FALLBACK_VALUE_ERROR")]
    #[case::value_error_with_fallback(
        "ValueError",
        Some("environment-key"),
        "FALLBACK_VALUE_ERROR_WITH_ENVIRONMENT"
    )]
    #[case::runtime_error_with_fallback(
        "RuntimeError",
        Some("environment-key"),
        "FALLBACK_RUNTIME_ERROR_WITH_ENVIRONMENT"
    )]
    #[tokio::test]
    async fn exceptions_are_logged_and_answered_from_the_environment(
        #[case] failure_type: &str,
        #[case] fallback: Option<&'static str>,
        #[case] name: &str,
    ) {
        let (resolver, _locals, _handler) = failing_resolver(failure_type, fallback);
        Python::attach(|py| assert!(logged_errors(py, name).is_empty()));
        let secret = resolver.get_secret(name, None).await.unwrap();
        assert_eq!(
            secret.map(|secret| match secret {
                litellm_secrets::Secret::String(value) => value.expose().to_owned(),
                other => panic!("unexpected secret {other:?}"),
            }),
            fallback.map(str::to_owned)
        );
        Python::attach(|py| {
            let calls = logged_errors(py, name);
            assert_eq!(calls.len(), 1);
            assert!(
                calls[0]
                    .get_item(3)
                    .unwrap()
                    .extract::<String>()
                    .unwrap()
                    .contains("sync_read_secret")
            );
        });
    }

    /// Installs a fake `get_secret_from_manager` that records its kwargs, runs `body`, and
    /// removes the fake handler again; parent package stubs persist for concurrent tests.
    /// Callers hold `handler_guard` before attaching so the GIL is never held while waiting on it.
    fn with_fake_handler<'py>(py: Python<'py>, body: impl FnOnce(&Bound<'py, PyDict>)) {
        let locals = PyDict::new(py);
        py.run(
            c"
import sys, types
previous_handler = sys.modules.get('litellm.secret_managers.secret_manager_handler')
calls = []
def get_secret_from_manager(**kwargs):
    calls.append(kwargs)
    return 'handled-' + kwargs['secret_name']
handler = types.ModuleType('litellm.secret_managers.secret_manager_handler')
handler.get_secret_from_manager = get_secret_from_manager
for name in ('litellm', 'litellm.secret_managers'):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules['litellm.secret_managers.secret_manager_handler'] = handler
",
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
        body(&locals);
        py.run(
            c"
if previous_handler is None:
    sys.modules.pop('litellm.secret_managers.secret_manager_handler', None)
else:
    sys.modules['litellm.secret_managers.secret_manager_handler'] = previous_handler
",
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
    }

    #[rstest]
    #[case("None")]
    #[case("True")]
    #[case("123")]
    #[case("{'key': 'value'}")]
    fn nonstring_results_are_absent_without_a_read_failure(#[case] expression: &str) {
        let _handler = handler_guard();
        Python::initialize();
        Python::attach(|py| {
            with_fake_handler(py, |locals| {
                locals.set_item("expression", expression).unwrap();
                py.run(
                    c"handler.get_secret_from_manager = lambda **kwargs: eval(expression)",
                    Some(locals),
                    Some(locals),
                )
                .unwrap();
                let reader = PythonSecretManager::new(
                    py.None(),
                    None,
                    None,
                    PythonContext::capture(py).unwrap(),
                );
                assert_eq!(reader.client.read(py, "KEY").unwrap(), None);
            });
        });
    }

    #[rstest]
    #[case::google_kms(KeyManagementSystem::GoogleKms)]
    #[case::azure_key_vault(KeyManagementSystem::AzureKeyVault)]
    #[case::aws_secret_manager(KeyManagementSystem::AwsSecretManager)]
    #[case::google_secret_manager(KeyManagementSystem::GoogleSecretManager)]
    #[case::hashicorp_vault(KeyManagementSystem::HashicorpVault)]
    #[case::cyberark(KeyManagementSystem::Cyberark)]
    #[case::local(KeyManagementSystem::Local)]
    #[case::aws_kms(KeyManagementSystem::AwsKms)]
    #[case::custom(KeyManagementSystem::Custom)]
    fn python_names_round_trip_through_serde(#[case] system: KeyManagementSystem) {
        assert_eq!(
            serde_json::to_value(system).unwrap(),
            serde_json::Value::String(python_name(system).to_owned())
        );
    }

    #[test]
    fn configured_systems_dispatch_through_the_python_handler_with_the_original_settings() {
        let _handler = handler_guard();
        Python::initialize();
        Python::attach(|py| {
            with_fake_handler(py, |locals| {
                let client = py.eval(c"object()", None, None).unwrap();
                let settings = py.eval(c"object()", None, None).unwrap();
                let reader = PythonSecretManager::new(
                    client.clone().unbind(),
                    Some(KeyManagementSystem::AzureKeyVault),
                    Some(settings.clone().unbind()),
                    PythonContext::capture(py).unwrap(),
                );
                assert_eq!(
                    reader.client.read(py, "API_KEY").unwrap().as_deref(),
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

    #[rstest]
    #[case::manually_assigned(None, "local")]
    #[case::custom(Some(KeyManagementSystem::Custom), "custom")]
    fn direct_readers_dispatch_through_the_python_handler_like_get_secret(
        #[case] system: Option<KeyManagementSystem>,
        #[case] key_manager: &str,
    ) {
        let _handler = handler_guard();
        Python::initialize();
        Python::attach(|py| {
            with_fake_handler(py, |locals| {
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
                    Some(locals),
                    Some(locals),
                )
                .unwrap();
                let manager = locals.get_item("manager").unwrap().unwrap();
                let reader = PythonSecretManager::new(
                    manager.clone().unbind(),
                    system,
                    None,
                    PythonContext::capture(py).unwrap(),
                );
                assert_eq!(
                    reader.client.read(py, "API_KEY").unwrap().as_deref(),
                    Some("handled-API_KEY")
                );
                assert_eq!(
                    manager
                        .getattr("names")
                        .unwrap()
                        .extract::<Vec<String>>()
                        .unwrap(),
                    Vec::<String>::new()
                );
                let calls = locals.get_item("calls").unwrap().unwrap();
                let call = calls.get_item(0).unwrap().cast_into::<PyDict>().unwrap();
                assert!(call.get_item("client").unwrap().unwrap().is(&manager));
                assert_eq!(
                    call.get_item("key_manager")
                        .unwrap()
                        .unwrap()
                        .extract::<String>()
                        .unwrap(),
                    key_manager
                );
            });
        });
    }
}
