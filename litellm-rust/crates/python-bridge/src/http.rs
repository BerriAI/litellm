use std::{
    collections::{BTreeSet, HashSet},
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, Mutex, PoisonError},
};

use litellm_core_utils::settings::ProcessEnvironment;
use litellm_http::{
    Client, ClientVariant, HttpClientConfig, HttpClientPool, HttpSettings, HttpSettingsLayer,
    Resolution, SslVerify, TlsSource, Unsupported,
    media::{PublicDnsResolver, UrlPolicy},
};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBool, PyDict, PyString},
};

use crate::{
    coercion::{Field, FieldSpec, ProjectionError},
    python_settings::{PythonSettings, Snapshot},
};

const SSL_VERIFY: FieldSpec<Option<SslVerify>> = FieldSpec::new("ssl_verify", decode_ssl_verify);
const SSL_CERTIFICATE: FieldSpec<Option<String>> =
    FieldSpec::new("ssl_certificate", |field| field.optional_strict_string());
const SSL_SECURITY_LEVEL: FieldSpec<Option<String>> =
    FieldSpec::new("ssl_security_level", |field| field.tuning_string());
const SSL_ECDH_CURVE: FieldSpec<Option<String>> =
    FieldSpec::new("ssl_ecdh_curve", |field| field.tuning_string());
const FORCE_IPV4: FieldSpec<bool> = FieldSpec::new("force_ipv4", |field| field.truthy());
const HTTP2: FieldSpec<bool> = FieldSpec::new("http2", |field| Ok(field.exact_true()));
const AIOHTTP_TRUST_ENV: FieldSpec<bool> =
    FieldSpec::new("aiohttp_trust_env", |field| field.truthy());
const DISABLE_AIOHTTP_TRUST_ENV: FieldSpec<bool> =
    FieldSpec::new("disable_aiohttp_trust_env", |field| field.truthy());
const DISABLE_AIOHTTP_TRANSPORT: FieldSpec<bool> =
    FieldSpec::new("disable_aiohttp_transport", |field| Ok(field.exact_true()));
const USER_AGENT: FieldSpec<String> = FieldSpec::new("user_agent", |field| field.schema_string());
const USER_URL_VALIDATION: FieldSpec<bool> =
    FieldSpec::new("user_url_validation", |field| field.truthy());
const USER_URL_ALLOWED_HOSTS: FieldSpec<Vec<String>> =
    FieldSpec::new("user_url_allowed_hosts", decode_hosts);

fn decode_hosts(field: &Field<'_>) -> Result<Vec<String>, ProjectionError> {
    Ok(field
        .string_collection()?
        .into_iter()
        .map(|host| litellm_http::media::normalize_host(&host))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect())
}

fn decode_ssl_verify(field: &Field<'_>) -> Result<Option<SslVerify>, ProjectionError> {
    let value = field.value();
    if value.is_none() {
        return Ok(None);
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(Some(if field.exact_true() {
            SslVerify::Enabled
        } else {
            SslVerify::Disabled
        }));
    }
    if value.is_instance_of::<PyString>() {
        return Ok(Some(match field.str_bool()? {
            Some(true) => SslVerify::Enabled,
            Some(false) => SslVerify::Disabled,
            None => SslVerify::CaBundle(field.strict_string()?.into()),
        }));
    }
    let context = value.py().import("ssl")?.getattr("SSLContext")?;
    if value.is_instance(&context)? {
        return Err(ProjectionError::UnsupportedLiveObject(field.expected(
            "a Boolean, Boolean string, CA path, or None; live SSLContext is unsupported",
        )?));
    }
    Err(field.invalid("a Boolean, Boolean string, CA path, or None"))
}

static POOL: LazyLock<HttpClientPool> =
    LazyLock::new(|| HttpClientPool::new(Arc::new(PublicDnsResolver)));

static REPORTED_UNSUPPORTED: LazyLock<Mutex<HashSet<Unsupported>>> = LazyLock::new(Mutex::default);

pub(crate) fn pool() -> &'static HttpClientPool {
    &POOL
}

pub(crate) fn call_config(
    py: Python<'_>,
    kwargs: &Bound<'_, PyDict>,
    asynchronous: bool,
) -> PyResult<HttpClientConfig> {
    let settings = HttpSettings::from_layers([
        for_call(call_ssl_verify(kwargs)?, asynchronous),
        HttpSettingsLayer::from_environment(&ProcessEnvironment),
        match PythonSettings::Http.read_or_unset(py)? {
            Some(snapshot) => configured(&snapshot)?,
            None => HttpSettingsLayer::default(),
        },
    ])
    .without_missing_files(&|path: &Path| path.exists());
    let resolution = Resolution::from(&settings);
    for unsupported in unreported(&REPORTED_UNSUPPORTED, resolution.unsupported) {
        crate::logger::capture(py).scope(|| litellm_tracing::warn!("{unsupported}"));
    }
    Ok(resolution.config)
}

pub(crate) fn host_client(py: Python<'_>, variant: ClientVariant) -> PyResult<Client> {
    let config = call_config(py, &PyDict::new(py), true)?;
    pool().client(&config, variant).map_err(client_error)
}

pub(crate) fn client_error(error: litellm_http::Error) -> PyErr {
    match error {
        litellm_http::Error::Read {
            tls_source: TlsSource::ClientIdentity,
            ..
        }
        | litellm_http::Error::InvalidPem {
            tls_source: TlsSource::ClientIdentity,
            ..
        } => PyValueError::new_err(
            "http_settings.ssl_certificate: expected a readable PEM certificate and private key",
        ),
        litellm_http::Error::Read {
            tls_source: TlsSource::CaBundle,
            ..
        }
        | litellm_http::Error::InvalidPem {
            tls_source: TlsSource::CaBundle,
            ..
        } => PyValueError::new_err("http_settings.ssl_verify: expected a readable PEM CA bundle"),
        _ => PyValueError::new_err("http_settings: native HTTP client configuration is invalid"),
    }
}

fn unreported(
    reported: &Mutex<HashSet<Unsupported>>,
    unsupported: Vec<Unsupported>,
) -> Vec<Unsupported> {
    let mut reported = reported.lock().unwrap_or_else(PoisonError::into_inner);
    unsupported
        .into_iter()
        .filter(|unsupported| reported.insert(unsupported.clone()))
        .collect()
}

pub(crate) fn url_policy(py: Python<'_>) -> PyResult<UrlPolicy> {
    match PythonSettings::UrlPolicy.read_or_unset(py)? {
        Some(snapshot) => project_url_policy(&snapshot),
        None => Ok(UrlPolicy::default()),
    }
}

fn project_url_policy(snapshot: &Snapshot<'_>) -> PyResult<UrlPolicy> {
    Ok(UrlPolicy {
        validate: snapshot.read(&USER_URL_VALIDATION)?,
        allowed_hosts: snapshot.read(&USER_URL_ALLOWED_HOSTS)?,
    })
}

fn call_ssl_verify(kwargs: &Bound<'_, PyDict>) -> PyResult<Option<SslVerify>> {
    match kwargs.get_item("ssl_verify")? {
        Some(value) => Ok(decode_ssl_verify(&Field::new(
            "request",
            "ssl_verify",
            value,
        ))?),
        None => Ok(None),
    }
}

fn for_call(call_ssl_verify: Option<SslVerify>, asynchronous: bool) -> HttpSettingsLayer {
    HttpSettingsLayer {
        ssl_verify: call_ssl_verify,
        disable_aiohttp_transport: (!asynchronous).then_some(true),
        ..HttpSettingsLayer::default()
    }
}

fn configured(snapshot: &Snapshot<'_>) -> PyResult<HttpSettingsLayer> {
    Ok(HttpSettingsLayer {
        ssl_verify: snapshot.read(&SSL_VERIFY)?,
        ssl_certificate: snapshot.read(&SSL_CERTIFICATE)?.map(PathBuf::from),
        ssl_security_level: snapshot.read(&SSL_SECURITY_LEVEL)?,
        ssl_ecdh_curve: snapshot.read(&SSL_ECDH_CURVE)?,
        force_ipv4: Some(snapshot.read(&FORCE_IPV4)?),
        http2: Some(snapshot.read(&HTTP2)?),
        aiohttp_trust_env: Some(snapshot.read(&AIOHTTP_TRUST_ENV)?),
        disable_aiohttp_trust_env: Some(snapshot.read(&DISABLE_AIOHTTP_TRUST_ENV)?),
        disable_aiohttp_transport: Some(snapshot.read(&DISABLE_AIOHTTP_TRANSPORT)?),
        user_agent: Some(snapshot.read(&USER_AGENT)?),
        ..HttpSettingsLayer::default()
    })
}

#[cfg(test)]
mod tests {
    use litellm_http::Verify;
    use pyo3::exceptions::PyRuntimeError;
    use rstest::rstest;

    use super::*;

    fn evaluate<'py>(py: Python<'py>, source: &str) -> Bound<'py, PyAny> {
        py.eval(&std::ffi::CString::new(source).unwrap(), None, None)
            .unwrap()
    }

    fn python_settings<'py>(py: Python<'py>, overrides: &str) -> Snapshot<'py> {
        let source = format!(
            "
import types
defaults = dict(
    ssl_verify=True,
    ssl_certificate=None,
    ssl_security_level=None,
    ssl_ecdh_curve=None,
    force_ipv4=False,
    http2=False,
    aiohttp_trust_env=False,
    disable_aiohttp_trust_env=False,
    disable_aiohttp_transport=False,
    user_agent='litellm/test',
)
defaults.update(dict({overrides}))
settings = types.SimpleNamespace(**defaults)
"
        );
        let locals = PyDict::new(py);
        let source = std::ffi::CString::new(source).unwrap();
        py.run(&source, Some(&locals), Some(&locals)).unwrap();
        PythonSettings::Http.snapshot(locals.get_item("settings").unwrap().unwrap())
    }

    #[test]
    fn default_python_settings_resolve_to_default_settings_with_verification_on() {
        Python::initialize();
        Python::attach(|py| {
            let layer = configured(&python_settings(py, "")).unwrap();
            assert_eq!(
                HttpSettings::from_layers([layer]),
                HttpSettings {
                    ssl_verify: Some(SslVerify::Enabled),
                    user_agent: Some("litellm/test".into()),
                    ..HttpSettings::default()
                }
            );
        });
    }

    #[test]
    fn client_error_uses_tls_source_when_paths_match() {
        Python::initialize();
        Python::attach(|py| {
            let path = PathBuf::from("/shared.pem");
            let ca_error = client_error(litellm_http::Error::InvalidPem {
                path: path.clone(),
                message: "invalid".into(),
                tls_source: TlsSource::CaBundle,
            });
            assert_eq!(
                ca_error.to_string(),
                "ValueError: http_settings.ssl_verify: expected a readable PEM CA bundle"
            );
            let client_error = client_error(litellm_http::Error::InvalidPem {
                path,
                message: "invalid".into(),
                tls_source: TlsSource::ClientIdentity,
            });
            assert!(client_error.is_instance_of::<PyValueError>(py));
            assert_eq!(
                client_error.to_string(),
                "ValueError: http_settings.ssl_certificate: expected a readable PEM certificate and private key"
            );
        });
    }

    #[test]
    fn python_settings_flow_into_the_configured_layer() {
        Python::initialize();
        Python::attach(|py| {
            let layer = configured(&python_settings(
                py,
                "
ssl_verify='/etc/ssl/corp.pem',
ssl_certificate='/etc/ssl/client.pem',
ssl_security_level='2',
ssl_ecdh_curve='X25519',
force_ipv4=True,
http2=True,
aiohttp_trust_env=True,
disable_aiohttp_trust_env=True,
disable_aiohttp_transport=True,
user_agent='litellm/9.9.9',
",
            ))
            .unwrap();
            assert_eq!(
                layer,
                HttpSettingsLayer {
                    ssl_verify: Some(SslVerify::CaBundle("/etc/ssl/corp.pem".into())),
                    ssl_certificate: Some("/etc/ssl/client.pem".into()),
                    ssl_security_level: Some("2".into()),
                    ssl_ecdh_curve: Some("X25519".into()),
                    force_ipv4: Some(true),
                    http2: Some(true),
                    aiohttp_trust_env: Some(true),
                    disable_aiohttp_trust_env: Some(true),
                    disable_aiohttp_transport: Some(true),
                    user_agent: Some("litellm/9.9.9".into()),
                    ..HttpSettingsLayer::default()
                }
            );
        });
    }

    #[test]
    fn user_agent_environment_variable_beats_the_python_default() {
        Python::initialize();
        Python::attach(|py| {
            let settings = HttpSettings::from_layers([
                HttpSettingsLayer::from_environment(&|name: &str| {
                    (name == "LITELLM_USER_AGENT").then(|| "operator/1".to_string())
                }),
                configured(&python_settings(py, "")).unwrap(),
            ]);
            assert_eq!(settings.user_agent.as_deref(), Some("operator/1"));
        });
    }

    #[rstest]
    #[case::disabled("ssl_verify=False", Verify::Disabled)]
    #[case::disabled_string("ssl_verify='False'", Verify::Disabled)]
    #[case::enabled_string("ssl_verify='true'", Verify::BuiltInRoots)]
    #[case::bundle("ssl_verify='/tmp/ca.pem'", Verify::CaBundle("/tmp/ca.pem".into()))]
    fn ssl_verify_global_resolves_like_get_ssl_verify(
        #[case] overrides: &str,
        #[case] expected: Verify,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let layer = configured(&python_settings(py, overrides)).unwrap();
            let config = Resolution::from(&HttpSettings::from_layers([layer])).config;
            assert_eq!(config.verify, expected);
        });
    }

    #[rstest]
    #[case("ssl_verify=object()")]
    #[case("ssl_verify=__import__('ssl').SSLContext(__import__('ssl').PROTOCOL_TLS_CLIENT)")]
    #[case("ssl_certificate=1")]
    fn invalid_http_configuration_is_terminal(#[case] overrides: &str) {
        Python::initialize();
        Python::attach(|py| {
            let error = configured(&python_settings(py, overrides)).unwrap_err();
            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(error.to_string().contains("http_settings.ssl_"));
        });
    }

    #[test]
    fn unsupported_settings_are_reported_once_per_process() {
        let reported = Mutex::default();
        let curve = Unsupported::EcdhCurve("secp521r1".into());
        let level = Unsupported::SecurityLevel("@SECLEVEL=1".into());
        assert_eq!(
            unreported(&reported, vec![curve.clone(), level.clone()]),
            [curve.clone(), level]
        );
        assert_eq!(unreported(&reported, vec![curve]), []);
    }

    #[test]
    fn mutable_globals_use_their_consumer_operations() {
        Python::initialize();
        Python::attach(|py| {
            let layer = configured(&python_settings(py,
                "force_ipv4='yes', http2=1, disable_aiohttp_transport=1, aiohttp_trust_env=[1], disable_aiohttp_trust_env=[], ssl_security_level=1, ssl_ecdh_curve=[]"
            )).unwrap();
            assert_eq!(layer.force_ipv4, Some(true));
            assert_eq!(layer.http2, Some(false));
            assert_eq!(layer.disable_aiohttp_transport, Some(false));
            assert_eq!(layer.aiohttp_trust_env, Some(true));
            assert_eq!(layer.disable_aiohttp_trust_env, Some(false));
            assert_eq!(layer.ssl_security_level, None);
            assert_eq!(layer.ssl_ecdh_curve, None);
            let error = configured(&python_settings(py, "user_agent=1")).unwrap_err();
            assert!(error.is_instance_of::<PyRuntimeError>(py));
        });
    }

    fn configured_ssl_verify(ssl_verify: SslVerify) -> HttpSettingsLayer {
        HttpSettingsLayer {
            ssl_verify: Some(ssl_verify),
            ..HttpSettingsLayer::default()
        }
    }

    #[test]
    fn call_ssl_verify_beats_the_configured_value() {
        Python::initialize();
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("ssl_verify", false).unwrap();
            let call = for_call(call_ssl_verify(&kwargs).unwrap(), true);
            let settings =
                HttpSettings::from_layers([call, configured_ssl_verify(SslVerify::Enabled)]);
            assert_eq!(settings.ssl_verify, Some(SslVerify::Disabled));
        });
    }

    #[test]
    fn absent_call_ssl_verify_keeps_the_configured_value() {
        Python::initialize();
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("ssl_verify", py.None()).unwrap();
            let call = for_call(call_ssl_verify(&kwargs).unwrap(), true);
            let settings =
                HttpSettings::from_layers([call, configured_ssl_verify(SslVerify::Disabled)]);
            assert_eq!(settings.ssl_verify, Some(SslVerify::Disabled));
        });
    }

    #[test]
    fn live_ssl_context_argument_raises_instead_of_using_another_layer() {
        Python::initialize();
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            let ssl = py.import("ssl").unwrap();
            let context = ssl
                .getattr("SSLContext")
                .unwrap()
                .call1((ssl.getattr("PROTOCOL_TLS_CLIENT").unwrap(),))
                .unwrap();
            kwargs.set_item("ssl_verify", context).unwrap();
            let error = call_ssl_verify(&kwargs).unwrap_err();
            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(error.to_string().contains("request.ssl_verify"));
            assert!(error.to_string().contains("SSLContext"));
        });
    }

    #[test]
    fn url_policy_uses_truthiness_and_normalized_owned_hosts() {
        Python::initialize();
        Python::attach(|py| {
            let value = py.eval(c"__import__('types').SimpleNamespace(user_url_validation=[], user_url_allowed_hosts=['B.test', 'a.test.', 'b.test'])", None, None).unwrap();
            assert_eq!(
                project_url_policy(&PythonSettings::UrlPolicy.snapshot(value)).unwrap(),
                UrlPolicy {
                    validate: false,
                    allowed_hosts: vec!["a.test".into(), "b.test".into()],
                }
            );
        });
    }

    #[rstest]
    #[case::asynchronous(true, false)]
    #[case::synchronous(false, true)]
    fn synchronous_calls_honor_environment_proxies_even_when_aiohttp_opts_out(
        #[case] asynchronous: bool,
        #[case] expected: bool,
    ) {
        let opted_out = HttpSettingsLayer {
            disable_aiohttp_trust_env: Some(true),
            disable_aiohttp_transport: Some(false),
            ..HttpSettingsLayer::default()
        };
        let settings = HttpSettings::from_layers([for_call(None, asynchronous), opted_out]);
        assert_eq!(settings.trust_proxy_env, expected);
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
                decode_hosts(&Field::new(
                    "url_policy",
                    "user_url_allowed_hosts",
                    evaluate(py, source)
                ))
                .unwrap(),
                expected
            );
        });
    }

    #[test]
    fn projection_releases_the_source_collection() {
        Python::initialize();
        Python::attach(|py| {
            let source = evaluate(py, "['A.test']");
            let projected = decode_hosts(&Field::new("test", "hosts", source.clone())).unwrap();
            source.call_method1("append", ("b.test",)).unwrap();
            assert_eq!(projected, ["a.test"]);
            assert_eq!(
                decode_hosts(&Field::new("test", "hosts", source)).unwrap(),
                ["a.test", "b.test"]
            );
        });
    }
}
