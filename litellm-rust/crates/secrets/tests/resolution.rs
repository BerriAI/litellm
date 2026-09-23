use std::{future::Future, pin::Pin, sync::Arc};

use litellm_core_utils::settings::Lookup;
use litellm_secrets::{
    Error, ExternalSecretManager, FailurePolicy, KeyManagementSettings, KeyManagementSystem,
    OidcResolver, Secret, SecretManager, SecretManagerState, SecretResolver, SecretValue,
    secret_manager_would_be_consulted,
};

fn resolver(value: Option<&str>) -> SecretResolver {
    let value = value.map(str::to_owned);
    SecretResolver::new_python_compatible(
        Arc::new(SecretManagerState::default()),
        Arc::new(move |_: &str| value.clone()),
        OidcResolver::default(),
    )
}

#[rstest::rstest]
#[case::environment(false)]
#[case::manager(true)]
#[tokio::test]
async fn native_reads_preserve_strings_and_report_conversion_errors(#[case] managed: bool) {
    for raw in ["True", " FALSE ", "", "(True)", "{\"key\":1}"] {
        let state = if managed {
            SecretManagerState::new(
                SecretManager::External(Arc::new(FixedManager::custom(Ok(Some(Secret::String(
                    SecretValue::new(raw),
                )))))),
                KeyManagementSettings::default(),
            )
        } else {
            SecretManagerState::default()
        };
        let resolver = SecretResolver::new(
            Arc::new(state),
            Arc::new(move |_: &str| Some(raw.to_owned())),
            OidcResolver::default(),
        );
        assert_eq!(
            resolver
                .get_secret_str("key", None)
                .await
                .unwrap()
                .unwrap()
                .expose(),
            raw
        );
        assert_eq!(
            resolver.get_secret("key", None).await.unwrap(),
            Some(Secret::String(SecretValue::new(raw)))
        );
        if raw == "True" || raw == " FALSE " {
            assert_eq!(
                resolver.get_secret_bool("key", None).await.unwrap(),
                Some(raw == "True")
            );
        } else {
            assert!(matches!(
                resolver.get_secret_bool("key", None).await,
                Err(Error::TypeMismatch {
                    expected: "boolean"
                })
            ));
        }
    }
}

#[tokio::test]
async fn native_defaults_apply_to_absence_but_never_hide_provider_failures() {
    for reply in [Ok(None), Err(())] {
        let resolver = SecretResolver::new(
            Arc::new(SecretManagerState::new(
                SecretManager::External(Arc::new(FixedManager::custom(reply.clone()))),
                KeyManagementSettings::default(),
            )),
            Arc::new(|_: &str| None),
            OidcResolver::default(),
        );
        let result = resolver
            .get_secret_str("key", Some(SecretValue::new("default")))
            .await;
        match reply {
            Ok(None) => assert_eq!(result.unwrap().unwrap().expose(), "default"),
            Err(()) => assert!(matches!(result, Err(Error::MissingCiphertext))),
            Ok(Some(_)) => unreachable!(),
        }
    }
}

#[rstest::rstest]
#[case::lowercase_true("true", Some(true))]
#[case::padded_false(" FALSE ", Some(false))]
#[case::capitalized_true("True", Some(true))]
#[case::parenthesized("(True)", None)]
#[case::commented("False # comment", None)]
#[case::number("1", None)]
#[case::text("secret", None)]
#[tokio::test]
async fn environment_values_are_coerced_like_str_to_bool(
    #[case] input: &str,
    #[case] boolean: Option<bool>,
) {
    let resolver = resolver(Some(input));
    assert_eq!(
        resolver.get_secret("key", None).await.unwrap(),
        Some(boolean.map_or_else(|| Secret::String(SecretValue::new(input)), Secret::Bool))
    );
    assert_eq!(
        resolver
            .get_secret_str("key", None)
            .await
            .unwrap()
            .as_ref()
            .map(SecretValue::expose),
        boolean.is_none().then_some(input)
    );
    assert_eq!(
        resolver.get_secret_bool("key", Some(true)).await.unwrap(),
        boolean
    );
}

#[tokio::test]
async fn defaults_never_replace_an_absent_secret() {
    let missing = resolver(None);
    assert_eq!(
        missing
            .get_secret("key", Some(Secret::Bool(false)))
            .await
            .unwrap(),
        None
    );
    assert_eq!(
        missing.get_secret_bool("key", Some(false)).await.unwrap(),
        None
    );
    assert_eq!(
        missing
            .get_secret_str("key", Some(SecretValue::new("default")))
            .await
            .unwrap(),
        None
    );
    assert_eq!(
        resolver(Some(""))
            .get_secret_str("key", Some(SecretValue::new("default")))
            .await
            .unwrap()
            .unwrap()
            .expose(),
        ""
    );
}

struct FixedManager {
    reply: Result<Option<Secret>, ()>,
    system: KeyManagementSystem,
}

impl FixedManager {
    fn custom(reply: Result<Option<Secret>, ()>) -> Self {
        Self {
            reply,
            system: KeyManagementSystem::Custom,
        }
    }
}

impl ExternalSecretManager for FixedManager {
    fn system(&self) -> KeyManagementSystem {
        self.system
    }

    fn read_secret<'a>(
        &'a self,
        _name: &'a str,
        _settings: &'a KeyManagementSettings,
        _environment: &'a (dyn Lookup + Send + Sync),
    ) -> Pin<Box<dyn Future<Output = Result<Option<Secret>, Error>> + Send + 'a>> {
        Box::pin(async move { self.reply.clone().map_err(|()| Error::MissingCiphertext) })
    }
}

fn managed(reply: Result<Option<Secret>, ()>, environment: Option<&'static str>) -> SecretResolver {
    SecretResolver::new_python_compatible(
        Arc::new(SecretManagerState::new(
            SecretManager::External(Arc::new(FixedManager::custom(reply))),
            KeyManagementSettings::default(),
        )),
        Arc::new(move |_: &str| environment.map(str::to_owned)),
        OidcResolver::default(),
    )
    .with_failure_policy(FailurePolicy::EnvironmentFallback)
}

#[tokio::test]
async fn custom_manager_absence_uses_environment_instead_of_the_default() {
    assert_eq!(
        managed(Ok(None), Some("environment"))
            .get_secret("key", Some(Secret::Bool(true)))
            .await
            .unwrap(),
        Some(Secret::String(SecretValue::new("environment")))
    );
}

#[rstest::rstest]
#[case::capitalized_true("True", Some(Secret::Bool(true)), Some(true))]
#[case::parenthesized_false("(False)", Some(Secret::Bool(false)), Some(false))]
#[case::lowercase_true("true", None, Some(true))]
#[case::number("1", None, None)]
#[case::text("secret", None, None)]
#[tokio::test]
async fn manager_strings_are_coerced_like_literal_eval(
    #[case] input: &'static str,
    #[case] literal: Option<Secret>,
    #[case] boolean: Option<bool>,
) {
    let resolver = managed(Ok(Some(Secret::String(SecretValue::new(input)))), None);
    assert_eq!(
        resolver.get_secret("key", None).await.unwrap(),
        Some(
            literal
                .clone()
                .unwrap_or_else(|| Secret::String(SecretValue::new(input)))
        )
    );
    assert_eq!(
        resolver
            .get_secret_str("key", None)
            .await
            .unwrap()
            .as_ref()
            .map(SecretValue::expose),
        literal.is_none().then_some(input)
    );
    assert_eq!(
        resolver.get_secret_bool("key", None).await.unwrap(),
        boolean
    );
}

#[rstest::rstest]
#[case::boolean(Secret::Bool(false))]
#[case::object(Secret::from_json(serde_json::json!({"key": 1})))]
#[case::null(Secret::from_json(serde_json::Value::Null))]
#[tokio::test]
async fn non_string_manager_values_resolve_to_none(#[case] value: Secret) {
    let resolver = managed(Ok(Some(value)), Some("environment"));
    assert_eq!(resolver.get_secret("key", None).await.unwrap(), None);
    assert_eq!(resolver.get_secret_str("key", None).await.unwrap(), None);
    assert_eq!(resolver.get_secret_bool("key", None).await.unwrap(), None);
}

#[rstest::rstest]
#[case::capitalized_true(Some("True"), Some(Secret::Bool(true)))]
#[case::lowercase_true(Some("true"), Some(Secret::String(SecretValue::new("true"))))]
#[case::missing(None, None)]
#[tokio::test]
async fn manager_failures_fall_back_to_the_environment_like_literal_eval(
    #[case] environment: Option<&'static str>,
    #[case] expected: Option<Secret>,
) {
    assert_eq!(
        managed(Err(()), environment)
            .get_secret("key", Some(Secret::Bool(false)))
            .await
            .unwrap(),
        expected
    );
}

#[tokio::test]
async fn prefix_is_removed_once_and_resolved_from_environment() {
    let state = SecretManagerState::default();
    assert!(!secret_manager_would_be_consulted(
        &state,
        "os.environ/os.environ/KEY"
    ));
    let resolver = SecretResolver::new_python_compatible(
        Arc::new(state),
        Arc::new(|name: &str| (name == "os.environ/KEY").then(|| "value".into())),
        OidcResolver::default(),
    );
    assert_eq!(
        resolver
            .get_secret_str("os.environ/os.environ/KEY", None)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "value"
    );
}

#[tokio::test]
async fn resolver_future_can_run_on_a_tokio_worker() {
    let resolver = resolver(Some("worker-value"));
    let result = tokio::spawn(async move { resolver.get_secret_str("KEY", None).await })
        .await
        .unwrap()
        .unwrap();
    assert_eq!(result.unwrap().expose(), "worker-value");
}

#[rstest::rstest]
#[case::missing(None, None)]
#[case::empty(Some(""), None)]
#[case::whitespace(Some("   \t\n"), None)]
#[case::text(Some("abc"), Some("abc"))]
#[case::padded(Some("  xyz  "), Some("xyz"))]
#[case::python_controls(Some("\u{1c}\u{1d}\u{1e}\u{1f}"), None)]
#[case::unicode(Some("\u{a0}π\u{2003}"), Some("π"))]
fn normalization_matches_python_without_changing_embedded_whitespace(
    #[case] input: Option<&str>,
    #[case] expected: Option<&str>,
) {
    assert_eq!(
        litellm_secrets::normalize_nonempty_secret_str(input),
        expected
    );
}

#[rstest::rstest]
#[case::lowercase("true", false)]
#[case::capitalized("True", true)]
#[case::literal("(True)", true)]
#[tokio::test]
async fn excluded_hosted_keys_keep_the_python_manager_conversion_path(
    #[case] raw: &'static str,
    #[case] boolean: bool,
) {
    let resolver = SecretResolver::new_python_compatible(
        Arc::new(SecretManagerState::new(
            SecretManager::External(Arc::new(FixedManager::custom(Err(())))),
            KeyManagementSettings {
                hosted_keys: Some(vec!["OTHER".into()]),
                ..Default::default()
            },
        )),
        Arc::new(move |_: &str| Some(raw.to_owned())),
        OidcResolver::default(),
    );
    assert_eq!(
        resolver.get_secret("KEY", None).await.unwrap(),
        Some(if boolean {
            Secret::Bool(true)
        } else {
            Secret::String(SecretValue::new(raw))
        })
    );
}

#[rstest::rstest]
#[case::missing(Ok(None), None)]
#[case::empty(
    Ok(Some(Secret::String(SecretValue::new("")))),
    Some(Secret::String(SecretValue::new("")))
)]
#[case::failed(Err(()), Some(Secret::String(SecretValue::new("environment"))))]
#[tokio::test]
async fn azure_callback_absence_preserves_none_but_errors_fall_back(
    #[case] reply: Result<Option<Secret>, ()>,
    #[case] expected: Option<Secret>,
) {
    let resolver = SecretResolver::new_python_compatible(
        Arc::new(SecretManagerState::new(
            SecretManager::External(Arc::new(FixedManager {
                reply,
                system: KeyManagementSystem::AzureKeyVault,
            })),
            KeyManagementSettings::default(),
        )),
        Arc::new(|_: &str| Some("environment".into())),
        OidcResolver::default(),
    );
    assert_eq!(
        resolver
            .get_secret("key", Some(Secret::String(SecretValue::new("default"))))
            .await
            .unwrap(),
        expected
    );
}
