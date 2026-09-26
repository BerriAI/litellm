use std::{collections::BTreeMap, sync::Arc};

use litellm_core_utils::settings::Lookup;
use litellm_secrets::{Error, OidcResolver, Secret, SecretManagerState, SecretResolver};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{header, method, path, query_param},
};

fn environment(pairs: &[(&str, &str)]) -> Arc<dyn Lookup + Send + Sync> {
    let values: BTreeMap<String, String> = pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    Arc::new(move |name: &str| values.get(name).cloned())
}

#[rstest::rstest]
#[case::environment("oidc/env/TOKEN", "true")]
#[case::circleci("oidc/circleci/audience", "circle")]
#[case::circleci_v2("oidc/circleci_v2/audience", "circle-v2")]
#[tokio::test]
async fn environment_sources_resolve_expected_value(
    #[case] reference: &str,
    #[case] expected: &str,
) {
    let env = environment(&[
        ("TOKEN", "true"),
        ("CIRCLE_OIDC_TOKEN", "circle"),
        ("CIRCLE_OIDC_TOKEN_V2", "circle-v2"),
    ]);
    assert_eq!(
        OidcResolver::new(litellm_http::Client::plain_for_test())
            .resolve(reference, env.as_ref())
            .await
            .unwrap()
            .unwrap()
            .expose(),
        expected
    );
}

#[tokio::test]
async fn environment_sources_bypass_boolean_conversion_and_defaults() {
    let env = environment(&[("TOKEN", "true")]);
    let oidc = OidcResolver::new(litellm_http::Client::plain_for_test());
    let resolver = SecretResolver::new(Arc::new(SecretManagerState::default()), env, oidc);
    assert_eq!(
        resolver
            .get_secret_str("os.environ/oidc/env/TOKEN", None)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "true"
    );
    assert_eq!(
        resolver
            .get_secret_bool("oidc/env/TOKEN", None)
            .await
            .unwrap(),
        Some(true)
    );
    assert!(matches!(
        resolver
            .get_secret("oidc/env/MISSING", Some(Secret::Bool(true)))
            .await,
        Err(Error::MissingEnvironment)
    ));
    assert!(matches!(
        resolver.get_secret("oidc/invalid", None).await,
        Err(Error::InvalidOidc)
    ));
}

#[tokio::test]
async fn github_requests_are_authenticated_cached_and_revalidate_environment() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/token"))
        .and(query_param("audience", "https://service/oidc/path"))
        .and(header("authorization", "Bearer request-token"))
        .and(header("accept", "application/json; api-version=2.0"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"value":"identity-token"})),
        )
        .expect(1)
        .mount(&server)
        .await;
    let env = environment(&[
        (
            "ACTIONS_ID_TOKEN_REQUEST_URL",
            &format!("{}/token", server.uri()),
        ),
        ("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "request-token"),
    ]);
    let oidc = OidcResolver::new(litellm_http::Client::plain_for_test());
    for _ in 0..2 {
        assert_eq!(
            oidc.resolve("oidc/github/https://service/oidc/path", env.as_ref())
                .await
                .unwrap()
                .unwrap()
                .expose(),
            "identity-token"
        );
    }
    assert!(matches!(
        oidc.resolve(
            "oidc/github/https://service/oidc/path",
            environment(&[]).as_ref()
        )
        .await,
        Err(Error::MissingEnvironment)
    ));
}

#[tokio::test]
async fn file_allowlist_resolves_symlinks_while_environment_paths_remain_explicit() {
    let allowed = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let token = allowed.path().join("token");
    let private = outside.path().join("private");
    std::fs::write(&token, "token\r\n").unwrap();
    std::fs::write(&private, "outside").unwrap();
    let env = environment(&[
        (
            "LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS",
            allowed.path().to_str().unwrap(),
        ),
        ("PATH_TOKEN", private.to_str().unwrap()),
        ("AZURE_FEDERATED_TOKEN_FILE", token.to_str().unwrap()),
    ]);
    let oidc = OidcResolver::new(litellm_http::Client::plain_for_test());
    assert_eq!(
        oidc.resolve(&format!("oidc/file/{}", token.display()), env.as_ref())
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "token\n"
    );
    assert!(matches!(
        oidc.resolve("oidc/file/relative", env.as_ref()).await,
        Err(Error::UnsafeOidcPath)
    ));
    assert!(matches!(
        oidc.resolve(&format!("oidc/file/{}", private.display()), env.as_ref())
            .await,
        Err(Error::UnsafeOidcPath)
    ));
    assert_eq!(
        oidc.resolve("oidc/env_path/PATH_TOKEN", env.as_ref())
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "outside"
    );
    assert_eq!(
        oidc.resolve("oidc/azure/scope", env.as_ref())
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "token\n"
    );
    #[cfg(unix)]
    {
        let link = allowed.path().join("link");
        std::os::unix::fs::symlink(&private, &link).unwrap();
        assert!(matches!(
            oidc.resolve(&format!("oidc/file/{}", link.display()), env.as_ref())
                .await,
            Err(Error::UnsafeOidcPath)
        ));
    }
}

#[cfg(feature = "google")]
#[rstest::rstest]
#[case::at_refresh_boundary(serde_json::json!(1060), 2)]
#[case::beyond_refresh_boundary(serde_json::json!(1061), 1)]
#[case::already_expired(serde_json::json!(999), 2)]
#[case::string_expiry(serde_json::json!("999"), 2)]
#[case::fractional_expiry(serde_json::json!(1060.9), 2)]
#[case::negative_expiry(serde_json::json!(-1), 2)]
#[case::boolean_expiry(serde_json::json!(true), 2)]
#[case::padded_numeric_expiry(serde_json::json!(" 999 "), 2)]
#[case::null_expiry(serde_json::Value::Null, 1)]
#[case::unreadable_expiry(serde_json::json!("invalid"), 1)]
#[case::nonfinite_expiry(serde_json::json!("NaN"), 1)]
#[tokio::test]
async fn google_expiry_caps_cache_and_preserves_audience(
    #[case] expiry: serde_json::Value,
    #[case] calls: u64,
) {
    use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};
    fn now() -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(1000)
    }
    let server = MockServer::start().await;
    let token = format!(
        "{}.{}.signature",
        URL_SAFE_NO_PAD.encode(serde_json::json!({"alg":"RS256","typ":"JWT"}).to_string()),
        URL_SAFE_NO_PAD.encode(serde_json::json!({"exp":expiry}).to_string())
    );
    Mock::given(method("GET"))
        .and(header("metadata-flavor", "Google"))
        .and(query_param("audience", "https://service/oidc/path"))
        .respond_with(ResponseTemplate::new(200).set_body_string(&token))
        .expect(calls)
        .mount(&server)
        .await;
    let oidc = OidcResolver::new(litellm_http::Client::plain_for_test())
        .with_google_identity_endpoint(server.uri().parse().unwrap())
        .with_clock(now);
    for _ in 0..2 {
        assert_eq!(
            oidc.resolve(
                "oidc/google/https://service/oidc/path",
                environment(&[]).as_ref()
            )
            .await
            .unwrap()
            .unwrap()
            .expose(),
            token
        );
    }
}

#[cfg(not(feature = "google"))]
#[tokio::test]
async fn google_oidc_requires_its_build_feature() {
    assert!(matches!(
        OidcResolver::new(litellm_http::Client::plain_for_test())
            .resolve("oidc/google/audience", environment(&[]).as_ref())
            .await,
        Err(Error::UnsupportedOidc)
    ));
}

#[cfg(not(feature = "azure"))]
#[tokio::test]
async fn azure_oidc_without_a_token_file_requires_its_build_feature() {
    assert!(matches!(
        OidcResolver::new(litellm_http::Client::plain_for_test())
            .resolve("oidc/azure/scope", environment(&[]).as_ref())
            .await,
        Err(Error::UnsupportedOidc)
    ));
}

#[rstest::rstest]
#[case::missing_prefix("env/TOKEN", false)]
#[case::missing_audience_separator("oidc/env", false)]
#[case::unknown_provider("oidc/unknown/TOKEN", true)]
#[tokio::test]
async fn invalid_references_fail_before_environment_lookup(
    #[case] reference: &str,
    #[case] unsupported: bool,
) {
    let error = OidcResolver::new(litellm_http::Client::plain_for_test())
        .resolve(reference, &|_: &str| {
            panic!("invalid reference reached environment lookup")
        })
        .await
        .unwrap_err();
    assert!(matches!(error, Error::UnsupportedOidc) == unsupported);
    assert!(matches!(error, Error::InvalidOidc) != unsupported);
}

#[cfg(feature = "google")]
#[rstest::rstest]
#[case::opaque("opaque-token")]
#[case::missing_expiry("header.e30.signature")]
#[tokio::test]
async fn unreadable_expiry_keeps_python_cache_fallback(#[case] token: &str) {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_string(token))
        .expect(1)
        .mount(&server)
        .await;
    let resolver = OidcResolver::new(litellm_http::Client::plain_for_test())
        .with_google_identity_endpoint(server.uri().parse().unwrap());
    for _ in 0..2 {
        assert_eq!(
            resolver
                .resolve("oidc/google/audience", environment(&[]).as_ref())
                .await
                .unwrap()
                .unwrap()
                .expose(),
            token,
        );
    }
}

#[cfg(feature = "azure")]
#[rstest::rstest]
#[case::success(false)]
#[case::failed(true)]
#[tokio::test]
async fn azure_oidc_acquires_the_requested_scope_and_preserves_failures(#[case] failed: bool) {
    struct Provider(bool);
    impl litellm_secrets::azure::AzureTokenProvider for Provider {
        fn get_token<'a>(
            &'a self,
            scope: &'a str,
            environment: &'a (dyn Lookup + Send + Sync),
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<
                            litellm_secrets::SecretValue,
                            litellm_secrets::azure::Error,
                        >,
                    > + Send
                    + 'a,
            >,
        > {
            Box::pin(async move {
                assert_eq!(scope, "api://audience/path");
                assert_eq!(
                    environment.get("AZURE_CLIENT_ID").as_deref(),
                    Some("client-id")
                );
                if self.0 {
                    Err(litellm_secrets::azure::Error::MissingCredentials)
                } else {
                    Ok(litellm_secrets::SecretValue::new("azure-token"))
                }
            })
        }
    }
    let oidc = OidcResolver::new(litellm_http::Client::plain_for_test())
        .with_azure_token_provider(Arc::new(Provider(failed)));
    let resolver = SecretResolver::new_python_compatible(
        Arc::new(SecretManagerState::default()),
        environment(&[("AZURE_CLIENT_ID", "client-id")]),
        oidc,
    );
    let result = resolver
        .get_secret_str(
            "oidc/azure/api://audience/path",
            Some(litellm_secrets::SecretValue::new("fallback")),
        )
        .await;
    if failed {
        assert!(matches!(result, Err(Error::Azure(_))));
    } else {
        assert_eq!(result.unwrap().unwrap().expose(), "azure-token");
    }
}

#[rstest::rstest]
#[case::circleci("oidc/circleci/audience")]
#[case::circleci_v2("oidc/circleci_v2/audience")]
#[case::env("oidc/env/MISSING")]
#[case::env_path("oidc/env_path/MISSING")]
#[tokio::test]
async fn missing_oidc_environment_is_an_error(#[case] reference: &str) {
    assert!(matches!(
        OidcResolver::new(litellm_http::Client::plain_for_test())
            .resolve(reference, environment(&[]).as_ref())
            .await,
        Err(Error::MissingEnvironment)
    ));
}

#[cfg(feature = "google")]
#[tokio::test]
async fn google_oidc_failures_are_not_cached_or_hidden_by_defaults() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(403))
        .expect(2)
        .mount(&server)
        .await;
    let resolver = SecretResolver::new_python_compatible(
        Arc::new(SecretManagerState::default()),
        environment(&[]),
        OidcResolver::new(litellm_http::Client::plain_for_test())
            .with_google_identity_endpoint(server.uri().parse().unwrap()),
    );
    for _ in 0..2 {
        assert!(matches!(
            resolver
                .get_secret("oidc/google/audience", Some(Secret::Bool(true)))
                .await,
            Err(Error::OidcStatus(403))
        ));
    }
}

#[cfg(feature = "google")]
#[rstest::rstest]
#[case::short_lived(Some(1180), 120)]
#[case::long_lived(Some(100000), 3540)]
#[case::opaque(None, 3540)]
#[tokio::test]
async fn google_tokens_expire_at_the_python_cache_deadline(
    #[case] expiry: Option<u64>,
    #[case] ttl: u64,
) {
    use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
    use std::time::{Duration, UNIX_EPOCH};
    let server = MockServer::start().await;
    let token = expiry.map_or_else(
        || "opaque-token".to_owned(),
        |expiry| {
            format!(
                "{}.{}.signature",
                URL_SAFE_NO_PAD.encode(r#"{"alg":"RS256"}"#),
                URL_SAFE_NO_PAD.encode(serde_json::json!({"exp":expiry}).to_string())
            )
        },
    );
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_string(&token))
        .expect(2)
        .mount(&server)
        .await;
    let resolver = OidcResolver::new(litellm_http::Client::plain_for_test())
        .with_google_identity_endpoint(server.uri().parse().unwrap())
        .with_clock(|| UNIX_EPOCH + Duration::from_secs(1000));
    assert_eq!(
        resolver
            .resolve("oidc/google/audience", environment(&[]).as_ref())
            .await
            .unwrap()
            .unwrap()
            .expose(),
        token
    );
    let before_deadline = match ttl {
        120 => resolver.with_clock(|| UNIX_EPOCH + Duration::from_secs(1119)),
        3540 => resolver.with_clock(|| UNIX_EPOCH + Duration::from_secs(4539)),
        _ => unreachable!(),
    };
    assert_eq!(
        before_deadline
            .resolve("oidc/google/audience", environment(&[]).as_ref())
            .await
            .unwrap()
            .unwrap()
            .expose(),
        token
    );
    let at_deadline = match ttl {
        120 => before_deadline.with_clock(|| UNIX_EPOCH + Duration::from_secs(1120)),
        3540 => before_deadline.with_clock(|| UNIX_EPOCH + Duration::from_secs(4540)),
        _ => unreachable!(),
    };
    assert_eq!(
        at_deadline
            .resolve("oidc/google/audience", environment(&[]).as_ref())
            .await
            .unwrap()
            .unwrap()
            .expose(),
        token
    );
}

#[cfg(feature = "google")]
#[tokio::test]
async fn google_cache_uses_payload_expiry_without_requiring_a_jwt_header() {
    use std::time::{Duration, UNIX_EPOCH};
    let server = MockServer::start().await;
    let token = "ignored.eyJleHAiOjF9.ignored";
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_string(token))
        .expect(2)
        .mount(&server)
        .await;
    let resolver = OidcResolver::new(litellm_http::Client::plain_for_test())
        .with_google_identity_endpoint(server.uri().parse().unwrap())
        .with_clock(|| UNIX_EPOCH + Duration::from_secs(1000));
    for _ in 0..2 {
        assert_eq!(
            resolver
                .resolve("oidc/google/audience", environment(&[]).as_ref())
                .await
                .unwrap()
                .unwrap()
                .expose(),
            token
        );
    }
}
