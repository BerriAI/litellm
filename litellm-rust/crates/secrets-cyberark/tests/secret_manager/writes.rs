use super::*;

#[rstest]
#[tokio::test]
async fn rejected_write_token_is_reauthenticated_once() {
    let server = MockServer::start().await;
    mount_auth(&server, 2).await;
    Mock::given(path("/policies/acct/policy/root"))
        .respond_with(ResponseTemplate::new(201))
        .expect(1)
        .mount(&server)
        .await;
    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_for_response = Arc::clone(&attempts);
    Mock::given(method("POST"))
        .and(path("/secrets/acct/variable/key"))
        .and(body_string("value"))
        .respond_with(move |_: &Request| {
            if attempts_for_response.fetch_add(1, Ordering::SeqCst) == 0 {
                ResponseTemplate::new(401)
            } else {
                ResponseTemplate::new(200)
            }
        })
        .expect(2)
        .mount(&server)
        .await;
    let manager = manager(&server, Duration::from_secs(600));

    manager
        .async_write_secret("key", &SecretValue::new("value"), None)
        .await
        .unwrap();
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    assert_eq!(
        manager
            .async_read_secret("key")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "value"
    );
}

#[rstest]
#[case::created(201)]
#[case::unprocessable(422)]
#[case::server_error(500)]
#[tokio::test]
async fn writes_tolerate_policy_status_and_cache_value(#[case] policy_status: u16) {
    let server = MockServer::start().await;
    mount_auth(&server, 1).await;
    Mock::given(path("/policies/acct/policy/root"))
        .and(header("content-type", "application/x-yaml"))
        .and(body_string("- !variable \"team/app\"\n"))
        .respond_with(ResponseTemplate::new(policy_status))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(path("/secrets/acct/variable/team%2Fapp"))
        .and(body_string("v"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;
    let manager = manager(&server, Duration::from_secs(60));
    manager
        .async_write_secret("team/app", &SecretValue::new("v"), None)
        .await
        .unwrap();
    assert_eq!(
        manager
            .async_read_secret("team/app")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "v"
    );
}

#[rstest]
#[tokio::test]
async fn policy_load_conflict_is_retried_before_the_value_write() {
    let server = MockServer::start().await;
    mount_auth(&server, 1).await;
    let policy_loads = Arc::new(AtomicUsize::new(0));
    let policy_loads_for_response = Arc::clone(&policy_loads);
    Mock::given(path("/policies/acct/policy/root"))
        .respond_with(move |_: &Request| {
            if policy_loads_for_response.fetch_add(1, Ordering::SeqCst) < 2 {
                ResponseTemplate::new(409)
            } else {
                ResponseTemplate::new(201)
            }
        })
        .expect(3)
        .mount(&server)
        .await;
    let policy_loads_at_value_write = Arc::clone(&policy_loads);
    Mock::given(method("POST"))
        .and(path("/secrets/acct/variable/key"))
        .respond_with(move |_: &Request| {
            if policy_loads_at_value_write.load(Ordering::SeqCst) == 3 {
                ResponseTemplate::new(201)
            } else {
                ResponseTemplate::new(404)
            }
        })
        .expect(1)
        .mount(&server)
        .await;
    let manager = manager(&server, Duration::from_secs(60));

    manager
        .async_write_secret("key", &SecretValue::new("v"), None)
        .await
        .unwrap();
}

#[rstest]
#[tokio::test]
async fn failed_value_write_is_not_cached() {
    let server = MockServer::start().await;
    mount_auth(&server, 1).await;
    Mock::given(path("/policies/acct/policy/root"))
        .respond_with(ResponseTemplate::new(201))
        .mount(&server)
        .await;
    Mock::given(path("/secrets/acct/variable/key"))
        .and(body_string("v"))
        .respond_with(ResponseTemplate::new(403))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(path("/secrets/acct/variable/key"))
        .respond_with(ResponseTemplate::new(200).set_body_string("recovered"))
        .expect(1)
        .mount(&server)
        .await;
    let manager = manager(&server, Duration::from_secs(60));
    assert!(matches!(
        manager
            .async_write_secret("key", &SecretValue::new("v"), None)
            .await,
        Err(Error::Status(403))
    ));
    assert_eq!(
        manager
            .async_read_secret("key")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "recovered"
    );
}

#[rstest]
#[tokio::test]
async fn delete_invalidates_cache_and_reports_not_supported() {
    let server = MockServer::start().await;
    mount_auth(&server, 1).await;
    Mock::given(path("/secrets/acct/variable/key"))
        .respond_with(ResponseTemplate::new(200).set_body_string("v"))
        .expect(2)
        .mount(&server)
        .await;
    let manager = manager(&server, Duration::from_secs(60));
    assert_eq!(
        manager
            .async_read_secret("key")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "v"
    );
    assert_eq!(
        manager.async_delete_secret("key", Some(7)).await.unwrap(),
        DeleteOutcome::NotSupported
    );
    assert_eq!(
        manager
            .async_read_secret("key")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "v"
    );
}

#[rstest]
#[tokio::test]
async fn writes_match_python_parity_fixture(parity_fixture: ParityFixture) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(&parity_fixture.authenticate_path))
        .and(body_string(&parity_fixture.api_key))
        .respond_with(ResponseTemplate::new(200).set_body_string(&parity_fixture.token_json))
        .expect(1)
        .mount(&server)
        .await;
    let manager = CyberArkSecretManager::with_client(
        litellm_http::Client::plain_for_test(),
        server.uri().parse().unwrap(),
        parity_fixture.account,
        parity_fixture.username,
        SecretValue::new(parity_fixture.api_key),
        Some(Duration::from_secs(60)),
    );
    for secret in parity_fixture.secrets {
        Mock::given(method("POST"))
            .and(path(&parity_fixture.policy_path))
            .and(header(
                "authorization",
                &parity_fixture.authorization_header,
            ))
            .and(header("content-type", "application/x-yaml"))
            .and(body_string(&secret.policy_body))
            .respond_with(ResponseTemplate::new(201))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(RawPath(secret.path))
            .and(header(
                "authorization",
                &parity_fixture.authorization_header,
            ))
            .and(body_string("value"))
            .respond_with(ResponseTemplate::new(201))
            .expect(1)
            .mount(&server)
            .await;
        manager
            .async_write_secret(&secret.name, &SecretValue::new("value"), None)
            .await
            .unwrap();
        assert_eq!(
            manager
                .async_read_secret(&secret.name)
                .await
                .unwrap()
                .unwrap()
                .expose(),
            "value"
        );
    }
}

#[rstest]
#[tokio::test]
#[ignore]
async fn live_conjur_round_trip() {
    let endpoint: reqwest::Url = std::env::var("CYBERARK_API_BASE").unwrap().parse().unwrap();
    let account = std::env::var("CYBERARK_ACCOUNT").unwrap();
    let username = std::env::var("CYBERARK_USERNAME").unwrap();
    let api_key = SecretValue::new(std::env::var("CYBERARK_API_KEY").unwrap());
    let name = format!(
        "{}-{}",
        std::env::var("LITELLM_CONJUR_LIVE_SECRET_NAME").unwrap(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let manager = CyberArkSecretManager::with_client(
        litellm_http::Client::plain_for_test(),
        endpoint.clone(),
        account.clone(),
        username.clone(),
        api_key.clone(),
        Some(Duration::from_secs(60)),
    );

    assert!(manager.async_read_secret(&name).await.unwrap().is_none());
    for expected in ["first-π\n", " second-π "] {
        manager
            .async_write_secret(&name, &SecretValue::new(expected), None)
            .await
            .unwrap();
        let verifier = CyberArkSecretManager::with_client(
            litellm_http::Client::plain_for_test(),
            endpoint.clone(),
            account.clone(),
            username.clone(),
            api_key.clone(),
            Some(Duration::from_secs(60)),
        );
        assert_eq!(
            verifier
                .async_read_secret(&name)
                .await
                .unwrap()
                .unwrap()
                .expose(),
            expected
        );
    }
    manager
        .async_rotate_secret(&name, &name, &SecretValue::new("rotated-value"))
        .await
        .unwrap();
    let alias = format!("{name}-rotated");
    manager
        .async_rotate_secret(&name, &alias, &SecretValue::new("new-alias-value"))
        .await
        .unwrap();
    assert_eq!(
        manager
            .async_read_secret(&name)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "rotated-value"
    );
    assert_eq!(
        manager
            .async_read_secret(&alias)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "new-alias-value"
    );
}

#[rstest]
#[case::same_alias("old")]
#[case::new_alias("new")]
#[tokio::test]
async fn rotation_stores_the_replacement_and_retains_other_aliases(#[case] new_name: &'static str) {
    use std::sync::atomic::{AtomicBool, Ordering};
    let server = MockServer::start().await;
    mount_auth(&server, 1).await;
    let written = Arc::new(AtomicBool::new(false));
    let read_state = written.clone();
    Mock::given(method("GET"))
        .respond_with(move |request: &wiremock::Request| {
            let name = request.url.path().rsplit('/').next().unwrap();
            let value = if name == new_name && read_state.load(Ordering::SeqCst) {
                "new-value"
            } else {
                "old-value"
            };
            ResponseTemplate::new(200).set_body_string(value)
        })
        .expect(if new_name == "old" { 2 } else { 3 })
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/policies/acct/policy/root"))
        .and(body_string(format!("- !variable \"{new_name}\"\n")))
        .respond_with(ResponseTemplate::new(201))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(format!("/secrets/acct/variable/{new_name}")))
        .and(body_string("new-value"))
        .respond_with(move |_: &wiremock::Request| {
            written.store(true, Ordering::SeqCst);
            ResponseTemplate::new(201)
        })
        .expect(1)
        .mount(&server)
        .await;
    let manager = manager(&server, Duration::from_secs(60));
    manager
        .async_rotate_secret("old", new_name, &SecretValue::new("new-value"))
        .await
        .unwrap();
    assert_eq!(
        manager
            .async_read_secret(new_name)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "new-value"
    );
    assert_eq!(
        manager
            .async_read_secret("old")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        if new_name == "old" {
            "new-value"
        } else {
            "old-value"
        }
    );
    assert!(
        server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .all(|request| request.method != "DELETE")
    );
}

#[tokio::test]
async fn rotation_verifies_the_remote_value_instead_of_the_write_cache() {
    let server = MockServer::start().await;
    mount_auth(&server, 1).await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_string("unchanged"))
        .expect(2)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/policies/acct/policy/root"))
        .respond_with(ResponseTemplate::new(201))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/secrets/acct/variable/new"))
        .respond_with(ResponseTemplate::new(201))
        .expect(1)
        .mount(&server)
        .await;
    let result = manager(&server, Duration::from_secs(60))
        .async_rotate_secret("old", "new", &SecretValue::new("replacement"))
        .await;
    assert!(matches!(
        result,
        Err(litellm_secrets_types::RotationError::Verification {
            source: Error::Operation(litellm_secrets_types::Error::NewSecretMismatch),
            ..
        })
    ));
}

#[rstest]
#[case::colon("foo: bar")]
#[case::comment("foo # bar")]
#[case::plain("plain-alias")]
#[case::email("team/user@example.com")]
#[case::quote("needs \"quote\"")]
#[case::backslash("a\\b")]
#[tokio::test]
async fn policy_writes_preserve_yaml_metacharacters_as_one_variable(#[case] name: &'static str) {
    let server = MockServer::start().await;
    mount_auth(&server, 1).await;
    Mock::given(method("POST"))
        .and(path("/policies/acct/policy/root"))
        .respond_with(move |request: &wiremock::Request| {
            let body = std::str::from_utf8(&request.body).unwrap();
            let scalar = body
                .strip_prefix("- !variable ")
                .unwrap()
                .strip_suffix('\n')
                .unwrap();
            assert_eq!(serde_json::from_str::<String>(scalar).unwrap(), name);
            ResponseTemplate::new(201)
        })
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(body_string("value"))
        .respond_with(ResponseTemplate::new(201))
        .expect(1)
        .mount(&server)
        .await;
    let manager = manager(&server, Duration::from_secs(60));
    manager
        .async_write_secret(name, &SecretValue::new("value"), None)
        .await
        .unwrap();
    assert_eq!(
        manager
            .async_read_secret(name)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "value"
    );
}
