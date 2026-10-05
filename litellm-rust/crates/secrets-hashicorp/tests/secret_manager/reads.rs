use super::*;

#[rstest]
#[tokio::test]
async fn token_reads_use_vault_headers_and_cache_values(token_values: Vec<(&str, &str)>) {
    let server: MockServer = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/name"))
        .and(header("X-Vault-Token", "token"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(read_response(json!({"key": "value"}))),
        )
        .expect(1)
        .mount(&server)
        .await;
    let manager: HashicorpVault = manager(&server, &token_values);

    assert_eq!(
        manager
            .async_read_secret("name")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "value"
    );
    let requests = server.received_requests().await.unwrap();
    assert!(
        requests
            .iter()
            .all(|request| !request.headers.contains_key("X-Vault-Namespace"))
    );
    assert_eq!(
        manager
            .async_read_secret("name")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "value"
    );
}

#[rstest]
#[tokio::test]
async fn namespace_mount_and_prefix_are_sanitized_in_the_url() {
    let server: MockServer = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/kv-prod/data/virtual-keys/name"))
        .and(header("X-Vault-Namespace", "team-a"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(read_response(json!({"key": "value"}))),
        )
        .expect(1)
        .mount(&server)
        .await;
    let manager: HashicorpVault = manager(
        &server,
        &[
            ("HCP_VAULT_TOKEN", "token"),
            ("HCP_VAULT_SECRET_NAMESPACE", " /team-a/ "),
            ("HCP_VAULT_MOUNT_NAME", " /kv-prod/ "),
            ("HCP_VAULT_PATH_PREFIX", " /virtual-keys/ "),
        ],
    );

    let location = manager.secret_location("name").unwrap();
    assert_eq!(location.namespace.as_deref(), Some("team-a"));
    assert_eq!(location.mount, "kv-prod");
    assert_eq!(location.path, "virtual-keys/name");
    assert!(manager.async_read_secret("name").await.unwrap().is_some());
}

#[rstest]
#[tokio::test]
async fn approle_tokens_expire_after_the_vault_lease() {
    let server: MockServer = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/auth/approle/login"))
        .respond_with(ResponseTemplate::new(200).set_body_json(auth_response("login-token", 1)))
        .expect(2)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(read_response(json!({"key": "value"}))),
        )
        .expect(2)
        .mount(&server)
        .await;
    let manager: HashicorpVault = manager(
        &server,
        &[
            ("HCP_VAULT_APPROLE_ROLE_ID", "role"),
            ("HCP_VAULT_APPROLE_SECRET_ID", "secret"),
            ("HCP_VAULT_REFRESH_INTERVAL", "0"),
        ],
    );

    assert!(manager.async_read_secret("first").await.unwrap().is_some());
    tokio::time::sleep(Duration::from_secs(1) + Duration::from_millis(50)).await;
    assert!(manager.async_read_secret("second").await.unwrap().is_some());
}

#[derive(Clone, Copy)]
enum ExpectedRead {
    Missing,
    Malformed,
    NonString,
}

#[rstest]
#[case::missing(404, json!({"errors": ["missing"]}), ExpectedRead::Missing)]
#[case::malformed(200, json!({"data": "invalid"}), ExpectedRead::Malformed)]
#[case::missing_key(200, json!({}), ExpectedRead::Missing)]
#[case::non_string(200, json!({"key": 1}), ExpectedRead::NonString)]
#[tokio::test]
async fn read_responses_distinguish_absence_and_malformed_payloads(
    token_values: Vec<(&str, &str)>,
    #[case] status: u16,
    #[case] body: serde_json::Value,
    #[case] expected: ExpectedRead,
) {
    let server: MockServer = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(status).set_body_json(
            if status == 200 && !matches!(expected, ExpectedRead::Malformed) {
                read_response(body)
            } else {
                body
            },
        ))
        .expect(1)
        .mount(&server)
        .await;
    let result: Result<Option<SecretValue>, Error> = manager(&server, &token_values)
        .async_read_secret("name")
        .await;
    match expected {
        ExpectedRead::Missing => assert!(result.unwrap().is_none()),
        ExpectedRead::Malformed => assert!(matches!(result, Err(Error::MalformedPayload))),
        ExpectedRead::NonString => assert!(matches!(result, Err(Error::NonStringValue))),
    }
}

#[rstest]
#[tokio::test]
async fn base_manager_context_overrides_vault_location_and_data_key(
    token_values: Vec<(&str, &str)>,
) {
    let server: MockServer = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/alternate/data/managed/name"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(read_response(json!({"api_token": "value"}))),
        )
        .expect(2)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/alternate/data/managed/name"))
        .and(body_json(json!({
            "data": {"api_token": "updated", "description": "Managed key"}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {
                "created_time": "",
                "deletion_time": "",
                "custom_metadata": null,
                "destroyed": false,
                "version": 2
            },
            "lease_id": "",
            "lease_duration": 0,
            "renewable": false,
            "request_id": "",
            "warnings": null,
            "wrap_info": null
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/v1/alternate/data/managed/name"))
        .respond_with(ResponseTemplate::new(204))
        .expect(1)
        .mount(&server)
        .await;
    let manager: HashicorpVault = manager(&server, &token_values);
    let operation = HashicorpOperationContext {
        mount: Some(" /alternate/ ".to_owned()),
        path_prefix: Some(" /managed/ ".to_owned()),
        data_key: Some("api_token".to_owned()),
        ..HashicorpOperationContext::default()
    };
    let write_context = SecretWriteContext {
        description: Some("Managed key".to_owned()),
        operation: operation.clone(),
        ..SecretWriteContext::default()
    };

    assert_eq!(
        BaseSecretManager::async_read_secret(&manager, "name", &operation)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "value"
    );
    SecretWriter::async_write_secret(
        &manager,
        "name",
        &SecretValue::new("updated"),
        &write_context,
    )
    .await
    .unwrap();
    assert!(
        BaseSecretManager::async_read_secret(&manager, "name", &operation)
            .await
            .unwrap()
            .is_some()
    );
    SecretDeleter::async_delete_secret(&manager, "name", &operation)
        .await
        .unwrap();
}

#[rstest]
#[tokio::test]
async fn rejects_description_that_would_replace_the_secret_value(token_values: Vec<(&str, &str)>) {
    let server: MockServer = MockServer::start().await;
    let manager: HashicorpVault = manager(&server, &token_values);
    let context = SecretWriteContext {
        description: Some("metadata".to_owned()),
        operation: HashicorpOperationContext {
            data_key: Some("description".to_owned()),
            ..HashicorpOperationContext::default()
        },
        ..SecretWriteContext::default()
    };

    assert!(matches!(
        manager
            .async_write_secret_with_context("name", &SecretValue::new("secret"), &context)
            .await,
        Err(Error::DataKeyConflictsWithDescription)
    ));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[tokio::test]
async fn reads_cache_each_data_key_for_the_same_vault_path(token_values: Vec<(&str, &str)>) {
    let server: MockServer = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/name"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(read_response(json!({
                "key": "primary",
                "alternate": "secondary"
            }))),
        )
        .expect(2)
        .mount(&server)
        .await;
    let manager: HashicorpVault = manager(&server, &token_values);
    let alternate = HashicorpOperationContext {
        data_key: Some("alternate".to_owned()),
        ..HashicorpOperationContext::default()
    };

    assert_eq!(
        manager
            .async_read_secret("name")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "primary"
    );
    assert_eq!(
        BaseSecretManager::async_read_secret(&manager, "name", &alternate)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "secondary"
    );
    assert_eq!(
        manager
            .async_read_secret("name")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "primary"
    );
}

#[rstest]
#[tokio::test]
async fn base_manager_context_timeout_limits_vault_io(token_values: Vec<(&str, &str)>) {
    let server: MockServer = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/name"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(100))
                .set_body_json(read_response(json!({"key": "value"}))),
        )
        .expect(1)
        .mount(&server)
        .await;
    let manager: HashicorpVault = manager(&server, &token_values);
    let context = HashicorpOperationContext {
        timeout: Some(Duration::from_millis(10)),
        ..HashicorpOperationContext::default()
    };

    assert!(matches!(
        BaseSecretManager::async_read_secret(&manager, "name", &context).await,
        Err(Error::Timeout)
    ));
}

#[rstest]
#[tokio::test]
async fn concurrent_reads_share_a_load_but_verification_fetches_fresh(
    token_values: Vec<(&str, &str)>,
) {
    use litellm_secrets_types::SecretRotator;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let server = MockServer::start().await;
    let reads = AtomicUsize::new(0);
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/name"))
        .respond_with(move |_: &wiremock::Request| {
            let value = if reads.fetch_add(1, Ordering::SeqCst) == 0 {
                "old"
            } else {
                "new"
            };
            ResponseTemplate::new(200)
                .set_body_json(read_response(json!({"key": value})))
                .set_delay(Duration::from_millis(20))
        })
        .expect(2)
        .mount(&server)
        .await;
    let manager = manager(&server, &token_values);
    let (first, second) = tokio::join!(
        manager.async_read_secret("name"),
        manager.async_read_secret("name")
    );
    assert_eq!(first.unwrap().unwrap().expose(), "old");
    assert_eq!(second.unwrap().unwrap().expose(), "old");
    assert_eq!(
        manager
            .async_read_secret("name")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "old"
    );
    assert_eq!(
        manager
            .async_read_secret_fresh("name", &HashicorpOperationContext::default())
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "new"
    );
    assert_eq!(
        manager
            .async_read_secret("name")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "new"
    );
}
