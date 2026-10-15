use super::*;

#[rstest]
#[tokio::test]
async fn write_and_delete_invalidate_the_read_cache(token_values: Vec<(&str, &str)>) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let server = MockServer::start().await;
    let revision = Arc::new(AtomicUsize::new(0));
    let current = revision.clone();
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/name"))
        .respond_with(
            move |_: &wiremock::Request| match current.load(Ordering::SeqCst) {
                0 => ResponseTemplate::new(200).set_body_json(read_response(
                    json!({"key": "old", "alternate": "old-alternate"}),
                )),
                1 => ResponseTemplate::new(200).set_body_json(read_response(
                    json!({"key": "updated", "alternate": "updated-alternate"}),
                )),
                _ => ResponseTemplate::new(404).set_body_json(json!({"errors": ["missing"]})),
            },
        )
        .expect(5)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/unrelated"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(read_response(json!({"key": "unrelated"}))),
        )
        .expect(1)
        .mount(&server)
        .await;
    let written = revision.clone();
    Mock::given(method("POST"))
        .and(path("/v1/secret/data/name"))
        .respond_with(move |_: &wiremock::Request| {
            written.store(1, Ordering::SeqCst);
            ResponseTemplate::new(200).set_body_json(write_response(2))
        })
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/v1/secret/data/name"))
        .respond_with(move |_: &wiremock::Request| {
            revision.store(2, Ordering::SeqCst);
            ResponseTemplate::new(204)
        })
        .expect(1)
        .mount(&server)
        .await;
    let manager = manager(&server, &token_values);
    let alternate = HashicorpOperationContext {
        data_key: Some("alternate".into()),
        ..Default::default()
    };
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
        BaseSecretManager::async_read_secret(&manager, "name", &alternate)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "old-alternate"
    );
    assert_eq!(
        manager
            .async_read_secret("unrelated")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "unrelated"
    );
    manager
        .async_write_secret("name", SecretValue::new("updated"), None)
        .await
        .unwrap();
    assert_eq!(
        manager
            .async_read_secret("name")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "updated"
    );
    assert_eq!(
        BaseSecretManager::async_read_secret(&manager, "name", &alternate)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "updated-alternate"
    );
    manager.async_delete_secret("name").await.unwrap();
    assert!(manager.async_read_secret("name").await.unwrap().is_none());
    assert_eq!(
        manager
            .async_read_secret("unrelated")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "unrelated"
    );
}

#[rstest]
#[case::existing(200, 2)]
#[case::new_secret(404, 0)]
#[tokio::test]
async fn cas_required_writes_retry_with_the_current_version(
    token_values: Vec<(&str, &str)>,
    #[case] metadata_status: u16,
    #[case] expected_cas: u64,
) {
    let server: MockServer = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/secret/data/name"))
        .and(body_json(json!({"data": {"key": "value"}})))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({"errors": ["CAS required"]})))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/metadata/name"))
        .respond_with(
            ResponseTemplate::new(metadata_status).set_body_json(metadata_response(expected_cas)),
        )
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/secret/data/name"))
        .and(body_json(json!({
            "data": {"key": "value"},
            "options": {"cas": expected_cas}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(write_response(expected_cas + 1)))
        .expect(1)
        .mount(&server)
        .await;

    let result = manager(&server, &token_values)
        .async_write_secret("name", SecretValue::new("value"), None)
        .await;

    assert!(result.is_ok());
}

#[rstest]
#[tokio::test]
async fn failed_cas_lookup_preserves_the_write_error(token_values: Vec<(&str, &str)>) {
    let server: MockServer = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/secret/data/name"))
        .respond_with(
            ResponseTemplate::new(400).set_body_json(json!({"errors": ["write rejected"]})),
        )
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/metadata/name"))
        .respond_with(ResponseTemplate::new(403).set_body_json(json!({"errors": ["forbidden"]})))
        .expect(1)
        .mount(&server)
        .await;

    let result = manager(&server, &token_values)
        .async_write_secret("name", SecretValue::new("value"), None)
        .await;

    assert!(matches!(result, Err(Error::Status { status: 400 })));
}

#[rstest]
#[tokio::test]
async fn rotation_applies_timeout_to_each_request(token_values: Vec<(&str, &str)>) {
    let server = MockServer::start().await;
    let timeout = Duration::from_secs(1);
    let delay = timeout / 2;
    Mock::given(method("GET"))
        .and(header("X-Vault-Namespace", "team"))
        .and(path("/v1/alternate/data/managed/current"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(delay)
                .set_body_json(read_response(json!({"api_token": "original"}))),
        )
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(header("X-Vault-Namespace", "team"))
        .and(path("/v1/alternate/data/managed/new"))
        .and(body_json(json!({
            "data": {"api_token": "replacement", "description": "Rotated from current"}
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(delay)
                .set_body_json(json!({
                    "data": {
                        "created_time": "",
                        "deletion_time": "",
                        "custom_metadata": null,
                        "destroyed": false,
                        "version": 1
                    },
                    "lease_id": "",
                    "lease_duration": 0,
                    "renewable": false,
                    "request_id": "",
                    "warnings": null,
                    "wrap_info": null
                })),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(header("X-Vault-Namespace", "team"))
        .and(path("/v1/alternate/data/managed/new"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(delay)
                .set_body_json(read_response(json!({"api_token": "replacement"}))),
        )
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(header("X-Vault-Namespace", "team"))
        .and(path("/v1/alternate/data/managed/current"))
        .respond_with(ResponseTemplate::new(204).set_delay(delay))
        .mount(&server)
        .await;
    let manager = manager(&server, &token_values);
    let context = HashicorpOperationContext {
        namespace: Some("team".into()),
        timeout: Some(timeout),
        mount: Some("alternate".to_owned()),
        path_prefix: Some("managed".to_owned()),
        data_key: Some("api_token".to_owned()),
    };

    manager
        .async_rotate_secret_with_context(
            "current",
            "new",
            &SecretValue::new("replacement"),
            &context,
        )
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let operations: Vec<_> = requests
        .iter()
        .map(|request| (request.method.as_str(), request.url.path()))
        .collect();
    assert_eq!(
        operations,
        [
            ("GET", "/v1/alternate/data/managed/current"),
            ("POST", "/v1/alternate/data/managed/new"),
            ("GET", "/v1/alternate/data/managed/new"),
            ("DELETE", "/v1/alternate/data/managed/current"),
        ]
    );
}

#[rstest]
#[tokio::test]
#[ignore]
async fn live_vault_round_trip() {
    let environment: Arc<dyn Lookup + Send + Sync> =
        Arc::new(litellm_core_utils::settings::ProcessEnvironment);
    let manager: HashicorpVault = HashicorpVault::new(environment, true).unwrap();
    let name: String = std::env::var("LITELLM_VAULT_LIVE_SECRET_NAME").unwrap();
    let value: SecretValue = SecretValue::new("native-live-value");
    let location = manager.secret_location(&name).unwrap();
    println!(
        "native provenance: {} vaultrs {} {:?} {} {}",
        module_path!(),
        manager.config().address,
        location.namespace,
        location.mount,
        location.path
    );
    manager
        .async_write_secret(&name, value.clone(), None)
        .await
        .unwrap();
    assert_eq!(
        manager.async_read_secret(&name).await.unwrap().unwrap(),
        value
    );
    let replacement = SecretValue::new("replacement-π\n");
    manager
        .async_write_secret(&name, replacement.clone(), None)
        .await
        .unwrap();
    assert_eq!(
        manager.async_read_secret(&name).await.unwrap().unwrap(),
        replacement
    );
    manager.async_delete_secret(&name).await.unwrap();
    assert!(manager.async_read_secret(&name).await.unwrap().is_none());
}

#[rstest]
#[tokio::test]
async fn same_name_rotation_keeps_the_replacement(token_values: Vec<(&str, &str)>) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let server = MockServer::start().await;
    let reads = AtomicUsize::new(0);
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/name"))
        .respond_with(move |_: &wiremock::Request| {
            let value = if reads.fetch_add(1, Ordering::SeqCst) == 0 {
                "original"
            } else {
                "replacement"
            };
            ResponseTemplate::new(200).set_body_json(read_response(json!({"key": value})))
        })
        .expect(2)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/secret/data/name"))
        .and(body_json(json!({"data": {"key": "replacement", "description": "Rotated from name"}})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {"created_time": "", "deletion_time": "", "custom_metadata": null, "destroyed": false, "version": 2},
            "lease_id": "", "lease_duration": 0, "renewable": false, "request_id": "", "warnings": null, "wrap_info": null
        }))).expect(1).mount(&server).await;
    Mock::given(method("DELETE"))
        .respond_with(ResponseTemplate::new(204))
        .expect(0)
        .mount(&server)
        .await;
    let manager = manager(&server, &token_values);
    manager
        .async_rotate_secret("name", "name", &SecretValue::new("replacement"))
        .await
        .unwrap();
    assert_eq!(
        manager
            .async_read_secret("name")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "replacement"
    );
}

#[rstest]
#[case::verification(false)]
#[case::retirement(true)]
#[tokio::test]
async fn rotation_reports_partial_completion_without_losing_the_write_response(
    token_values: Vec<(&str, &str)>,
    #[case] verified: bool,
) {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/old"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(read_response(json!({"key": "old"}))),
        )
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/secret/data/new"))
        .respond_with(ResponseTemplate::new(200).set_body_json(write_response(2)))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/new"))
        .respond_with(ResponseTemplate::new(200).set_body_json(read_response(
            json!({"key": if verified { "replacement" } else { "stale" }}),
        )))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/v1/secret/data/old"))
        .respond_with(ResponseTemplate::new(403).set_body_json(json!({"errors": ["denied"]})))
        .expect(u64::from(verified))
        .mount(&server)
        .await;
    let result = manager(&server, &token_values)
        .async_rotate_secret("old", "new", &SecretValue::new("replacement"))
        .await;
    match result {
        Err(RotationError::Verification { response, source }) if !verified => {
            assert_eq!(response["version"], 2);
            assert!(matches!(
                source,
                Error::Operation(litellm_secrets_types::Error::NewSecretMismatch)
            ));
        }
        Err(RotationError::Retirement { response, source }) if verified => {
            assert_eq!(response["version"], 2);
            assert!(matches!(source, Error::Status { status: 403 }));
        }
        other => panic!("unexpected rotation outcome: {other:?}"),
    }
}

#[rstest]
#[case::different_namespace(
    " /team-b/ ",
    " /alternate/ ",
    " /prefix/ ",
    Some("team-b"),
    "/v1/alternate/data/prefix/key"
)]
#[case::clear_namespace("", "", "", None, "/v1/secret/data/key")]
#[tokio::test]
async fn operation_overrides_isolate_cached_targets_and_apply_to_writes_and_deletes(
    #[case] namespace: &str,
    #[case] mount: &str,
    #[case] prefix: &str,
    #[case] expected_namespace: Option<&str>,
    #[case] expected_path: &'static str,
) {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/configured/data/configured/key"))
        .and(header("X-Vault-Namespace", "team-a"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(read_response(json!({"key":"default-value"}))),
        )
        .expect(1)
        .mount(&server)
        .await;
    let namespace_header = expected_namespace.map(str::to_owned);
    Mock::given(path(expected_path))
        .respond_with(move |request: &wiremock::Request| {
            assert_eq!(
                request
                    .headers
                    .get("X-Vault-Namespace")
                    .map(|value| value.to_str().unwrap()),
                namespace_header.as_deref()
            );
            match request.method.as_str() {
                "GET" => ResponseTemplate::new(200)
                    .set_body_json(read_response(json!({"password":"override-value"}))),
                "POST" => {
                    assert_eq!(
                        request.body_json::<serde_json::Value>().unwrap(),
                        json!({"data":{"password":"written"}})
                    );
                    ResponseTemplate::new(200).set_body_json(write_response(2))
                }
                "DELETE" => ResponseTemplate::new(204),
                _ => panic!("unexpected method"),
            }
        })
        .expect(4)
        .mount(&server)
        .await;
    let manager = manager(
        &server,
        &[
            ("HCP_VAULT_TOKEN", "token"),
            ("HCP_VAULT_SECRET_NAMESPACE", "team-a"),
            ("HCP_VAULT_MOUNT_NAME", "configured"),
            ("HCP_VAULT_PATH_PREFIX", "configured"),
        ],
    );
    let context = HashicorpOperationContext {
        namespace: Some(namespace.into()),
        mount: Some(mount.into()),
        path_prefix: Some(prefix.into()),
        data_key: Some("password".into()),
        ..Default::default()
    };
    for _ in 0..2 {
        assert_eq!(
            manager
                .async_read_secret("key")
                .await
                .unwrap()
                .unwrap()
                .expose(),
            "default-value"
        );
        assert_eq!(
            BaseSecretManager::async_read_secret(&manager, "key", &context)
                .await
                .unwrap()
                .unwrap()
                .expose(),
            "override-value"
        );
    }
    SecretWriter::async_write_secret(
        &manager,
        "key",
        &SecretValue::new("written"),
        &SecretWriteContext {
            operation: context.clone(),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    SecretDeleter::async_delete_secret(&manager, "key", &context)
        .await
        .unwrap();
    assert_eq!(
        BaseSecretManager::async_read_secret(&manager, "key", &context)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "override-value"
    );
    assert_eq!(
        manager
            .async_read_secret("key")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "default-value"
    );
}
