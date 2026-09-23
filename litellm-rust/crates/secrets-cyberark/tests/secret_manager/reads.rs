use super::*;

#[rstest]
#[tokio::test]
async fn a_rejected_refreshed_token_surfaces_the_error_without_another_retry() {
    let server = MockServer::start().await;
    mount_auth(&server, 2).await;
    Mock::given(path("/secrets/acct/variable/first"))
        .respond_with(ResponseTemplate::new(200).set_body_string("first-value"))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(path("/secrets/acct/variable/second"))
        .respond_with(ResponseTemplate::new(401))
        .expect(2)
        .mount(&server)
        .await;
    let manager = manager(&server, Duration::from_secs(60));
    assert_eq!(
        manager
            .async_read_secret("first")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "first-value"
    );

    let result = tokio::time::timeout(Duration::from_secs(5), manager.async_read_secret("second"))
        .await
        .expect("authentication retries must terminate");

    assert!(matches!(result, Err(Error::Status(401))));
}

#[rstest]
#[case::not_found(404)]
#[case::unauthorized(401)]
#[case::forbidden(403)]
#[case::server_error(500)]
#[tokio::test]
async fn failed_reads_are_not_cached(#[case] status: u16) {
    let server = MockServer::start().await;
    mount_auth(&server, 1).await;
    let failing = Mock::given(path("/secrets/acct/variable/key"))
        .respond_with(ResponseTemplate::new(status))
        .expect(1)
        .mount_as_scoped(&server)
        .await;
    let manager = manager(&server, Duration::from_secs(60));
    let result = manager.async_read_secret("key").await;
    if status == 404 {
        assert_eq!(result.unwrap(), None);
    } else {
        assert!(matches!(result, Err(Error::Status(actual)) if actual == status));
    }
    drop(failing);
    Mock::given(path("/secrets/acct/variable/key"))
        .respond_with(ResponseTemplate::new(200).set_body_string("recovered"))
        .expect(1)
        .mount(&server)
        .await;
    for _ in 0..2 {
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
}

#[rstest]
#[tokio::test]
async fn expired_tokens_and_secrets_are_fetched_again() {
    let server = MockServer::start().await;
    mount_auth(&server, 2).await;
    Mock::given(path("/secrets/acct/variable/key"))
        .respond_with(ResponseTemplate::new(200).set_body_string("value"))
        .expect(2)
        .mount(&server)
        .await;
    let manager = manager(&server, Duration::from_millis(1));
    for _ in 0..2 {
        assert!(manager.async_read_secret("key").await.unwrap().is_some());
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

#[rstest]
#[case::plain("OPENAI_API_KEY")]
#[case::path("team/app/key")]
#[case::punctuation("a b+c.d-e_f~g")]
#[case::quote("needs \"quote\"")]
#[tokio::test]
async fn secret_names_use_python_quote_encoding(parity_fixture: ParityFixture, #[case] name: &str) {
    let secret = parity_fixture
        .secrets
        .iter()
        .find(|secret| secret.name == name)
        .unwrap();
    let server = MockServer::start().await;
    mount_auth(&server, 1).await;
    Mock::given(RawPath(secret.path.clone()))
        .respond_with(ResponseTemplate::new(200).set_body_string("value"))
        .expect(1)
        .mount(&server)
        .await;
    assert_eq!(
        manager(&server, Duration::from_secs(60))
            .async_read_secret(name)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        "value"
    );
}

#[rstest]
#[case::parent("../etc")]
#[case::embedded_parent("team/../etc")]
#[case::control("key\n")]
#[tokio::test]
async fn unsafe_names_fail_before_http_calls(#[case] name: &str) {
    let server = MockServer::start().await;
    let manager = manager(&server, Duration::from_secs(60));
    assert!(matches!(
        manager.async_read_secret(name).await,
        Err(Error::Operation(
            litellm_secrets_types::Error::UnsafeSecretName
        ))
    ));
    assert!(matches!(
        manager
            .async_write_secret(name, &SecretValue::new("v"), None)
            .await,
        Err(Error::Operation(
            litellm_secrets_types::Error::UnsafeSecretName
        ))
    ));
}
