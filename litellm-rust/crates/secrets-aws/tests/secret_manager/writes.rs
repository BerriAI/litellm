use super::*;

#[rstest]
#[tokio::test]
async fn same_name_rotation_uses_put_and_returns_its_response(
    default_settings: KeyManagementSettings,
) {
    let server = MockServer::start().await;
    Mock::given(header("x-amz-target", "secretsmanager.PutSecretValue"))
        .and(body_partial_json(
            json!({"SecretId":"key", "SecretString":"replacement"}),
        ))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"Name":"key", "VersionId":"version"})),
        )
        .expect(1)
        .mount(&server)
        .await;
    let response = manager(&server, default_settings)
        .async_rotate_secret("key", "key", &SecretValue::new("replacement"))
        .await
        .unwrap();
    match response {
        RotationResponse::Updated(output) => assert_eq!(output.version_id(), Some("version")),
        _ => panic!("rotation created a second secret"),
    }
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[rstest]
#[tokio::test]
async fn renamed_rotation_reads_creates_verifies_then_deletes(
    default_settings: KeyManagementSettings,
) {
    let server = MockServer::start().await;
    scripted_actions(
        &server,
        vec![
            Action {
                operation: "GetSecretValue",
                request: json!({"SecretId":"old"}),
                status: 200,
                response: json!({"SecretString":"old-value"}),
            },
            Action {
                operation: "CreateSecret",
                request: json!({"Name":"new", "Description":"Rotated from old", "SecretString":"replacement"}),
                status: 200,
                response: json!({"Name":"new"}),
            },
            Action {
                operation: "GetSecretValue",
                request: json!({"SecretId":"new"}),
                status: 200,
                response: json!({"SecretString":"replacement"}),
            },
            Action {
                operation: "DeleteSecret",
                request: json!({"SecretId":"old", "RecoveryWindowInDays":7}),
                status: 200,
                response: json!({"Name":"old"}),
            },
        ],
    ).await;
    assert!(matches!(
        manager(&server, default_settings)
            .async_rotate_secret("old", "new", &SecretValue::new("replacement"))
            .await
            .unwrap(),
        RotationResponse::Created(_)
    ));
}

#[rstest]
#[tokio::test]
async fn creation_passes_tags_and_kms_and_survives_replication_failure() {
    let server = MockServer::start().await;
    Mock::given(header("x-amz-target", "secretsmanager.CreateSecret"))
        .and(body_partial_json(json!({"Name":"key", "SecretString":"value", "KmsKeyId":"kms-key", "Tags":[{"Key":"stage", "Value":"test"}]})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"Name":"key"}))).expect(1).mount(&server).await;
    Mock::given(header(
        "x-amz-target",
        "secretsmanager.ReplicateSecretToRegions",
    ))
    .and(body_partial_json(
        json!({"SecretId":"key", "AddReplicaRegions":[{"Region":"replica-region"}]}),
    ))
    .respond_with(
        ResponseTemplate::new(400).set_body_json(json!({"__type":"InvalidRequestException"})),
    )
    .expect(1)
    .mount(&server)
    .await;
    let settings = KeyManagementSettings {
        kms_key_id: Some("kms-key".into()),
        tags: Some(std::collections::BTreeMap::from([(
            "stage".into(),
            "test".into(),
        )])),
        replica_regions: Some(vec!["replica-region".into()]),
        ..Default::default()
    };
    let manager = manager(&server, settings);
    assert_eq!(
        manager
            .async_write_secret("key", &SecretValue::new("value"), None)
            .await
            .unwrap()
            .name(),
        Some("key")
    );
    assert!(
        manager
            .async_replicate_secret("key", &[])
            .await
            .unwrap()
            .is_none()
    );
}

#[rstest]
#[tokio::test]
async fn trait_write_uses_typed_write_context(default_settings: KeyManagementSettings) {
    let server = MockServer::start().await;
    Mock::given(header("x-amz-target", "secretsmanager.CreateSecret"))
        .and(body_partial_json(json!({
            "Name": "key",
            "SecretString": "value",
            "Description": "created by caller",
            "Tags": [{"Key": "stage", "Value": "test"}],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"Name": "key"})))
        .expect(1)
        .mount(&server)
        .await;
    let context = SecretWriteContext {
        description: Some("created by caller".into()),
        tags: std::collections::BTreeMap::from([("stage".into(), "test".into())]),
        ..Default::default()
    };
    let response = SecretWriter::async_write_secret(
        &manager(&server, default_settings),
        "key",
        &SecretValue::new("value"),
        &context,
    )
    .await
    .unwrap();
    assert_eq!(response.name(), Some("key"));
}

#[rstest]
#[tokio::test]
async fn trait_delete_uses_the_provider_recovery_policy(default_settings: KeyManagementSettings) {
    let server = MockServer::start().await;
    Mock::given(header("x-amz-target", "secretsmanager.DeleteSecret"))
        .and(body_partial_json(json!({"SecretId": "key"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"Name": "key"})))
        .expect(1)
        .mount(&server)
        .await;
    let response = SecretDeleter::async_delete_secret(
        &manager(&server, default_settings),
        "key",
        &AwsOperationContext::default(),
    )
    .await
    .unwrap();
    assert_eq!(response.name(), Some("key"));
}

#[rstest]
#[case::write(false)]
#[case::rotate_back(true)]
#[tokio::test]
async fn recovery_window_alias_is_restored_updated_and_tagged(#[case] rotate: bool) {
    let server = MockServer::start().await;
    let description = if rotate {
        "Rotated from old"
    } else {
        "description"
    };
    let write = json!({"Name":"key", "SecretString":"new", "Description":description,
        "KmsKeyId":"kms", "Tags":[{"Key":"stage", "Value":"test"}]});
    let actions = if rotate {
        vec![Action {
            operation: "GetSecretValue",
            request: json!({"SecretId":"old"}),
            status: 200,
            response: json!({"SecretString":"old"}),
        }]
    } else {
        vec![]
    };
    let recovery = vec![
        Action {
            operation: "CreateSecret",
            request: write,
            status: 400,
            response: json!({"__type":"ResourceExistsException"}),
        },
        Action {
            operation: "DescribeSecret",
            request: json!({"SecretId":"key"}),
            status: 200,
            response: json!({"DeletedDate":1}),
        },
        Action {
            operation: "RestoreSecret",
            request: json!({"SecretId":"key"}),
            status: 200,
            response: json!({"Name":"key"}),
        },
        Action {
            operation: "UpdateSecret",
            request: json!({"SecretId":"key", "SecretString":"new", "Description":description,
            "KmsKeyId":"kms"}),
            status: 200,
            response: json!({"ARN":"restored-arn", "Name":"key", "VersionId":"new-version"}),
        },
        Action {
            operation: "TagResource",
            request: json!({"SecretId":"key", "Tags":[{"Key":"stage", "Value":"test"}]}),
            status: 200,
            response: json!({}),
        },
    ];
    let verification = if rotate {
        vec![
            Action {
                operation: "GetSecretValue",
                request: json!({"SecretId":"key"}),
                status: 200,
                response: json!({"SecretString":"new"}),
            },
            Action {
                operation: "DeleteSecret",
                request: json!({"SecretId":"old", "RecoveryWindowInDays":7}),
                status: 200,
                response: json!({}),
            },
        ]
    } else {
        vec![]
    };
    scripted_actions(
        &server,
        actions
            .into_iter()
            .chain(recovery)
            .chain(verification)
            .collect(),
    )
    .await;
    let manager = manager(
        &server,
        KeyManagementSettings {
            kms_key_id: Some("kms".into()),
            tags: Some(std::collections::BTreeMap::from([(
                "stage".into(),
                "test".into(),
            )])),
            ..Default::default()
        },
    );
    let output = if rotate {
        match manager
            .async_rotate_secret("old", "key", &SecretValue::new("new"))
            .await
            .unwrap()
        {
            RotationResponse::Created(output) => output,
            _ => panic!("expected restored alias"),
        }
    } else {
        manager
            .async_write_secret("key", &SecretValue::new("new"), Some(description))
            .await
            .unwrap()
    };
    assert_eq!(
        (output.arn(), output.name(), output.version_id()),
        (Some("restored-arn"), Some("key"), Some("new-version"))
    );
}

#[rstest]
#[case::live(200, json!({"Name":"key"}))]
#[case::missing(400, json!({"__type":"ResourceNotFoundException"}))]
#[case::denied(400, json!({"__type":"AccessDeniedException"}))]
#[tokio::test]
async fn create_failure_does_not_overwrite_an_alias_without_a_deletion_date(
    #[case] status: u16,
    #[case] described: serde_json::Value,
) {
    let server = MockServer::start().await;
    scripted_actions(
        &server,
        vec![
            Action {
                operation: "CreateSecret",
                request: json!({"Name":"key", "SecretString":"new"}),
                status: 400,
                response: json!({"__type":"ResourceExistsException"}),
            },
            Action {
                operation: "DescribeSecret",
                request: json!({"SecretId":"key"}),
                status,
                response: described,
            },
        ],
    )
    .await;
    assert!(matches!(
        manager(&server, Default::default())
            .async_write_secret("key", &SecretValue::new("new"), None)
            .await,
        Err(Error::Create(_))
    ));
}

#[derive(Clone, Copy, Debug)]
enum RecoveryFailure {
    Restore,
    Update,
    Tag,
    DeleteAfterUpdate,
    DeleteAfterTag,
}

#[rstest]
#[case::unconfigured(None)]
#[case::empty(Some(vec![]))]
#[case::configured(Some(vec!["region-a".into(), "region-b".into()]))]
#[tokio::test]
async fn creation_replicates_only_to_configured_regions(#[case] regions: Option<Vec<String>>) {
    let server = MockServer::start().await;
    let create = vec![Action {
        operation: "CreateSecret",
        request: json!({"Name":"key", "SecretString":"value", "KmsKeyId":"kms-key"}),
        status: 200,
        response: json!({"Name":"key", "VersionId":"created"}),
    }];
    let replicate = regions
        .as_ref()
        .filter(|regions| !regions.is_empty())
        .map(|regions| Action {
            operation: "ReplicateSecretToRegions",
            request: json!({"SecretId":"key", "AddReplicaRegions":regions.iter()
            .map(|region| json!({"Region":region})).collect::<Vec<_>>()}),
            status: 200,
            response: json!({"ARN":"replica-arn"}),
        });
    scripted_actions(&server, create.into_iter().chain(replicate).collect()).await;
    let environment: Arc<dyn litellm_core_utils::settings::Lookup + Send + Sync> = {
        let endpoint = server.uri();
        Arc::new(move |name: &str| match name {
            "AWS_BEDROCK_RUNTIME_ENDPOINT" => Some(endpoint.clone()),
            "AWS_ACCESS_KEY_ID" | "AWS_SECRET_ACCESS_KEY" => Some("test".into()),
            _ => None,
        })
    };
    let manager = AwsSecretsManagerV2::load_aws_secret_manager(
        Some(true),
        KeyManagementSettings {
            aws_region_name: Some("us-east-1".into()),
            replica_regions: regions,
            kms_key_id: Some("kms-key".into()),
            ..Default::default()
        },
        environment,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        manager
            .async_write_secret("key", &SecretValue::new("value"), None)
            .await
            .unwrap()
            .version_id(),
        Some("created")
    );
}

#[rstest]
#[case::success(200)]
#[case::denied(403)]
#[tokio::test]
async fn direct_replication_returns_response_or_service_error(#[case] status: u16) {
    let server = MockServer::start().await;
    scripted_actions(
        &server,
        vec![Action {
            operation: "ReplicateSecretToRegions",
            request: json!({"SecretId":"key", "AddReplicaRegions":[{"Region":"region-a"}, {"Region":"region-b"}]}),
            status,
            response: if status == 200 {
                json!({"ARN":"replicated-arn"})
            } else {
                json!({"__type":"AccessDeniedException"})
            },
        }],
    ).await;
    let result = manager(&server, Default::default())
        .async_replicate_secret("key", &["region-a".into(), "region-b".into()])
        .await;
    if status == 200 {
        assert_eq!(result.unwrap().unwrap().arn(), Some("replicated-arn"));
    } else {
        assert!(matches!(result, Err(Error::Replicate(_))));
    }
}

#[rstest]
#[case::create(false)]
#[case::replicate(true)]
#[tokio::test]
async fn write_and_replication_timeouts_remain_errors(#[case] replicate: bool) {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(1))
                .set_body_json(json!({})),
        )
        .expect(if replicate { 1 } else { 2 })
        .mount(&server)
        .await;
    let client = Client::from_conf(
        client_builder(&server)
            .timeout_config(
                aws_sdk_secretsmanager::config::timeout::TimeoutConfig::builder()
                    .operation_timeout(Duration::from_millis(50))
                    .build(),
            )
            .build(),
    );
    let manager = AwsSecretsManagerV2::new(client, Default::default());
    if replicate {
        assert!(matches!(
            manager
                .async_replicate_secret("key", &["region".into()])
                .await,
            Err(Error::Replicate(_))
        ));
    } else {
        assert!(matches!(
            manager
                .async_write_secret("key", &SecretValue::new("value"), None)
                .await,
            Err(Error::Create(_))
        ));
    }
}

#[rstest]
#[case::text("value")]
#[case::json(r#"{"api_key":"test","metadata":{"team":"test"},"temperature":0.7}"#)]
#[case::empty("")]
#[case::unicode(" π\n ")]
#[tokio::test]
async fn write_read_delete_preserves_the_complete_secret_string(#[case] value: &str) {
    let server = MockServer::start().await;
    scripted_actions(
        &server,
        vec![
            Action {
                operation: "CreateSecret",
                request: json!({"Name":"key", "SecretString":value, "Description":"description"}),
                status: 200,
                response: json!({"Name":"key"}),
            },
            Action {
                operation: "GetSecretValue",
                request: json!({"SecretId":"key"}),
                status: 200,
                response: json!({"SecretString":value}),
            },
            Action {
                operation: "DeleteSecret",
                request: json!({"SecretId":"key", "RecoveryWindowInDays":7}),
                status: 200,
                response: json!({"Name":"key"}),
            },
        ],
    )
    .await;
    let manager = manager(&server, Default::default());
    assert_eq!(
        manager
            .async_write_secret("key", &SecretValue::new(value), Some("description"))
            .await
            .unwrap()
            .name(),
        Some("key")
    );
    assert_eq!(
        manager
            .async_read_secret("key")
            .await
            .unwrap()
            .unwrap()
            .expose(),
        value
    );
    assert_eq!(
        manager
            .async_delete_secret("key", Some(7))
            .await
            .unwrap()
            .name(),
        Some("key")
    );
}

#[rstest]
#[case::restore(RecoveryFailure::Restore)]
#[case::update(RecoveryFailure::Update)]
#[case::tag(RecoveryFailure::Tag)]
#[case::delete_after_update(RecoveryFailure::DeleteAfterUpdate)]
#[case::delete_after_tag(RecoveryFailure::DeleteAfterTag)]
#[tokio::test]
async fn failed_update_reschedules_deletion_of_a_restored_alias(#[case] failure: RecoveryFailure) {
    let server = MockServer::start().await;
    let response = |failed| {
        if failed {
            (400, json!({"__type":"InvalidRequestException"}))
        } else {
            (200, json!({}))
        }
    };
    let (restore_status, restore_body) = response(matches!(failure, RecoveryFailure::Restore));
    let (update_status, update_body) = response(matches!(
        failure,
        RecoveryFailure::Update | RecoveryFailure::DeleteAfterUpdate
    ));
    let (delete_status, delete_body) = response(matches!(
        failure,
        RecoveryFailure::DeleteAfterUpdate | RecoveryFailure::DeleteAfterTag
    ));
    let prefix = [
        Action {
            operation: "CreateSecret",
            request: json!({"Name":"key", "SecretString":"new", "Tags":[{"Key":"stage", "Value":"test"}]}),
            status: 400,
            response: json!({"__type":"ResourceExistsException"}),
        },
        Action {
            operation: "DescribeSecret",
            request: json!({"SecretId":"key"}),
            status: 200,
            response: json!({"DeletedDate":1}),
        },
        Action {
            operation: "RestoreSecret",
            request: json!({"SecretId":"key"}),
            status: restore_status,
            response: restore_body,
        },
    ];
    let update = (!matches!(failure, RecoveryFailure::Restore)).then_some(Action {
        operation: "UpdateSecret",
        request: json!({"SecretId":"key", "SecretString":"new"}),
        status: update_status,
        response: update_body,
    });
    let tag = matches!(
        failure,
        RecoveryFailure::Tag | RecoveryFailure::DeleteAfterTag
    )
    .then_some(Action {
        operation: "TagResource",
        request: json!({"SecretId":"key", "Tags":[{"Key":"stage", "Value":"test"}]}),
        status: 400,
        response: json!({"__type":"InvalidRequestException"}),
    });
    let delete = (!matches!(failure, RecoveryFailure::Restore)).then_some(Action {
        operation: "DeleteSecret",
        request: json!({"SecretId":"key", "RecoveryWindowInDays":7}),
        status: delete_status,
        response: delete_body,
    });
    scripted_actions(
        &server,
        prefix
            .into_iter()
            .chain(update)
            .chain(tag)
            .chain(delete)
            .collect(),
    )
    .await;
    let error = manager(
        &server,
        KeyManagementSettings {
            tags: Some(std::collections::BTreeMap::from([(
                "stage".into(),
                "test".into(),
            )])),
            ..Default::default()
        },
    )
    .async_write_secret("key", &SecretValue::new("new"), None)
    .await
    .unwrap_err();
    match failure {
        RecoveryFailure::Restore => assert!(matches!(error, Error::Restore(_))),
        RecoveryFailure::Update => assert!(matches!(error, Error::Update(_))),
        RecoveryFailure::Tag => assert!(matches!(error, Error::Tag(_))),
        RecoveryFailure::DeleteAfterUpdate | RecoveryFailure::DeleteAfterTag => {
            assert!(matches!(error, Error::Delete(_)))
        }
    }
}
