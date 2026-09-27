use std::{collections::BTreeMap, time::Duration};

use litellm_secrets_types::{
    AwsOperationContext, AzureOperationContext, CyberarkOperationContext, GoogleOperationContext,
    HashicorpOperationContext, KeyManagementSystem, SecretOperationContext, SecretValue,
    SecretWriteContext,
};
use rstest::{fixture, rstest};

#[fixture]
fn timeout() -> Duration {
    Duration::from_secs(30)
}

#[fixture]
fn aws_context(timeout: Duration) -> SecretOperationContext {
    SecretOperationContext::Aws(AwsOperationContext {
        timeout: Some(timeout),
        region_name: Some("us-west-2".into()),
        role_name: Some("role".into()),
        external_id: Some(SecretValue::new("external-id")),
        web_identity_token: Some(SecretValue::new("web-identity-token")),
        ..AwsOperationContext::default()
    })
}

#[rstest]
fn operation_context_preserves_backend_specific_values_and_redacts_secrets(
    aws_context: SecretOperationContext,
    timeout: Duration,
) {
    assert_eq!(aws_context.timeout(), Some(timeout));
    assert_eq!(
        aws_context,
        SecretOperationContext::Aws(AwsOperationContext {
            timeout: Some(timeout),
            region_name: Some("us-west-2".into()),
            role_name: Some("role".into()),
            external_id: Some(SecretValue::new("external-id")),
            web_identity_token: Some(SecretValue::new("web-identity-token")),
            ..AwsOperationContext::default()
        })
    );
    let debug = format!("{aws_context:?}");
    assert!(!debug.contains("external-id"));
    assert!(!debug.contains("web-identity-token"));
}

#[rstest]
#[case::default(SecretOperationContext::Default, None)]
#[case::aws(
    SecretOperationContext::Aws(AwsOperationContext {
        timeout: Some(Duration::from_secs(1)),
        ..AwsOperationContext::default()
    }),
    Some(Duration::from_secs(1))
)]
#[case::hashicorp(
    SecretOperationContext::Hashicorp(HashicorpOperationContext {
        timeout: Some(Duration::from_secs(2)),
        ..HashicorpOperationContext::default()
    }),
    Some(Duration::from_secs(2))
)]
#[case::cyberark(
    SecretOperationContext::Cyberark(CyberarkOperationContext {
        timeout: Some(Duration::from_secs(3)),
    }),
    Some(Duration::from_secs(3))
)]
fn operation_context_reports_each_backend_timeout(
    #[case] context: SecretOperationContext,
    #[case] expected: Option<Duration>,
) {
    assert_eq!(context.timeout(), expected);
}

#[rstest]
fn write_context_keeps_tags_separate_from_the_operation_context() {
    let context = SecretWriteContext {
        description: Some("Managed virtual key".into()),
        tags: BTreeMap::from([("team".into(), "team-id".into())]),
        operation: SecretOperationContext::Hashicorp(HashicorpOperationContext {
            mount: Some("secret".into()),
            path_prefix: Some("teams".into()),
            data_key: Some("api_key".into()),
            ..HashicorpOperationContext::default()
        }),
    };

    assert_eq!(context.tags.get("team"), Some(&"team-id".into()));
    assert_eq!(context.operation.timeout(), None);
    assert_eq!(
        context.operation,
        SecretOperationContext::Hashicorp(HashicorpOperationContext {
            mount: Some("secret".into()),
            path_prefix: Some("teams".into()),
            data_key: Some("api_key".into()),
            ..HashicorpOperationContext::default()
        })
    );
}

#[rstest]
fn rotation_write_context_preserves_the_operation_context(aws_context: SecretOperationContext) {
    let context = SecretWriteContext::rotated_from("current", aws_context.clone());

    assert_eq!(context.description.as_deref(), Some("Rotated from current"));
    assert!(context.tags.is_empty());
    assert_eq!(context.operation, aws_context);
}

#[rstest]
#[case(
    KeyManagementSystem::AwsSecretManager,
    SecretOperationContext::Aws(Default::default())
)]
#[case(KeyManagementSystem::AzureKeyVault, SecretOperationContext::Azure(AzureOperationContext { timeout: Some(Duration::from_secs(1)) }))]
#[case(KeyManagementSystem::GoogleSecretManager, SecretOperationContext::Google(GoogleOperationContext { timeout: Some(Duration::from_secs(1)) }))]
#[case(
    KeyManagementSystem::HashicorpVault,
    SecretOperationContext::Hashicorp(Default::default())
)]
#[case(
    KeyManagementSystem::Cyberark,
    SecretOperationContext::Cyberark(Default::default())
)]
fn provider_context_accepts_only_its_owner(
    #[case] owner: KeyManagementSystem,
    #[case] context: SecretOperationContext,
) {
    for system in [
        KeyManagementSystem::AwsSecretManager,
        KeyManagementSystem::AzureKeyVault,
        KeyManagementSystem::GoogleSecretManager,
        KeyManagementSystem::HashicorpVault,
        KeyManagementSystem::Cyberark,
    ] {
        assert_eq!(context.validate_for(system).is_ok(), system == owner);
        assert!(SecretOperationContext::Default.validate_for(system).is_ok());
    }
    if matches!(
        owner,
        KeyManagementSystem::AzureKeyVault | KeyManagementSystem::GoogleSecretManager
    ) {
        assert_eq!(context.timeout(), Some(Duration::from_secs(1)));
    }
}
