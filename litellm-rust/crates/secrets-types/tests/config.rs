use litellm_secrets_types::{
    AccessMode, KeyManagementSettings, KeyManagementSystem, Secret, SecretValue,
};
use serde_json::json;

#[test]
fn config_preserves_defaults_nulls_and_serialized_names() {
    let empty: KeyManagementSettings = serde_json::from_value(json!({})).unwrap();
    assert_eq!(empty, KeyManagementSettings::default());
    assert_eq!(empty.access_mode, AccessMode::ReadOnly);
    assert_eq!(empty.store_virtual_keys, Some(false));
    assert_eq!(empty.prefix_for_stored_virtual_keys, "litellm/");
    let configured: KeyManagementSettings = serde_json::from_value(json!({
        "hosted_keys": [], "store_virtual_keys": null, "access_mode": "write_only",
        "aws_web_identity_token": "private-token", "aws_external_id": "private-id",
        "tags": {"stage": "test"}, "replica_regions": ["test-region"]
    }))
    .unwrap();
    assert!(!configured.access_mode.readable());
    assert_eq!(configured.store_virtual_keys, None);
    assert_eq!(configured.hosted_keys.as_deref(), Some([].as_slice()));
    assert!(!format!("{configured:?}").contains("private-"));
    let serialized = serde_json::to_value(&configured).unwrap();
    assert_eq!(serialized["access_mode"], "write_only");
    assert_eq!(serialized["aws_web_identity_token"], "private-token");
    assert_eq!(
        serde_json::from_value::<KeyManagementSettings>(serialized).unwrap(),
        configured
    );
}

#[rstest::rstest]
#[case::aws_kms("aws_kms", KeyManagementSystem::AwsKms)]
#[case::aws_secret_manager("aws_secret_manager", KeyManagementSystem::AwsSecretManager)]
#[case::google_kms("google_kms", KeyManagementSystem::GoogleKms)]
#[case::google_secret_manager("google_secret_manager", KeyManagementSystem::GoogleSecretManager)]
#[case::azure_key_vault("azure_key_vault", KeyManagementSystem::AzureKeyVault)]
#[case::hashicorp_vault("hashicorp_vault", KeyManagementSystem::HashicorpVault)]
#[case::cyberark("cyberark", KeyManagementSystem::Cyberark)]
#[case::custom("custom", KeyManagementSystem::Custom)]
#[case::local("local", KeyManagementSystem::Local)]
fn key_management_system_serialization_round_trips(
    #[case] name: &str,
    #[case] system: KeyManagementSystem,
) {
    assert_eq!(
        serde_json::from_value::<KeyManagementSystem>(json!(name)).unwrap(),
        system
    );
    assert_eq!(serde_json::to_value(system).unwrap(), name);
}

#[test]
fn secret_debug_never_exposes_values() {
    assert!(
        !format!("{:?}", Secret::String(SecretValue::new("sensitive-value")))
            .contains("sensitive-value")
    );
    assert!(!format!("{:?}", Secret::Bool(true)).contains("true"));
}
