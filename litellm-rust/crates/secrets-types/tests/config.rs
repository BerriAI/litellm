use litellm_secrets_types::{
    AccessMode, KeyManagementSettings, KeyManagementSystem, Secret, SecretValue,
};
use rstest::{fixture, rstest};
use serde_json::json;

#[fixture]
fn default_settings() -> KeyManagementSettings {
    serde_json::from_value(json!({})).unwrap()
}

#[rstest]
fn config_preserves_defaults_nulls_and_serialized_names(default_settings: KeyManagementSettings) {
    assert_eq!(default_settings, KeyManagementSettings::default());
    assert_eq!(default_settings.access_mode, AccessMode::ReadOnly);
    assert_eq!(default_settings.store_virtual_keys, Some(false));
    assert_eq!(default_settings.prefix_for_stored_virtual_keys, "litellm/");
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

#[rstest]
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

#[rstest]
#[case::read_only(AccessMode::ReadOnly, true)]
#[case::write_only(AccessMode::WriteOnly, false)]
#[case::read_and_write(AccessMode::ReadAndWrite, true)]
fn access_mode_reports_readability(#[case] mode: AccessMode, #[case] expected: bool) {
    assert_eq!(mode.readable(), expected);
}

#[rstest]
#[case::string(json!("value"), Secret::String(SecretValue::new("value")))]
#[case::boolean(json!(true), Secret::Bool(true))]
#[case::number(json!(7), Secret::Json(json!(7)))]
#[case::null(json!(null), Secret::Json(json!(null)))]
#[case::array(json!(["value"]), Secret::Json(json!(["value"])))]
#[case::object(json!({"key": "value"}), Secret::Json(json!({"key": "value"})))]
fn secret_conversion_preserves_json_types(
    #[case] value: serde_json::Value,
    #[case] expected: Secret,
) {
    assert_eq!(Secret::from_json(value), expected);
}

#[rstest]
#[case::string(Secret::String(SecretValue::new("sensitive-value")), "sensitive-value")]
#[case::boolean(Secret::Bool(true), "true")]
#[case::json(Secret::Json(json!({"private": "value"})), "private")]
fn secret_debug_never_exposes_values(#[case] secret: Secret, #[case] sensitive: &str) {
    assert!(!format!("{secret:?}").contains(sensitive));
}
