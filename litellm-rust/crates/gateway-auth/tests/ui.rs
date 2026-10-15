use axum_login::{AuthUser, AuthnBackend};
use litellm_auth_types::SecretValue;
use litellm_gateway_auth::{UiBackend, UiCredentials};
use rstest::{fixture, rstest};

#[fixture]
fn backend() -> UiBackend {
    UiBackend::new("admin".into(), SecretValue::new("test-password")).unwrap()
}

#[rstest]
#[case::valid("admin", "test-password", true)]
#[case::wrong_password("admin", "wrong", false)]
#[case::wrong_username("other", "test-password", false)]
#[case::blank_password("admin", "", false)]
#[tokio::test]
async fn authenticates_both_credentials(
    backend: UiBackend,
    #[case] username: &str,
    #[case] password: &str,
    #[case] succeeds: bool,
) {
    let user = backend
        .authenticate(UiCredentials {
            username: username.into(),
            password: SecretValue::new(password),
        })
        .await
        .unwrap();
    assert_eq!(user.is_some(), succeeds);
}

#[rstest]
#[case::empty_username("", "password")]
#[case::empty_password("admin", "")]
#[case::whitespace_username("   ", "password")]
#[case::whitespace_password("admin", "   ")]
fn refuses_empty_configuration(#[case] username: &str, #[case] password: &str) {
    assert!(UiBackend::new(username.into(), SecretValue::new(password)).is_err());
}

#[rstest]
#[tokio::test]
async fn restores_only_configured_users(backend: UiBackend) {
    assert_eq!(
        backend
            .get_user(&"admin".into())
            .await
            .unwrap()
            .unwrap()
            .username,
        "admin"
    );
    assert!(backend.get_user(&"other".into()).await.unwrap().is_none());
}

#[rstest]
#[tokio::test]
async fn password_changes_invalidate_session_auth_hash(backend: UiBackend) {
    let original = backend.get_user(&"admin".into()).await.unwrap().unwrap();
    let replaced = UiBackend::new("admin".into(), SecretValue::new("new-password"))
        .unwrap()
        .get_user(&"admin".into())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(original.id(), replaced.id());
    assert_ne!(original.session_auth_hash(), replaced.session_auth_hash());
}
