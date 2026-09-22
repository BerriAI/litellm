use std::sync::atomic::{AtomicUsize, Ordering};

use litellm_secrets_types::{
    BaseSecretManager, Error, HashicorpOperationContext, SecretOperationContext, SecretValue,
    SecretWriteContext, async_rotate_secret, validate_secret_name,
};
use rstest::{fixture, rstest};

struct Manager {
    step: AtomicUsize,
    absent_at: Option<usize>,
    operation: SecretOperationContext,
}

impl BaseSecretManager for Manager {
    type Error = Error;
    type WriteResponse = &'static str;
    type DeleteResponse = ();

    async fn async_read_secret(
        &self,
        name: &str,
        context: &SecretOperationContext,
    ) -> Result<Option<SecretValue>, Error> {
        assert_eq!(context, &self.operation);
        let step = self.step.fetch_add(1, Ordering::SeqCst);
        assert_eq!(name, if step == 0 { "old" } else { "new" });
        Ok((self.absent_at != Some(step)).then(|| SecretValue::new("value")))
    }

    async fn async_write_secret(
        &self,
        name: &str,
        value: &SecretValue,
        context: &SecretWriteContext,
    ) -> Result<Self::WriteResponse, Error> {
        assert_eq!(self.step.fetch_add(1, Ordering::SeqCst), 1);
        assert_eq!(name, "new");
        assert_eq!(value.expose(), "replacement");
        assert_eq!(context.description.as_deref(), Some("Rotated from old"));
        assert!(context.tags.is_empty());
        assert_eq!(context.operation, self.operation);
        Ok("provider-response")
    }

    async fn async_delete_secret(
        &self,
        name: &str,
        recovery_window_in_days: Option<u32>,
        context: &SecretOperationContext,
    ) -> Result<(), Error> {
        assert_eq!(self.step.fetch_add(1, Ordering::SeqCst), 3);
        assert_eq!(name, "old");
        assert_eq!(recovery_window_in_days, Some(7));
        assert_eq!(context, &self.operation);
        Ok(())
    }
}

#[fixture]
fn replacement() -> SecretValue {
    SecretValue::new("replacement")
}

#[rstest]
#[case::default(SecretOperationContext::Default)]
#[case::backend_specific(SecretOperationContext::Hashicorp(HashicorpOperationContext {
    mount: Some("alternate".to_owned()),
    ..HashicorpOperationContext::default()
}))]
#[tokio::test]
async fn rotation_verifies_before_deleting_and_returns_provider_response(
    replacement: SecretValue,
    #[case] operation: SecretOperationContext,
) {
    let manager = Manager {
        step: AtomicUsize::new(0),
        absent_at: None,
        operation: operation.clone(),
    };
    assert_eq!(
        async_rotate_secret(&manager, "old", "new", &replacement, &operation,)
            .await
            .unwrap(),
        "provider-response"
    );
    assert_eq!(manager.step.load(Ordering::SeqCst), 4);
}

#[rstest]
#[case::current_secret_missing(0, Error::CurrentSecretMissing, 1)]
#[case::new_secret_missing(2, Error::NewSecretMissing, 3)]
#[tokio::test]
async fn missing_old_or_new_value_stops_rotation_before_deletion(
    replacement: SecretValue,
    #[case] absent_at: usize,
    #[case] expected: Error,
    #[case] calls: usize,
) {
    let manager = Manager {
        step: AtomicUsize::new(0),
        absent_at: Some(absent_at),
        operation: SecretOperationContext::default(),
    };
    assert_eq!(
        async_rotate_secret(
            &manager,
            "old",
            "new",
            &replacement,
            &SecretOperationContext::default(),
        )
        .await
        .unwrap_err(),
        expected
    );
    assert_eq!(manager.step.load(Ordering::SeqCst), calls);
}

#[rstest]
#[case::parent("..")]
#[case::parent_prefix("../../../other-app/creds")]
#[case::nested_parent_prefix("litellm/../../secret")]
#[case::parent_segment("foo/../bar")]
#[case::parent_suffix("foo/..")]
#[case::single_parent_prefix("../foo")]
#[case::line_feed("foo\nbar")]
#[case::carriage_return("foo\rbar")]
#[case::tab("foo\tbar")]
#[case::null("foo\0bar")]
#[case::delete("foo\u{7f}bar")]
#[case::next_line("foo\u{85}bar")]
#[case::line_separator("foo\u{2028}bar")]
#[case::paragraph_separator("foo\u{2029}bar")]
fn names_reject_path_traversal_and_control_characters(#[case] name: &str) {
    assert_eq!(validate_secret_name(name), Err(Error::UnsafeSecretName));
}

#[rstest]
#[case::plain_alias("plain-alias")]
#[case::key_with_digits("my-key-123")]
#[case::service_path("prod/my-service-key")]
#[case::email_path("team/user@example.com")]
#[case::colon("foo: bar")]
#[case::spaced_fragment("foo # bar")]
#[case::query("foo?evil=1")]
#[case::fragment("foo#bar")]
#[case::long_alias(&"a".repeat(500))]
#[case::embedded_double_dot("release-1.0..2")]
#[case::middle_double_dot("my..key")]
#[case::leading_double_dot("..foo")]
#[case::trailing_double_dot("foo..")]
#[case::version_double_dot("v2.0..1-beta")]
#[case::empty("")]
#[case::three_dots("...")]
fn names_allow_safe_values(#[case] name: &str) {
    assert_eq!(validate_secret_name(name), Ok(()));
}
