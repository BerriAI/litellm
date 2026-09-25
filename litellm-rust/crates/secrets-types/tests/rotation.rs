use std::sync::atomic::{AtomicUsize, Ordering};

use litellm_secrets_types::{
    BaseSecretManager, Error, HashicorpOperationContext, RotationError, SecretDeleter,
    SecretOperationContext, SecretRotator, SecretValue, SecretWriteContext, SecretWriter,
    async_rotate_secret, validate_secret_name,
};
use rstest::{fixture, rstest};

struct Manager {
    step: AtomicUsize,
    absent_at: Option<usize>,
    verified_value: &'static str,
    operation: SecretOperationContext,
    delete_error: bool,
    fail_at: Option<usize>,
}

impl BaseSecretManager for Manager {
    type Error = Error;
    type Context = SecretOperationContext;

    async fn async_read_secret(
        &self,
        _name: &str,
        _context: &Self::Context,
    ) -> Result<Option<SecretValue>, Error> {
        panic!("rotation must bypass cached reads")
    }
}

impl SecretRotator for Manager {
    type RotationResponse = &'static str;

    async fn async_write_replacement(
        &self,
        current_name: &str,
        new_name: &str,
        value: &SecretValue,
        context: &Self::Context,
    ) -> Result<Self::RotationResponse, Error> {
        self.async_write_secret(
            new_name,
            value,
            &SecretWriteContext::rotated_from(current_name, context.clone()),
        )
        .await
    }

    async fn async_read_secret_fresh(
        &self,
        name: &str,
        context: &SecretOperationContext,
    ) -> Result<Option<SecretValue>, Error> {
        assert_eq!(context, &self.operation);
        let step = self.step.fetch_add(1, Ordering::SeqCst);
        assert_eq!(name, if step == 0 { "old" } else { "new" });
        if self.fail_at == Some(step) {
            return Err(Error::UnsafeSecretName);
        }
        Ok((self.absent_at != Some(step)).then(|| {
            SecretValue::new(if step == 0 {
                "value"
            } else {
                self.verified_value
            })
        }))
    }
}

impl SecretWriter for Manager {
    type WriteResponse = &'static str;

    async fn async_write_secret(
        &self,
        name: &str,
        value: &SecretValue,
        context: &SecretWriteContext,
    ) -> Result<Self::WriteResponse, Error> {
        assert_eq!(self.step.fetch_add(1, Ordering::SeqCst), 1);
        if self.fail_at == Some(1) {
            return Err(Error::UnsafeSecretName);
        }
        assert_eq!(name, "new");
        assert_eq!(value.expose(), "replacement");
        assert_eq!(context.description.as_deref(), Some("Rotated from old"));
        assert!(context.tags.is_empty());
        assert_eq!(context.operation, self.operation);
        Ok("provider-response")
    }
}

impl SecretDeleter for Manager {
    type DeleteResponse = ();

    async fn async_delete_secret(
        &self,
        name: &str,
        context: &SecretOperationContext,
    ) -> Result<(), Error> {
        assert_eq!(self.step.fetch_add(1, Ordering::SeqCst), 3);
        assert_eq!(name, "old");
        assert_eq!(context, &self.operation);
        if self.delete_error {
            Err(Error::UnsafeSecretName)
        } else {
            Ok(())
        }
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
        delete_error: false,
        fail_at: None,
        absent_at: None,
        verified_value: "replacement",
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
#[case::current_secret_missing(0, RotationError::Read(Error::CurrentSecretMissing), 1)]
#[case::new_secret_missing(2, RotationError::Verification { response: Box::new("provider-response"), source: Error::NewSecretMissing }, 3)]
#[tokio::test]
async fn missing_old_or_new_value_stops_rotation_before_deletion(
    replacement: SecretValue,
    #[case] absent_at: usize,
    #[case] expected: RotationError<&'static str, Error>,
    #[case] calls: usize,
) {
    let manager = Manager {
        step: AtomicUsize::new(0),
        delete_error: false,
        fail_at: None,
        absent_at: Some(absent_at),
        verified_value: "replacement",
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

#[tokio::test]
async fn a_different_replacement_never_deletes_the_current_secret() {
    let manager = Manager {
        step: AtomicUsize::new(0),
        delete_error: false,
        fail_at: None,
        absent_at: None,
        verified_value: "stale-value",
        operation: SecretOperationContext::Default,
    };
    assert_eq!(
        async_rotate_secret(
            &manager,
            "old",
            "new",
            &SecretValue::new("replacement"),
            &SecretOperationContext::Default
        )
        .await,
        Err(RotationError::Verification {
            response: Box::new("provider-response"),
            source: Error::NewSecretMismatch
        })
    );
    assert_eq!(manager.step.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn failed_retirement_preserves_the_verified_replacement_response() {
    let manager = Manager {
        step: AtomicUsize::new(0),
        absent_at: None,
        verified_value: "replacement",
        operation: SecretOperationContext::Default,
        delete_error: true,
        fail_at: None,
    };
    assert_eq!(
        async_rotate_secret(
            &manager,
            "old",
            "new",
            &SecretValue::new("replacement"),
            &SecretOperationContext::Default
        )
        .await,
        Err(RotationError::Retirement {
            response: Box::new("provider-response"),
            source: Error::UnsafeSecretName
        })
    );
}

#[rstest]
#[case::read(0, RotationError::Read(Error::UnsafeSecretName))]
#[case::write(1, RotationError::Write(Error::UnsafeSecretName))]
#[case::verification(2, RotationError::Verification { response: Box::new("provider-response"), source: Error::UnsafeSecretName })]
#[tokio::test]
async fn provider_failures_stop_rotation_before_retirement(
    #[case] fail_at: usize,
    #[case] expected: RotationError<&'static str, Error>,
) {
    let manager = Manager {
        step: AtomicUsize::new(0),
        absent_at: None,
        verified_value: "replacement",
        operation: SecretOperationContext::default(),
        delete_error: false,
        fail_at: Some(fail_at),
    };
    assert_eq!(
        async_rotate_secret(
            &manager,
            "old",
            "new",
            &SecretValue::new("replacement"),
            &SecretOperationContext::default()
        )
        .await
        .unwrap_err(),
        expected
    );
    assert_eq!(manager.step.load(Ordering::SeqCst), fail_at + 1);
}
