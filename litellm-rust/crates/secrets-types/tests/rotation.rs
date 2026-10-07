use std::sync::atomic::{AtomicUsize, Ordering};

use litellm_secrets_types::{
    BaseSecretManager, Error, SecretValue, async_rotate_secret, validate_secret_name,
};

struct Manager {
    step: AtomicUsize,
    absent_at: Option<usize>,
}

impl BaseSecretManager for Manager {
    type Error = Error;
    type WriteResponse = &'static str;
    type DeleteResponse = ();

    async fn async_read_secret(&self, name: &str) -> Result<Option<SecretValue>, Error> {
        let step = self.step.fetch_add(1, Ordering::SeqCst);
        assert_eq!(name, if step == 0 { "old" } else { "new" });
        Ok((self.absent_at != Some(step)).then(|| SecretValue::new("value")))
    }

    async fn async_write_secret(
        &self,
        name: &str,
        value: &SecretValue,
        description: Option<&str>,
    ) -> Result<Self::WriteResponse, Error> {
        assert_eq!(self.step.fetch_add(1, Ordering::SeqCst), 1);
        assert_eq!(name, "new");
        assert_eq!(value.expose(), "replacement");
        assert_eq!(description, Some("Rotated from old"));
        Ok("provider-response")
    }

    async fn async_delete_secret(
        &self,
        name: &str,
        recovery_window_in_days: i64,
    ) -> Result<(), Error> {
        assert_eq!(self.step.fetch_add(1, Ordering::SeqCst), 3);
        assert_eq!(name, "old");
        assert_eq!(recovery_window_in_days, 7);
        Ok(())
    }
}

#[tokio::test]
async fn rotation_verifies_before_deleting_and_returns_provider_response() {
    let manager = Manager {
        step: AtomicUsize::new(0),
        absent_at: None,
    };
    assert_eq!(
        async_rotate_secret(&manager, "old", "new", &SecretValue::new("replacement"))
            .await
            .unwrap(),
        "provider-response"
    );
    assert_eq!(manager.step.load(Ordering::SeqCst), 4);
}

#[rstest::rstest]
#[case::current_secret_missing(0, Error::CurrentSecretMissing, 1)]
#[case::new_secret_missing(2, Error::NewSecretMissing, 3)]
#[tokio::test]
async fn missing_old_or_new_value_stops_rotation_before_deletion(
    #[case] absent_at: usize,
    #[case] expected: Error,
    #[case] calls: usize,
) {
    let manager = Manager {
        step: AtomicUsize::new(0),
        absent_at: Some(absent_at),
    };
    assert_eq!(
        async_rotate_secret(&manager, "old", "new", &SecretValue::new("replacement"))
            .await
            .unwrap_err(),
        expected
    );
    assert_eq!(manager.step.load(Ordering::SeqCst), calls);
}

#[rstest::rstest]
#[case::parent("..")]
#[case::parent_prefix("../x")]
#[case::parent_segment("x/../y")]
#[case::parent_suffix("x/..")]
#[case::line_feed("line\n")]
#[case::next_line("\u{85}")]
#[case::line_separator("\u{2028}")]
#[case::paragraph_separator("\u{2029}")]
fn names_reject_path_traversal_and_control_characters(#[case] name: &str) {
    assert_eq!(validate_secret_name(name), Err(Error::UnsafeSecretName));
}

#[rstest::rstest]
#[case::embedded_double_dot("release-1.0..2")]
#[case::path_separator("folder/key")]
#[case::empty("")]
#[case::three_dots("...")]
fn names_allow_safe_values(#[case] name: &str) {
    assert_eq!(validate_secret_name(name), Ok(()));
}
