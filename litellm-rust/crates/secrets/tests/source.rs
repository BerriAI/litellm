#[cfg(test)]
mod tests {
    use rstest::rstest;

    use litellm_secrets::source::{EnvironmentSecrets, SecretSource};

    #[rstest]
    #[case::lowercase_true("LITELLM_ENVIRONMENT_SECRETS_TRUE", "true", None)]
    #[case::padded_false("LITELLM_ENVIRONMENT_SECRETS_FALSE", " FALSE ", None)]
    #[case::text("LITELLM_ENVIRONMENT_SECRETS_TEXT", "secret", Some("secret"))]
    #[tokio::test]
    async fn python_environment_values_are_absent_like_get_secret_str(
        #[case] name: &'static str,
        #[case] value: &str,
        #[case] expected: Option<&str>,
    ) {
        unsafe { std::env::set_var(name, value) };
        let secret = EnvironmentSecrets::python_compatible()
            .resolve(&[name])
            .await
            .unwrap()
            .get(name);
        unsafe { std::env::remove_var(name) };
        assert_eq!(secret.as_deref(), expected);
    }
}

#[tokio::test]
async fn dynamic_names_use_the_same_resolver_and_snapshots_never_do_fresh_lookups() {
    use litellm_secrets::source::SecretSource;
    use litellm_secrets::{OidcResolver, SecretManagerState, SecretResolver};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    let calls = Arc::new(AtomicUsize::new(0));
    let reads = calls.clone();
    let source = SecretResolver::new(
        Arc::new(SecretManagerState::default()),
        Arc::new(move |name: &str| {
            reads.fetch_add(1, Ordering::SeqCst);
            (name != "missing").then(|| name.to_owned())
        }),
        OidcResolver::default(),
    );
    let snapshot = source.resolve(&["declared", "missing"]).await.unwrap();
    let name = format!("runtime-{}", "key");
    assert_eq!(snapshot.get("declared").as_deref(), Some("declared"));
    assert_eq!(snapshot.get("missing"), None);
    assert_eq!(snapshot.get(&name), None);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        SecretSource::get_secret_str(&source, &name)
            .await
            .unwrap()
            .unwrap()
            .expose(),
        name
    );
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}
