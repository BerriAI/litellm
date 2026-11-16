use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    sync::Arc,
};

use futures_util::future::{BoxFuture, FutureExt, try_join_all};
use litellm_env::Lookup;
use litellm_python_compat::flags::str_to_bool;
use litellm_secret_value::SecretValue;
use litellm_secrets::{
    Error,
    source::{SecretSource, Secrets},
};

const ENVIRONMENT_PREFIX: &str = "os.environ/";
const OIDC_PREFIX: &str = "oidc/";

/// Python's `get_secret_str` while no secret manager is readable: names come from the process
/// environment, read in Rust when the route looks them up, and `oidc/` references go to `python`.
pub(super) struct LocalSecrets {
    python: Arc<dyn SecretSource>,
}

impl LocalSecrets {
    pub(super) fn new(python: Arc<dyn SecretSource>) -> Self {
        Self { python }
    }
}

fn python_name(name: &str) -> Cow<'_, str> {
    if name.starts_with(ENVIRONMENT_PREFIX) {
        Cow::Owned(name.replace(ENVIRONMENT_PREFIX, ""))
    } else {
        Cow::Borrowed(name)
    }
}

fn is_oidc(name: &str) -> bool {
    python_name(name).starts_with(OIDC_PREFIX)
}

fn environment_str(name: &str) -> Option<SecretValue> {
    let name = python_name(name);
    if name.contains(['=', '\0']) {
        return None;
    }
    std::env::var(name.as_ref())
        .ok()
        .filter(|value| str_to_bool(value).is_none())
        .map(SecretValue::new)
}

impl SecretSource for LocalSecrets {
    fn get_secret_str<'a>(
        &'a self,
        name: &'a str,
    ) -> BoxFuture<'a, Result<Option<SecretValue>, Error>> {
        if is_oidc(name) {
            return self.python.get_secret_str(name);
        }
        futures_util::future::ready(Ok(environment_str(name))).boxed()
    }

    fn resolve<'a>(&'a self, names: &'a [&str]) -> BoxFuture<'a, Result<Secrets, Error>> {
        Box::pin(async move {
            let (oidc, environment): (Vec<&str>, Vec<&str>) =
                names.iter().copied().partition(|name| is_oidc(name));
            let oidc = try_join_all(oidc.into_iter().map(|name| async move {
                self.python
                    .get_secret_str(name)
                    .await
                    .map(|value| (name.to_owned(), value))
            }))
            .await?
            .into_iter()
            .collect();
            Ok(Arc::new(LocalLookup {
                environment: environment.into_iter().map(str::to_owned).collect(),
                oidc,
            }) as Secrets)
        })
    }
}

struct LocalLookup {
    environment: HashSet<String>,
    oidc: HashMap<String, Option<SecretValue>>,
}

impl Lookup for LocalLookup {
    fn get(&self, name: &str) -> Option<String> {
        let value = if self.environment.contains(name) {
            environment_str(name)
        } else {
            self.oidc.get(name).cloned().flatten()
        };
        value.map(|value| value.expose().to_owned())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use futures_util::future::{BoxFuture, FutureExt};
    use litellm_secret_value::SecretValue;
    use litellm_secrets::{Error, source::SecretSource};
    use rstest::rstest;

    use super::LocalSecrets;

    #[derive(Default)]
    struct Recording(Mutex<Vec<String>>);

    impl Recording {
        fn reads(&self) -> Vec<String> {
            self.0.lock().unwrap().clone()
        }
    }

    impl SecretSource for Recording {
        fn get_secret_str<'a>(
            &'a self,
            name: &'a str,
        ) -> BoxFuture<'a, Result<Option<SecretValue>, Error>> {
            self.0.lock().unwrap().push(name.to_owned());
            futures_util::future::ready(Ok(Some(SecretValue::new(format!("python-{name}")))))
                .boxed()
        }
    }

    fn local() -> (LocalSecrets, Arc<Recording>) {
        let python = Arc::new(Recording::default());
        (
            LocalSecrets::new(Arc::clone(&python) as Arc<dyn SecretSource>),
            python,
        )
    }

    fn set(name: &str, value: Option<&str>) {
        match value {
            Some(value) => unsafe { std::env::set_var(name, value) },
            None => unsafe { std::env::remove_var(name) },
        }
    }

    #[rstest]
    #[case::plain(
        "LITELLM_LOCAL_PLAIN",
        "LITELLM_LOCAL_PLAIN",
        Some("value"),
        Some("value")
    )]
    #[case::empty("LITELLM_LOCAL_EMPTY", "LITELLM_LOCAL_EMPTY", Some(""), Some(""))]
    #[case::unset("LITELLM_LOCAL_UNSET", "LITELLM_LOCAL_UNSET", None, None)]
    #[case::reference(
        "LITELLM_LOCAL_REFERENCE",
        "os.environ/LITELLM_LOCAL_REFERENCE",
        Some("value"),
        Some("value")
    )]
    #[case::every_marker_removed(
        "LITELLM_LOCAL_TWICE",
        "os.environ/os.environ/LITELLM_LOCAL_TWICE",
        Some("value"),
        Some("value")
    )]
    #[case::marker_inside_only_when_prefixed(
        "LITELLM_LOCAL_INNER",
        "LITELLM_LOCAL_os.environ/INNER",
        Some("value"),
        None
    )]
    #[case::boolean_true("LITELLM_LOCAL_TRUE", "LITELLM_LOCAL_TRUE", Some(" TRUE\n"), None)]
    #[case::boolean_false("LITELLM_LOCAL_FALSE", "LITELLM_LOCAL_FALSE", Some("False"), None)]
    #[case::boolean_like("LITELLM_LOCAL_ONE", "LITELLM_LOCAL_ONE", Some("1"), Some("1"))]
    #[case::equals_in_name("LITELLM_LOCAL_EQ", "LITELLM_LOCAL_EQ=x", Some("x=y"), None)]
    #[tokio::test]
    async fn reads_what_get_secret_str_reads_from_the_environment(
        #[case] variable: &str,
        #[case] name: &str,
        #[case] value: Option<&str>,
        #[case] expected: Option<&str>,
    ) {
        set(variable, value);
        let (secrets, python) = local();

        let read = secrets.get_secret_str(name).await.unwrap();
        let resolved = secrets.resolve(&[name]).await.unwrap().get(name);
        set(variable, None);

        assert_eq!(read.as_ref().map(SecretValue::expose), expected);
        assert_eq!(resolved.as_deref(), expected);
        assert!(python.reads().is_empty());
    }

    #[rstest]
    #[case::direct("oidc/env/LITELLM_LOCAL_TOKEN")]
    #[case::referenced("os.environ/oidc/env/LITELLM_LOCAL_TOKEN")]
    #[tokio::test]
    async fn oidc_references_are_left_to_python(#[case] name: &str) {
        let (secrets, python) = local();

        let read = secrets.get_secret_str(name).await.unwrap();
        let resolved = secrets
            .resolve(&[name, "LITELLM_LOCAL_BESIDE"])
            .await
            .unwrap();

        let expected = format!("python-{name}");
        assert_eq!(
            read.as_ref().map(SecretValue::expose),
            Some(expected.as_str())
        );
        assert_eq!(resolved.get(name).as_deref(), Some(expected.as_str()));
        assert_eq!(python.reads(), vec![name.to_owned(), name.to_owned()]);
    }

    #[tokio::test]
    async fn resolved_names_are_read_when_looked_up_and_others_are_absent() {
        let name = "LITELLM_LOCAL_LAZY";
        let other = "LITELLM_LOCAL_NOT_RESOLVED";
        set(name, None);
        set(other, Some("present"));
        let (secrets, python) = local();

        let resolved = secrets.resolve(&[name]).await.unwrap();
        set(name, Some("set-after-resolve"));
        let looked_up = resolved.get(name);
        let unrequested = resolved.get(other);
        set(name, None);
        set(other, None);

        assert_eq!(looked_up.as_deref(), Some("set-after-resolve"));
        assert_eq!(unrequested, None);
        assert!(python.reads().is_empty());
    }
}
