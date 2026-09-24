use std::{cell::OnceCell, collections::HashMap, sync::Arc};

use futures_util::future::{BoxFuture, try_join_all};
use litellm_core_utils::settings::Lookup;

use crate::{Error, SecretResolver, SecretValue};

pub type Secrets = Arc<dyn Lookup + Send + Sync>;

pub trait SecretSource: Send + Sync {
    fn get_secret_str<'a>(
        &'a self,
        name: &'a str,
    ) -> BoxFuture<'a, Result<Option<SecretValue>, Error>>;

    fn resolve<'a>(&'a self, names: &'a [&str]) -> BoxFuture<'a, Result<Secrets, Error>> {
        Box::pin(async move {
            let values = try_join_all(names.iter().map(|name| async move {
                self.get_secret_str(name)
                    .await
                    .map(|value| ((*name).to_owned(), value))
            }))
            .await?
            .into_iter()
            .collect();
            Ok(Arc::new(SecretSnapshot { values }) as Secrets)
        })
    }
}

impl SecretSource for SecretResolver {
    fn get_secret_str<'a>(
        &'a self,
        name: &'a str,
    ) -> BoxFuture<'a, Result<Option<SecretValue>, Error>> {
        Box::pin(SecretResolver::get_secret_str(self, name, None))
    }
}

#[derive(Default)]
pub struct EnvironmentSecrets(SecretResolver);

impl EnvironmentSecrets {
    pub fn python_compatible() -> Self {
        Self(SecretResolver::new_python_compatible(
            Arc::new(crate::SecretManagerState::default()),
            Arc::new(litellm_core_utils::settings::ProcessEnvironment),
            crate::OidcResolver::default(),
        ))
    }
}

impl SecretSource for EnvironmentSecrets {
    fn get_secret_str<'a>(
        &'a self,
        name: &'a str,
    ) -> BoxFuture<'a, Result<Option<SecretValue>, Error>> {
        Box::pin(self.0.get_secret_str(name, None))
    }
}

struct SecretSnapshot {
    values: HashMap<String, Option<SecretValue>>,
}

impl Lookup for SecretSnapshot {
    fn get(&self, name: &str) -> Option<String> {
        self.values
            .get(name)
            .and_then(Option::as_ref)
            .map(|value| value.expose().to_owned())
    }
}

struct OnDemand<'a> {
    fetched: &'a [(String, Option<SecretValue>)],
    first_missing: OnceCell<String>,
}

impl Lookup for OnDemand<'_> {
    fn get(&self, name: &str) -> Option<String> {
        match self.fetched.iter().find(|(fetched, _)| fetched == name) {
            Some((_, value)) => value.as_ref().map(|value| value.expose().to_owned()),
            None => {
                let _ = self.first_missing.set(name.to_owned());
                None
            }
        }
    }
}

pub async fn resolve_on_demand<T, E>(
    source: &dyn SecretSource,
    attempt: impl Fn(&dyn Lookup) -> Result<T, E>,
) -> Result<T, E>
where
    E: From<Error>,
{
    let mut fetched: Vec<(String, Option<SecretValue>)> = Vec::new();
    loop {
        let lookup = OnDemand {
            fetched: &fetched,
            first_missing: OnceCell::new(),
        };
        let outcome = attempt(&lookup);
        let Some(name) = lookup.first_missing.into_inner() else {
            return outcome;
        };
        let value = source.get_secret_str(&name).await?;
        fetched.push((name, value));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use rstest::rstest;

    use super::*;

    struct RecordingSource {
        values: &'static [(&'static str, &'static str)],
        failing: Option<&'static str>,
        reads: Mutex<Vec<String>>,
    }

    impl RecordingSource {
        fn new(values: &'static [(&'static str, &'static str)]) -> Self {
            Self {
                values,
                failing: None,
                reads: Mutex::new(Vec::new()),
            }
        }

        fn reads(&self) -> Vec<String> {
            self.reads.lock().unwrap().clone()
        }
    }

    impl SecretSource for RecordingSource {
        fn get_secret_str<'a>(
            &'a self,
            name: &'a str,
        ) -> BoxFuture<'a, Result<Option<SecretValue>, Error>> {
            Box::pin(async move {
                self.reads.lock().unwrap().push(name.to_owned());
                if self.failing == Some(name) {
                    return Err(Error::ManagedSecretMissing);
                }
                Ok(self
                    .values
                    .iter()
                    .find(|(key, _)| *key == name)
                    .map(|(_, value)| SecretValue::new(*value)))
            })
        }
    }

    fn key_then_token(lookup: &dyn Lookup) -> Result<String, Error> {
        lookup
            .get("KEY")
            .or_else(|| lookup.get("TOKEN"))
            .ok_or(Error::MissingEnvironment)
    }

    #[rstest]
    #[case::first_name_found(&[("KEY", "k"), ("TOKEN", "t")], Ok("k"), &["KEY"])]
    #[case::falls_through_to_the_second_name(&[("TOKEN", "t")], Ok("t"), &["KEY", "TOKEN"])]
    #[case::nothing_found(&[], Err(()), &["KEY", "TOKEN"])]
    #[tokio::test]
    async fn reads_only_the_names_the_attempt_asks_for_in_order(
        #[case] values: &'static [(&'static str, &'static str)],
        #[case] expected: Result<&str, ()>,
        #[case] expected_reads: &[&str],
    ) {
        let source = RecordingSource::new(values);
        let outcome = resolve_on_demand(&source, key_then_token).await;
        assert_eq!(
            (outcome.as_deref().map_err(|_| ()), source.reads()),
            (
                expected,
                expected_reads.iter().map(ToString::to_string).collect()
            )
        );
    }

    #[tokio::test]
    async fn an_attempt_that_needs_no_secret_reads_none() {
        let source = RecordingSource {
            failing: Some("KEY"),
            ..RecordingSource::new(&[])
        };
        let outcome: Result<&str, Error> = resolve_on_demand(&source, |_| Ok("given")).await;
        assert_eq!(
            (outcome.ok(), source.reads()),
            (Some("given"), Vec::<String>::new())
        );
    }

    #[tokio::test]
    async fn a_failed_read_ends_the_resolution() {
        let source = RecordingSource {
            failing: Some("KEY"),
            ..RecordingSource::new(&[("TOKEN", "t")])
        };
        let outcome = resolve_on_demand(&source, key_then_token).await;
        assert_eq!(
            (
                matches!(outcome, Err(Error::ManagedSecretMissing)),
                source.reads()
            ),
            (true, vec!["KEY".to_string()])
        );
    }
}
