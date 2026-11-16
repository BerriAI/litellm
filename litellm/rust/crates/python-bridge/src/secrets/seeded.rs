use std::{
    collections::HashMap,
    sync::{Arc, Mutex, PoisonError},
};

use futures_util::future::{BoxFuture, FutureExt};
use litellm_env::Lookup;
use litellm_secret_value::SecretValue;
use litellm_secrets::{
    Error,
    source::{SecretSource, Secrets},
};

/// A call's secret source that answers the secrets the host already read for the call with the
/// values it read, so a secret Python reads once is not read a second time.
pub(crate) struct SeededSecrets {
    source: Arc<dyn SecretSource>,
    read: Mutex<HashMap<String, Option<SecretValue>>>,
}

impl SeededSecrets {
    pub(crate) fn new(source: Arc<dyn SecretSource>) -> Self {
        Self {
            source,
            read: Mutex::default(),
        }
    }

    pub(crate) fn seed(&self, read: impl IntoIterator<Item = (String, Option<String>)>) {
        self.read
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .extend(
                read.into_iter()
                    .map(|(name, value)| (name, value.map(SecretValue::new))),
            );
    }
}

impl SecretSource for SeededSecrets {
    fn get_secret_str<'a>(
        &'a self,
        name: &'a str,
    ) -> BoxFuture<'a, Result<Option<SecretValue>, Error>> {
        let seeded = self
            .read
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(name)
            .cloned();
        match seeded {
            Some(value) => futures_util::future::ready(Ok(value)).boxed(),
            None => self.source.get_secret_str(name),
        }
    }

    fn resolve<'a>(&'a self, names: &'a [&str]) -> BoxFuture<'a, Result<Secrets, Error>> {
        let (seeded, unseeded) = {
            let read = self.read.lock().unwrap_or_else(PoisonError::into_inner);
            let (seeded, unseeded): (Vec<&str>, Vec<&str>) = names
                .iter()
                .copied()
                .partition(|name| read.contains_key(*name));
            let seeded: HashMap<String, Option<SecretValue>> = seeded
                .into_iter()
                .map(|name| (name.to_owned(), read.get(name).cloned().flatten()))
                .collect();
            (seeded, unseeded)
        };
        Box::pin(async move {
            let source = self.source.resolve(&unseeded).await?;
            Ok(Arc::new(SeededLookup { seeded, source }) as Secrets)
        })
    }
}

struct SeededLookup {
    seeded: HashMap<String, Option<SecretValue>>,
    source: Secrets,
}

impl Lookup for SeededLookup {
    fn get(&self, name: &str) -> Option<String> {
        match self.seeded.get(name) {
            Some(value) => value.as_ref().map(|value| value.expose().to_owned()),
            None => self.source.get(name),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[derive(Default)]
    struct Counting(AtomicUsize);

    impl SecretSource for Counting {
        fn get_secret_str<'a>(
            &'a self,
            name: &'a str,
        ) -> BoxFuture<'a, Result<Option<SecretValue>, Error>> {
            self.0.fetch_add(1, Ordering::SeqCst);
            futures_util::future::ready(Ok(Some(SecretValue::new(format!("source-{name}")))))
                .boxed()
        }
    }

    #[tokio::test]
    async fn a_seeded_secret_is_answered_without_reading_the_source() {
        let source = Arc::new(Counting::default());
        let secrets = SeededSecrets::new(Arc::clone(&source) as Arc<dyn SecretSource>);
        secrets.seed([
            (
                "AZURE_API_BASE".to_owned(),
                Some("http://seeded".to_owned()),
            ),
            ("UNSET".to_owned(), None),
        ]);

        let seeded = secrets.get_secret_str("AZURE_API_BASE").await.unwrap();
        let unset = secrets.get_secret_str("UNSET").await.unwrap();
        let other = secrets.get_secret_str("OTHER").await.unwrap();

        assert_eq!(
            seeded.map(|value| value.expose().to_owned()).as_deref(),
            Some("http://seeded")
        );
        assert!(unset.is_none());
        assert_eq!(
            other.map(|value| value.expose().to_owned()).as_deref(),
            Some("source-OTHER")
        );
        assert_eq!(source.0.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn resolve_answers_seeded_names_and_leaves_the_rest_to_the_source() {
        let source = Arc::new(Counting::default());
        let secrets = SeededSecrets::new(Arc::clone(&source) as Arc<dyn SecretSource>);
        secrets.seed([
            (
                "AZURE_API_BASE".to_owned(),
                Some("http://seeded".to_owned()),
            ),
            ("UNSET".to_owned(), None),
        ]);

        let resolved = secrets
            .resolve(&["AZURE_API_BASE", "UNSET", "OTHER"])
            .await
            .unwrap();

        assert_eq!(
            resolved.get("AZURE_API_BASE").as_deref(),
            Some("http://seeded")
        );
        assert_eq!(resolved.get("UNSET"), None);
        assert_eq!(resolved.get("OTHER").as_deref(), Some("source-OTHER"));
        assert_eq!(resolved.get("NOT_REQUESTED"), None);
        assert_eq!(source.0.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn resolve_keeps_a_lazy_source_lazy() {
        let name = "LITELLM_SEEDED_LAZY";
        unsafe { std::env::remove_var(name) };
        let python = Arc::new(Counting::default());
        let secrets = SeededSecrets::new(Arc::new(crate::secrets::local::LocalSecrets::new(
            Arc::clone(&python) as Arc<dyn SecretSource>,
        )));

        let resolved = secrets.resolve(&[name]).await.unwrap();
        unsafe { std::env::set_var(name, "set-after-resolve") };
        let looked_up = resolved.get(name);
        unsafe { std::env::remove_var(name) };

        assert_eq!(looked_up.as_deref(), Some("set-after-resolve"));
        assert_eq!(python.0.load(Ordering::SeqCst), 0);
    }
}
