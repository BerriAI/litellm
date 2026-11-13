use std::{
    collections::HashMap,
    sync::{Arc, Mutex, PoisonError},
};

use futures_util::future::{BoxFuture, FutureExt};
use litellm_secret_value::SecretValue;
use litellm_secrets::{Error, source::SecretSource};

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
}
