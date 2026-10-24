use std::sync::{Arc, LazyLock};

use litellm_model_catalog::{BUNDLED_REGISTRY, Catalog, Installed, Origin, git_blob_id};
use time::OffsetDateTime;
use tokio::sync::watch;

use crate::CostMapSettings;

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, strum::VariantArray)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    Local,
    Remote,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Provenance {
    pub source: Source,
    pub url: Option<String>,
    pub is_env_forced: bool,
    pub fallback_reason: Option<String>,
    pub loaded_at: OffsetDateTime,
    pub checked_at: Option<OffsetDateTime>,
}

#[derive(Clone, Debug)]
pub struct Served {
    pub installed: Installed,
    pub provenance: Provenance,
}

#[derive(Clone)]
pub struct CatalogReader(watch::Receiver<Arc<Served>>);

pub struct CatalogWriter(watch::Sender<Arc<Served>>);

static BUNDLED: LazyLock<Installed> = LazyLock::new(|| {
    let catalog = Catalog::parse(BUNDLED_REGISTRY)
        .expect("bundled model catalog must parse")
        .value;
    Installed {
        catalog: Arc::new(catalog),
        revision: git_blob_id(BUNDLED_REGISTRY),
        etag: None,
        origin: Origin::Bundled,
    }
});

pub fn bundled(settings: &CostMapSettings, now: OffsetDateTime) -> (CatalogWriter, CatalogReader) {
    let served = Served {
        installed: BUNDLED.clone(),
        provenance: Provenance {
            source: Source::Local,
            url: settings.remote_url().map(str::to_owned),
            is_env_forced: settings.is_env_forced(),
            fallback_reason: settings
                .remote_url()
                .map(|_| "remote revalidation pending".to_owned()),
            loaded_at: now,
            checked_at: None,
        },
    };
    let (sender, receiver) = watch::channel(Arc::new(served));
    (CatalogWriter(sender), CatalogReader(receiver))
}

pub(crate) fn bundled_model_count() -> usize {
    BUNDLED.catalog.model_count()
}

impl CatalogReader {
    pub fn current(&self) -> Arc<Served> {
        self.0.borrow().clone()
    }

    pub async fn changed(&mut self) -> Result<(), watch::error::RecvError> {
        self.0.changed().await
    }
}

impl CatalogWriter {
    pub(crate) fn current(&self) -> Arc<Served> {
        self.0.borrow().clone()
    }

    pub(crate) fn install(&self, served: Served) {
        self.0.send_replace(Arc::new(served));
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn at(seconds: i64) -> OffsetDateTime {
        OffsetDateTime::from_unix_timestamp(seconds).unwrap()
    }

    #[rstest]
    #[tokio::test]
    async fn snapshots_remain_stable_and_notify_after_install() {
        let (writer, mut reader) = bundled(&CostMapSettings::default(), at(1_767_323_040));
        let previous = reader.current();
        let previous_catalog = previous.installed.catalog.clone();
        let previous_revision = previous.installed.revision.clone();
        let mut next = (*previous).clone();
        next.installed.catalog = Arc::new((*previous_catalog).clone());
        next.installed.revision = "remote-revision".to_owned();
        next.installed.etag = Some("\"remote\"".to_owned());
        next.installed.origin = Origin::Remote {
            url: "https://catalog.example.test/prices.json".to_owned(),
        };
        next.provenance.source = Source::Remote;
        next.provenance.url = Some("https://catalog.example.test/prices.json".to_owned());
        next.provenance.fallback_reason = None;
        next.provenance.checked_at = Some(at(1_767_323_100));
        writer.install(next);
        reader.changed().await.unwrap();
        let current = reader.current();
        assert!(Arc::ptr_eq(&previous.installed.catalog, &previous_catalog));
        assert_eq!(previous.installed.revision, previous_revision);
        assert!(previous.provenance.checked_at.is_none());
        assert_eq!(previous.provenance.source, Source::Local);
        assert_eq!(
            previous.provenance.fallback_reason.as_deref(),
            Some("remote revalidation pending")
        );
        assert!(!Arc::ptr_eq(
            &previous.installed.catalog,
            &current.installed.catalog
        ));
        assert_eq!(current.installed.revision, "remote-revision");
        assert_eq!(current.installed.etag.as_deref(), Some("\"remote\""));
        assert_eq!(current.provenance.source, Source::Remote);
        assert_eq!(
            current.provenance.url.as_deref(),
            Some("https://catalog.example.test/prices.json")
        );
        assert_eq!(current.provenance.fallback_reason, None);
        assert_eq!(current.provenance.checked_at, Some(at(1_767_323_100)));
    }

    #[rstest]
    fn installation_succeeds_without_receivers() {
        let (writer, reader) = bundled(&CostMapSettings::default(), at(1_767_323_040));
        drop(reader);
        let mut next = (*writer.current()).clone();
        next.installed.revision = "unobserved-remote-revision".to_owned();
        writer.install(next);
        assert_eq!(
            writer.current().installed.revision,
            "unobserved-remote-revision"
        );
    }

    #[rstest]
    fn disabled_remote_has_no_pending_fallback() {
        let settings = CostMapSettings::from_partial(crate::CostMapKeysPartial {
            local_only: Some(true),
            offline: None,
            url: None,
        });
        let (_, reader) = bundled(&settings, at(1_767_323_040));
        let served = reader.current();
        assert_eq!(served.provenance.url, None);
        assert_eq!(served.provenance.fallback_reason, None);
        assert!(served.provenance.is_env_forced);
    }
}
