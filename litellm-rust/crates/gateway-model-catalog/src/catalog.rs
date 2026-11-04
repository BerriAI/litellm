use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use arc_swap::ArcSwap;
use litellm_model_catalog::{BUNDLED_REGISTRY, Catalog, IntegrityLimits};
use litellm_model_catalog_refresh::{
    FetchRequest, Origin, Pause, Provenance, Refreshed, Retrying, SourceClients, SourceFetch,
    refresh,
};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::{Settings, Unavailable};

const MAX_ATTEMPTS: i64 = 3;
const TIMEOUT: Duration = Duration::from_secs(5);

/// One installed map and where it came from, swapped as a whole so a reader never mixes two installs
#[derive(Debug)]
pub struct Installed {
    pub catalog: Arc<Catalog>,
    pub provenance: Provenance,
}

/// The gateway's cost map. It boots from the bundled map, revalidates against the remote source once in the
/// background, and `/reload/model_cost_map` fetches again on demand
pub struct ModelCatalog {
    installed: ArcSwap<Installed>,
    remote: Option<Remote>,
    bundled_models: usize,
    limits: IntegrityLimits,
    shutdown: CancellationToken,
}

struct Remote {
    url: String,
    fetch: Result<SourceFetch, Arc<litellm_model_catalog_refresh::Error>>,
}

impl ModelCatalog {
    /// Never touches the network. A bundled map that does not parse is a build defect and fails boot
    pub fn boot(
        settings: Settings,
        http: litellm_http::Client,
        env: &dyn Fn(&str) -> Option<String>,
        shutdown: CancellationToken,
    ) -> Result<Self, litellm_model_catalog::Error> {
        let bundled = bundled()?;
        let remote = settings.remote_url.map(|url| Remote {
            fetch: SourceFetch::new(&url, SourceClients { http, env }).map_err(Arc::new),
            url,
        });
        let provenance = Provenance {
            origin: Origin::Bundled {
                attempted: remote.as_ref().map(|remote| remote.url.clone()),
            },
            is_env_forced: remote.is_none(),
            fallback_reason: None,
            loaded_at: SystemTime::now(),
            revision: Some(litellm_model_catalog::revision(BUNDLED_REGISTRY)),
            etag: None,
        };
        Ok(Self {
            bundled_models: bundled.model_count(),
            installed: ArcSwap::from_pointee(Installed {
                catalog: Arc::new(bundled),
                provenance,
            }),
            remote,
            limits: settings.limits,
            shutdown,
        })
    }

    pub fn installed(&self) -> Arc<Installed> {
        self.installed.load_full()
    }

    /// The stale-while-revalidate refresh of the boot map, `None` when the bundled map is forced
    pub fn spawn_revalidate(self: &Arc<Self>) -> Option<JoinHandle<()>> {
        self.remote.as_ref()?;
        let catalog = Arc::clone(self);
        Some(tokio::spawn(async move { catalog.revalidate().await }))
    }

    /// Replaces only the boot map, so a reload that landed first is never overwritten by this older fetch
    pub async fn revalidate(&self) {
        let boot = self.installed();
        let Some(remote) = &self.remote else {
            return;
        };
        let replacement = match self.fetch(remote).await {
            Ok(fetched) => fetched,
            Err(Unavailable::ShuttingDown) => return,
            Err(unavailable) => {
                tracing::warn!(
                    reason = %litellm_error::Report::new(&unavailable),
                    "could not refresh the model cost map, keeping the bundled map"
                );
                kept_after(&boot, &unavailable)
            }
        };
        self.installed
            .compare_and_swap(&boot, Arc::new(replacement));
    }

    /// `refetch_model_cost_map`: a failure keeps the current map, and a forced bundled map is installed again
    pub async fn reload(&self) -> Result<Arc<Installed>, Unavailable> {
        let installed = match &self.remote {
            None => restamped(&self.installed()),
            Some(remote) => self.fetch(remote).await?,
        };
        let installed = Arc::new(installed);
        self.installed.store(Arc::clone(&installed));
        Ok(installed)
    }

    async fn fetch(&self, remote: &Remote) -> Result<Installed, Unavailable> {
        let fetch = remote.fetch.as_ref().map_err(|cause| Unavailable::Source {
            url: remote.url.clone(),
            cause: Arc::clone(cause),
        })?;
        let request = FetchRequest {
            if_none_match: None,
            timeout: TIMEOUT,
        };
        let mut pause = TokioPause;
        let Ok(refreshed) = tokio::select! {
            () = self.shutdown.cancelled() => return Err(Unavailable::ShuttingDown),
            refreshed = refresh(fetch, &request, MAX_ATTEMPTS, &mut pause) => refreshed,
        };
        let (body, revision, etag) = match refreshed {
            Refreshed::Adopted {
                body,
                revision,
                etag,
            } => (body, revision, etag),
            Refreshed::NotModified => {
                return Err(Unavailable::NotModified {
                    url: remote.url.clone(),
                });
            }
            Refreshed::Failed { reason } => return Err(Unavailable::Fetch { reason }),
        };
        let catalog = validated(&body, &remote.url, self.bundled_models, self.limits)?;
        Ok(Installed {
            catalog: Arc::new(catalog),
            provenance: Provenance {
                origin: Origin::Remote {
                    url: remote.url.clone(),
                },
                is_env_forced: false,
                fallback_reason: None,
                loaded_at: SystemTime::now(),
                revision: Some(revision),
                etag,
            },
        })
    }
}

fn restamped(installed: &Installed) -> Installed {
    Installed {
        catalog: Arc::clone(&installed.catalog),
        provenance: Provenance {
            loaded_at: SystemTime::now(),
            ..installed.provenance.clone()
        },
    }
}

fn bundled() -> Result<Catalog, litellm_model_catalog::Error> {
    let loaded = Catalog::parse(BUNDLED_REGISTRY)?;
    Ok(loaded.value.without_rejected(&loaded.errors).0)
}

/// Rows that fail validation are left out, and the rest must still pass the size check against the bundled map
fn validated(
    body: &[u8],
    url: &str,
    reference: usize,
    limits: IntegrityLimits,
) -> Result<Catalog, Unavailable> {
    let loaded = Catalog::parse(body).map_err(|cause| Unavailable::Invalid {
        url: url.to_owned(),
        cause,
    })?;
    let (catalog, rejected) = loaded.value.without_rejected(&loaded.errors);
    for error in rejected {
        tracing::warn!(
            path = error.path(),
            kind = <&str>::from(error.kind()),
            "model cost map row rejected"
        );
    }
    catalog
        .check_size(reference, limits)
        .map_err(|cause| Unavailable::Integrity {
            url: url.to_owned(),
            cause,
        })?;
    Ok(catalog)
}

/// The boot map stays, now standing in for the failed fetch, as Python's `_keep_bundled_after`
fn kept_after(boot: &Installed, unavailable: &Unavailable) -> Installed {
    let fallback_reason = match unavailable {
        Unavailable::Integrity { .. } => "Remote data failed integrity validation".to_owned(),
        other => format!("Remote fetch failed: {}", litellm_error::Report::new(other)),
    };
    Installed {
        catalog: Arc::clone(&boot.catalog),
        provenance: Provenance {
            fallback_reason: Some(fallback_reason),
            loaded_at: SystemTime::now(),
            ..boot.provenance.clone()
        },
    }
}

struct TokioPause;

impl Pause for TokioPause {
    type Stop = Infallible;

    fn jitter(&mut self) -> f64 {
        rand::random()
    }

    fn retrying(&mut self, retrying: &Retrying<'_>) {
        tracing::warn!(
            attempt = retrying.attempt,
            max_attempts = retrying.max_attempts,
            reason = retrying.reason,
            wait_seconds = retrying.wait_seconds,
            "model cost map fetch failed, retrying"
        );
    }

    async fn sleep(&mut self, seconds: f64) -> Result<(), Infallible> {
        tokio::time::sleep(Duration::try_from_secs_f64(seconds).unwrap_or_default()).await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use litellm_model_catalog::revision;
    use rstest::rstest;
    use serde_json::json;
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::test_support::{catalog, document, file, input_price};

    #[rstest]
    #[tokio::test]
    async fn boot_serves_the_bundled_map_and_names_the_source_it_will_revalidate_against() {
        let url = "http://127.0.0.1:1/unreachable.json".to_owned();
        let catalog = catalog(Some(url.clone()), CancellationToken::new());
        let installed = catalog.installed();
        assert_eq!(
            installed.provenance.origin,
            Origin::Bundled {
                attempted: Some(url)
            }
        );
        assert!(!installed.provenance.is_env_forced);
        assert_eq!(installed.provenance.fallback_reason, None);
        assert_eq!(
            installed.provenance.revision.as_deref(),
            Some(revision(BUNDLED_REGISTRY).as_str())
        );
        assert_eq!(
            installed.catalog.model_count(),
            bundled().unwrap().model_count()
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_forced_bundled_map_never_revalidates() {
        let catalog = Arc::new(catalog(None, CancellationToken::new()));
        assert!(catalog.spawn_revalidate().is_none());
        let provenance = &catalog.installed().provenance;
        assert_eq!(provenance.origin, Origin::Bundled { attempted: None });
        assert!(provenance.is_env_forced);
    }

    #[rstest]
    #[tokio::test]
    async fn revalidate_adopts_the_remote_map_with_its_revision() {
        let body = document(&[("gpt-test", 0.25)]);
        let (_directory, url) = file(&body);
        let catalog = catalog(Some(url.clone()), CancellationToken::new());
        catalog.revalidate().await;
        let installed = catalog.installed();
        assert_eq!(installed.provenance.origin, Origin::Remote { url });
        assert_eq!(installed.provenance.revision, Some(revision(&body)));
        assert!(!installed.provenance.is_env_forced);
        assert_eq!(
            input_price(&installed.catalog, "gpt-test"),
            Some(json!(0.25))
        );
        assert_eq!(installed.catalog.model_count(), 1);
    }

    #[rstest]
    #[case::unreadable(None, "Remote fetch failed: could not fetch file://")]
    #[case::not_json(Some(b"not json".as_slice()), "Remote fetch failed: invalid model cost map from file://")]
    #[case::every_row_invalid(
        Some(br#"{"bad": {"input_cost_per_token": "free"}}"#.as_slice()),
        "Remote data failed integrity validation"
    )]
    #[tokio::test]
    async fn a_failed_revalidation_keeps_the_bundled_map_and_records_why(
        #[case] body: Option<&[u8]>,
        #[case] fallback_prefix: &str,
    ) {
        let (directory, url) = file(body.unwrap_or_default());
        if body.is_none() {
            std::fs::remove_file(directory.path().join("model_prices.json")).unwrap();
        }
        let catalog = catalog(Some(url.clone()), CancellationToken::new());
        let boot = catalog.installed();
        catalog.revalidate().await;
        let installed = catalog.installed();
        assert!(Arc::ptr_eq(&installed.catalog, &boot.catalog));
        assert_eq!(
            installed.provenance.origin,
            Origin::Bundled {
                attempted: Some(url)
            }
        );
        let reason = installed.provenance.fallback_reason.as_deref().unwrap();
        assert!(reason.starts_with(fallback_prefix), "{reason}");
        assert_eq!(installed.provenance.revision, boot.provenance.revision);
    }

    #[rstest]
    #[tokio::test]
    async fn a_reload_that_lands_during_revalidation_is_not_overwritten() {
        let server = MockServer::start().await;
        let stale = document(&[("gpt-test", 1.0)]);
        let fresh = document(&[("gpt-test", 2.0)]);
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_bytes(stale)
                    .set_delay(Duration::from_millis(500)),
            )
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(fresh.clone()))
            .mount(&server)
            .await;
        let catalog = Arc::new(catalog(Some(server.uri()), CancellationToken::new()));
        let revalidating = catalog.spawn_revalidate().unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        catalog.reload().await.unwrap();
        revalidating.await.unwrap();
        let installed = catalog.installed();
        assert_eq!(installed.provenance.revision, Some(revision(&fresh)));
        assert_eq!(
            input_price(&installed.catalog, "gpt-test"),
            Some(json!(2.0))
        );
    }

    #[rstest]
    #[case::in_flight_fetch(ResponseTemplate::new(200).set_delay(Duration::from_secs(30)))]
    #[case::retry_wait(ResponseTemplate::new(503))]
    #[tokio::test]
    async fn shutdown_cancels_revalidation_and_keeps_the_boot_map(
        #[case] response: ResponseTemplate,
    ) {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(response)
            .mount(&server)
            .await;
        let shutdown = CancellationToken::new();
        let catalog = Arc::new(catalog(Some(server.uri()), shutdown.clone()));
        let boot = catalog.installed();
        let revalidating = catalog.spawn_revalidate().unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;
        let cancelled = Instant::now();
        shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(1), revalidating)
            .await
            .expect("revalidation stops at shutdown")
            .unwrap();
        assert!(cancelled.elapsed() < Duration::from_secs(1));
        assert!(Arc::ptr_eq(&catalog.installed(), &boot));
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }

    #[rstest]
    #[tokio::test]
    async fn a_failed_reload_keeps_the_current_map_untouched() {
        let (directory, url) = file(&document(&[("gpt-test", 0.5)]));
        let catalog = catalog(Some(url), CancellationToken::new());
        catalog.revalidate().await;
        let before = catalog.installed();
        std::fs::write(directory.path().join("model_prices.json"), b"[]").unwrap();
        let unavailable = catalog.reload().await.unwrap_err();
        assert!(
            matches!(unavailable, Unavailable::Invalid { .. }),
            "{unavailable:?}"
        );
        assert!(Arc::ptr_eq(&catalog.installed(), &before));
    }

    #[rstest]
    #[case::unsupported_scheme("ftp://example.com/map.json")]
    #[case::s3_without_key("s3://bucket/")]
    #[tokio::test]
    async fn a_source_that_cannot_be_built_fails_every_fetch(#[case] url: &str) {
        let catalog = catalog(Some(url.to_owned()), CancellationToken::new());
        let unavailable = catalog.reload().await.unwrap_err();
        assert!(
            matches!(unavailable, Unavailable::Source { .. }),
            "{unavailable:?}"
        );
        assert_eq!(unavailable.to_string(), format!("could not fetch {url}"));
    }

    #[rstest]
    #[tokio::test]
    async fn reloading_a_forced_bundled_map_restamps_it_without_fetching() {
        let catalog = catalog(None, CancellationToken::new());
        let boot = catalog.installed();
        let reloaded = catalog.reload().await.unwrap();
        assert!(Arc::ptr_eq(&reloaded.catalog, &boot.catalog));
        assert!(reloaded.provenance.loaded_at >= boot.provenance.loaded_at);
        assert_eq!(
            Provenance {
                loaded_at: boot.provenance.loaded_at,
                ..reloaded.provenance.clone()
            },
            boot.provenance
        );
        assert!(Arc::ptr_eq(&catalog.installed(), &reloaded));
    }

    fn limits(min_model_count: usize) -> IntegrityLimits {
        IntegrityLimits::new(min_model_count, 0.0).unwrap()
    }

    #[rstest]
    #[case::valid_rows_only(
        br#"{"a": {"input_cost_per_token": 1}, "b": {"input_cost_per_token": 2}}"#,
        2,
        Ok(2)
    )]
    #[case::invalid_row_left_out(
        br#"{"a": {"input_cost_per_token": 1}, "b": {"input_cost_per_token": "x"}}"#,
        1,
        Ok(1)
    )]
    #[case::size_counts_rows_left_after_rejection(
        br#"{"a": {"input_cost_per_token": 1}, "b": {"input_cost_per_token": "x"}}"#,
        2,
        Err(true)
    )]
    fn fetched_maps_serve_their_valid_rows_and_must_keep_enough_of_them(
        #[case] body: &[u8],
        #[case] min_model_count: usize,
        #[case] expected: Result<usize, bool>,
    ) {
        let served = validated(body, "file:///map.json", 0, limits(min_model_count))
            .map(|catalog| catalog.model_count())
            .map_err(|unavailable| matches!(unavailable, Unavailable::Integrity { .. }));
        assert_eq!(served, expected);
    }

    #[rstest]
    fn a_map_that_shrank_against_the_bundled_one_is_refused() {
        let body = document(&[("a", 1.0), ("b", 1.0)]);
        let shrunk = validated(
            &body,
            "file:///map.json",
            5,
            IntegrityLimits::new(0, 0.5).unwrap(),
        );
        assert!(matches!(shrunk, Err(Unavailable::Integrity { .. })));
        let kept = validated(
            &body,
            "file:///map.json",
            4,
            IntegrityLimits::new(0, 0.5).unwrap(),
        );
        assert!(kept.is_ok());
    }
}
