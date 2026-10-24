use std::{sync::Arc, time::Duration};

use http::header::{ETAG, IF_NONE_MATCH, RETRY_AFTER};
use litellm_http::Client;
use litellm_model_catalog::{
    Installed, IntegrityLimits, Origin, Reply, RetryPolicy, Revalidation, Step,
};
use time::OffsetDateTime;
use tokio_util::sync::CancellationToken;

use crate::{CatalogWriter, Provenance, Served, Source, state::bundled_model_count};

pub async fn revalidate(
    writer: CatalogWriter,
    client: Client,
    url: String,
    seed: Option<u64>,
    shutdown: CancellationToken,
) {
    let current = writer.current();
    let policy = RetryPolicy::default();
    let (mut revalidation, mut step) = Revalidation::new(url, &current.installed, policy, seed);
    loop {
        if shutdown.is_cancelled() {
            return;
        }
        step = match step {
            Step::Fetch(request) => {
                let reply = tokio::select! {
                    biased;
                    () = shutdown.cancelled() => return,
                    reply = perform(&client, request) => reply,
                };
                let work = tokio::task::spawn_blocking(move || {
                    let step = revalidation.on_reply(reply);
                    (revalidation, filter_adopt(step, policy))
                });
                match work.await {
                    Ok((next, step)) => {
                        revalidation = next;
                        step
                    }
                    Err(_) => {
                        tracing::warn!("model cost map revalidation worker failed");
                        return;
                    }
                }
            }
            Step::Sleep(wait) => {
                if !sleep_or_cancel(wait, &shutdown).await {
                    return;
                }
                revalidation.on_slept()
            }
            Step::Adopt { installed, errors } => {
                if shutdown.is_cancelled() {
                    return;
                }
                writer.install(adopted(installed, errors, &current.provenance));
                return;
            }
            Step::Done { reason } => {
                if shutdown.is_cancelled() {
                    return;
                }
                writer.install(checked(&current, reason));
                return;
            }
        };
    }
}

async fn sleep_or_cancel(wait: Duration, shutdown: &CancellationToken) -> bool {
    tokio::select! {
        biased;
        () = shutdown.cancelled() => false,
        () = tokio::time::sleep(wait) => true,
    }
}

async fn perform(client: &Client, request: litellm_model_catalog::FetchRequest) -> Reply {
    let mut builder = client.get(&request.url).timeout(request.timeout);
    if let Some(etag) = request.if_none_match {
        builder = builder.header(IF_NONE_MATCH, etag);
    }
    let response = match builder.send().await {
        Ok(response) => response,
        Err(error) => {
            return Reply::Transport {
                reason: error.without_url().to_string(),
            };
        }
    };
    let status = response.status().as_u16();
    let etag = response
        .headers()
        .get(ETAG)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let retry_after = response
        .headers()
        .get(RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let body = match response.bytes().await {
        Ok(body) => body,
        Err(error) => {
            return Reply::Transport {
                reason: error.without_url().to_string(),
            };
        }
    };
    Reply::Response {
        status,
        etag,
        retry_after,
        body,
    }
}

fn filter_adopt(step: Step, policy: RetryPolicy) -> Step {
    let Step::Adopt { installed, errors } = step else {
        return step;
    };
    let (catalog, rejected) = (*installed.catalog).clone().without_rejected(&errors);
    for error in rejected {
        tracing::warn!(pointer = %error.path, "rejected model cost map row");
    }
    let limits = IntegrityLimits::new(policy.min_model_count, policy.min_reference_ratio)
        .expect("default retry policy has valid integrity limits");
    if let Err(error) = catalog.check_size(bundled_model_count(), limits) {
        return Step::Done {
            reason: Some(error.to_string()),
        };
    }
    Step::Adopt {
        installed: Installed {
            catalog: Arc::new(catalog),
            ..installed
        },
        errors,
    }
}

fn adopted(
    installed: Installed,
    _errors: Vec<litellm_model_catalog::Located>,
    previous: &Provenance,
) -> Served {
    let now = OffsetDateTime::now_utc();
    let url = match &installed.origin {
        Origin::Bundled => previous.url.clone(),
        Origin::Remote { url } => Some(url.clone()),
    };
    Served {
        installed,
        provenance: Provenance {
            source: Source::Remote,
            url,
            is_env_forced: previous.is_env_forced,
            fallback_reason: None,
            loaded_at: now,
            checked_at: Some(now),
        },
    }
}

fn checked(current: &Served, reason: Option<String>) -> Served {
    let now = OffsetDateTime::now_utc();
    Served {
        installed: current.installed.clone(),
        provenance: Provenance {
            source: current.provenance.source.clone(),
            url: current.provenance.url.clone(),
            is_env_forced: current.provenance.is_env_forced,
            fallback_reason: reason,
            loaded_at: current.provenance.loaded_at,
            checked_at: Some(now),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        time::Duration,
    };

    use axum::{
        Router,
        body::Body,
        extract::State,
        http::{HeaderMap, StatusCode},
        response::Response,
        routing::get,
    };
    use bytes::Bytes;
    use futures_util::{StreamExt, stream};
    use litellm_http::{
        ClientVariant, HttpClientPool, HttpSettings, Resolution, media::PublicDnsResolver,
    };
    use litellm_model_catalog::BUNDLED_REGISTRY;
    use rstest::rstest;
    use tokio::{
        net::TcpListener,
        sync::{Notify, mpsc},
        task::JoinHandle,
        time::timeout,
    };

    use super::*;

    async fn serve(router: Router) -> (String, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        (format!("http://{address}/"), task)
    }

    fn client() -> Client {
        let pool = HttpClientPool::new(Arc::new(PublicDnsResolver));
        let http = Resolution::from(&HttpSettings::default()).config;
        pool.client(&http, ClientVariant::Provider).unwrap()
    }

    #[rstest]
    #[tokio::test]
    async fn not_modified_preserves_installed_catalog_and_etag() {
        let (seen_tx, mut seen_rx) = mpsc::unbounded_channel();
        let router =
            Router::new().route(
                "/",
                get(
                    |State(tx): State<mpsc::UnboundedSender<Option<String>>>,
                     headers: HeaderMap| async move {
                        tx.send(
                            headers
                                .get(IF_NONE_MATCH)
                                .and_then(|value| value.to_str().ok())
                                .map(str::to_owned),
                        )
                        .unwrap();
                        StatusCode::NOT_MODIFIED
                    },
                ),
            );
        let (url, server) = serve(router.with_state(seen_tx)).await;
        let (writer, mut reader) = crate::bundled(
            &crate::CostMapSettings::default(),
            OffsetDateTime::now_utc(),
        );
        let mut initial = (*writer.current()).clone();
        initial.installed.etag = Some("\"current\"".to_owned());
        initial.installed.origin = Origin::Remote { url: url.clone() };
        initial.provenance.source = Source::Remote;
        initial.provenance.fallback_reason = Some("previous failure".to_owned());
        let original_catalog = initial.installed.catalog.clone();
        let original_revision = initial.installed.revision.clone();
        let original_loaded_at = initial.provenance.loaded_at;
        let original_checked_at = OffsetDateTime::from_unix_timestamp(1).unwrap();
        initial.provenance.checked_at = Some(original_checked_at);
        writer.install(initial);
        reader.changed().await.unwrap();
        let task = tokio::spawn(revalidate(
            writer,
            client(),
            url,
            Some(1),
            CancellationToken::new(),
        ));
        let observed = timeout(Duration::from_secs(5), seen_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(observed.as_deref(), Some("\"current\""));
        timeout(Duration::from_secs(5), reader.changed())
            .await
            .unwrap()
            .unwrap();
        task.await.unwrap();
        let updated = reader.current();
        assert!(Arc::ptr_eq(&updated.installed.catalog, &original_catalog));
        assert_eq!(updated.installed.revision, original_revision);
        assert_eq!(updated.installed.etag.as_deref(), Some("\"current\""));
        assert_eq!(updated.provenance.source, Source::Remote);
        assert_eq!(updated.provenance.fallback_reason, None);
        assert_eq!(updated.provenance.loaded_at, original_loaded_at);
        assert!(updated.provenance.checked_at.unwrap() > original_checked_at);
        server.abort();
    }

    #[derive(Clone)]
    struct CallCount(Arc<AtomicUsize>);

    async fn fail_body_once(State(count): State<CallCount>) -> Response<Body> {
        if count.0.fetch_add(1, Ordering::SeqCst) == 0 {
            let chunks = stream::iter([Ok::<_, io::Error>(Bytes::from_static(b"{\"partial\":"))])
                .chain(stream::once(async {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    Err(io::Error::other("truncated response"))
                }));
            return Response::builder()
                .status(StatusCode::OK)
                .body(Body::from_stream(chunks))
                .unwrap();
        }
        Response::builder()
            .status(StatusCode::OK)
            .header(ETAG, "\"recovered\"")
            .body(Body::from(BUNDLED_REGISTRY))
            .unwrap()
    }

    #[rstest]
    #[tokio::test]
    async fn truncated_body_is_a_transport_reply() {
        let count = CallCount(Arc::new(AtomicUsize::new(0)));
        let router = Router::new()
            .route("/", get(fail_body_once))
            .with_state(count.clone());
        let (url, server) = serve(router).await;
        let request = litellm_model_catalog::FetchRequest {
            url,
            if_none_match: None,
            timeout: Duration::from_secs(5),
        };
        let reply = timeout(Duration::from_secs(5), perform(&client(), request))
            .await
            .unwrap();
        assert!(matches!(reply, Reply::Transport { .. }));
        assert_eq!(count.0.load(Ordering::SeqCst), 1);
        server.abort();
    }

    #[rstest]
    #[tokio::test]
    async fn truncated_body_retries_and_recovers() {
        let count = CallCount(Arc::new(AtomicUsize::new(0)));
        let router = Router::new()
            .route("/", get(fail_body_once))
            .with_state(count.clone());
        let (url, server) = serve(router).await;
        let (writer, mut reader) = crate::bundled(
            &crate::CostMapSettings::default(),
            OffsetDateTime::now_utc(),
        );
        let task = tokio::spawn(revalidate(
            writer,
            client(),
            url,
            Some(1),
            CancellationToken::new(),
        ));
        timeout(Duration::from_secs(10), reader.changed())
            .await
            .unwrap()
            .unwrap();
        task.await.unwrap();
        assert_eq!(count.0.load(Ordering::SeqCst), 2);
        assert_eq!(
            reader.current().installed.etag.as_deref(),
            Some("\"recovered\"")
        );
        assert_eq!(reader.current().provenance.source, Source::Remote);
        server.abort();
    }

    #[derive(Clone)]
    struct Gate {
        started: Arc<Notify>,
        release: Arc<Notify>,
    }

    async fn wait_for_release(State(gate): State<Gate>) -> Response<Body> {
        gate.started.notify_one();
        gate.release.notified().await;
        Response::new(Body::from(BUNDLED_REGISTRY))
    }

    #[rstest]
    #[tokio::test]
    async fn shutdown_during_fetch_keeps_the_bundled_snapshot() {
        let gate = Gate {
            started: Arc::new(Notify::new()),
            release: Arc::new(Notify::new()),
        };
        let router = Router::new()
            .route("/", get(wait_for_release))
            .with_state(gate.clone());
        let (url, server) = serve(router).await;
        let (writer, reader) = crate::bundled(
            &crate::CostMapSettings::default(),
            OffsetDateTime::now_utc(),
        );
        let revision = reader.current().installed.revision.clone();
        let shutdown = CancellationToken::new();
        let task = tokio::spawn(revalidate(writer, client(), url, Some(1), shutdown.clone()));
        timeout(Duration::from_secs(5), gate.started.notified())
            .await
            .unwrap();
        shutdown.cancel();
        timeout(Duration::from_secs(1), task)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(reader.current().installed.revision, revision);
        gate.release.notify_one();
        server.abort();
    }

    #[rstest]
    #[tokio::test]
    async fn sleep_wait_can_be_cancelled_after_it_is_polled() {
        let shutdown = CancellationToken::new();
        let sleep = sleep_or_cancel(Duration::from_secs(30), &shutdown);
        tokio::pin!(sleep);
        assert!(futures_util::poll!(sleep.as_mut()).is_pending());
        shutdown.cancel();
        assert!(!sleep.await);
    }
}
