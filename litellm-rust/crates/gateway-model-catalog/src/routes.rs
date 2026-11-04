use std::sync::Arc;
use std::time::SystemTime;

use axum::{
    Json, Router,
    extract::State,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use chrono::{DateTime, Utc};
use litellm_model_catalog::Origin;
use serde::Serialize;

use crate::{Error, Installed, ModelCatalog};

/// Mounted behind the gateway's admin authentication, as the Python routes are
pub fn router(catalog: Arc<ModelCatalog>) -> Router {
    Router::new()
        .route("/reload/model_cost_map", post(reload))
        .route("/model/cost_map/source", get(source))
        .with_state(catalog)
}

#[derive(Serialize)]
enum Source {
    #[serde(rename = "local")]
    Local,
    #[serde(rename = "remote")]
    Remote,
}

#[derive(Serialize)]
struct SourceInfo<'a> {
    source: Source,
    url: Option<&'a str>,
    is_env_forced: bool,
    fallback_reason: Option<&'a str>,
    loaded_at: String,
    source_revision: Option<&'a str>,
    etag: Option<&'a str>,
    model_count: usize,
}

#[derive(Serialize)]
struct Reloaded<'a> {
    message: String,
    status: &'static str,
    models_count: usize,
    timestamp: String,
    source_revision: Option<&'a str>,
    etag: Option<&'a str>,
}

async fn source(State(catalog): State<Arc<ModelCatalog>>) -> Response {
    let installed = catalog.installed();
    let provenance = &installed.provenance;
    let (source, url) = match &provenance.origin {
        Origin::Bundled { attempted } => (Source::Local, attempted.as_deref()),
        Origin::Remote { url } => (Source::Remote, Some(url.as_str())),
    };
    Json(SourceInfo {
        source,
        url,
        is_env_forced: provenance.is_env_forced,
        fallback_reason: provenance.fallback_reason.as_deref(),
        loaded_at: isoformat(provenance.loaded_at),
        source_revision: provenance.revision.as_deref(),
        etag: provenance.etag.as_deref(),
        model_count: model_count(&installed),
    })
    .into_response()
}

async fn reload(State(catalog): State<Arc<ModelCatalog>>) -> Result<Response, Error> {
    let installed = catalog.reload().await.map_err(Error::Reload)?;
    let models_count = model_count(&installed);
    tracing::info!(models_count, "model cost map reloaded");
    Ok(Json(Reloaded {
        message: format!("Price data reloaded successfully! {models_count} models updated."),
        status: "success",
        models_count,
        timestamp: isoformat(installed.provenance.loaded_at),
        source_revision: installed.provenance.revision.as_deref(),
        etag: installed.provenance.etag.as_deref(),
    })
    .into_response())
}

/// Every name a model is served under, aliases included, as `len(litellm.model_cost)` counts them
fn model_count(installed: &Installed) -> usize {
    installed.catalog.served().count()
}

/// Python's `datetime.isoformat()` for an aware UTC time, which leaves out a zero fraction
fn isoformat(time: SystemTime) -> String {
    let utc = DateTime::<Utc>::from(time);
    let format = match utc.timestamp_subsec_micros() {
        0 => "%Y-%m-%dT%H:%M:%S+00:00",
        _ => "%Y-%m-%dT%H:%M:%S%.6f+00:00",
    };
    utc.format(format).to_string()
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, UNIX_EPOCH};

    use axum::body::{Body, to_bytes};
    use axum::http::{Request, StatusCode};
    use litellm_model_catalog::revision;
    use rstest::rstest;
    use serde_json::{Value, json};
    use tokio_util::sync::CancellationToken;
    use tower::ServiceExt;

    use super::*;
    use crate::test_support::{catalog, document, file};

    async fn call(catalog: &Arc<ModelCatalog>, method: &str, path: &str) -> (StatusCode, Value) {
        let request = Request::builder()
            .method(method)
            .uri(path)
            .body(Body::empty())
            .unwrap();
        let response = router(Arc::clone(catalog)).oneshot(request).await.unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, serde_json::from_slice(&body).unwrap())
    }

    #[rstest]
    #[tokio::test]
    async fn the_source_reports_the_bundled_map_then_the_adopted_remote_one() {
        let body = document(&[("gpt-test", 0.25)]);
        let (_directory, url) = file(&body);
        let catalog = Arc::new(catalog(Some(url.clone()), CancellationToken::new()));
        let (status, before) = call(&catalog, "GET", "/model/cost_map/source").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(before["source"], "local");
        assert_eq!(before["url"], json!(url));
        assert_eq!(before["is_env_forced"], false);
        assert_eq!(before["etag"], Value::Null);
        assert_eq!(
            before["model_count"],
            json!(catalog.installed().catalog.served().count())
        );
        catalog.revalidate().await;
        let (_, after) = call(&catalog, "GET", "/model/cost_map/source").await;
        assert_eq!(
            after,
            json!({
                "source": "remote",
                "url": url,
                "is_env_forced": false,
                "fallback_reason": null,
                "loaded_at": isoformat(catalog.installed().provenance.loaded_at),
                "source_revision": revision(&body),
                "etag": null,
                "model_count": 1,
            })
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_reload_answers_with_the_new_map_s_count_and_revision() {
        let body = br#"{"gpt-test": {"input_cost_per_token": 1, "aliases": ["gpt-test-latest"]}}"#;
        let (_directory, url) = file(body);
        let catalog = Arc::new(catalog(Some(url), CancellationToken::new()));
        let (status, reloaded) = call(&catalog, "POST", "/reload/model_cost_map").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            reloaded,
            json!({
                "message": "Price data reloaded successfully! 2 models updated.",
                "status": "success",
                "models_count": 2,
                "timestamp": isoformat(catalog.installed().provenance.loaded_at),
                "source_revision": revision(body),
                "etag": null,
            })
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_failed_reload_is_a_502_with_python_s_detail() {
        let catalog = Arc::new(catalog(
            Some("ftp://example.com/map.json".to_owned()),
            CancellationToken::new(),
        ));
        let (status, failed) = call(&catalog, "POST", "/reload/model_cost_map").await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        let detail = failed["detail"].as_str().unwrap();
        assert!(
            detail.starts_with(
                "Failed to reload model cost map: could not fetch ftp://example.com/map.json: "
            ),
            "{detail}"
        );
        let (_, source) = call(&catalog, "GET", "/model/cost_map/source").await;
        assert_eq!(source["source"], "local");
        assert_eq!(source["fallback_reason"], Value::Null);
    }

    /// `datetime.fromtimestamp(seconds, timezone.utc).isoformat()` in CPython
    #[rstest]
    #[case::fraction(
        Duration::from_micros(1_793_062_923_004_500),
        "2026-10-27T01:02:03.004500+00:00"
    )]
    #[case::whole_second(Duration::from_secs(1_793_062_923), "2026-10-27T01:02:03+00:00")]
    #[case::sub_microsecond_is_dropped(
        Duration::from_nanos(1_793_062_923_000_000_999),
        "2026-10-27T01:02:03+00:00"
    )]
    fn timestamps_format_like_python_isoformat(
        #[case] since_epoch: Duration,
        #[case] expected: &str,
    ) {
        assert_eq!(isoformat(UNIX_EPOCH + since_epoch), expected);
    }
}
