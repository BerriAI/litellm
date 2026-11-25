//! The JWKS and OIDC discovery documents JWT verification fetches, and the cache that keeps each
//! one for its own TTL. It is kept apart from `AuthStore`'s key cache, so a JWT document
//! never shares keyspace or eviction with a virtual key, and concurrent fetches of one URL share a
//! single request.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use litellm_http::{HttpClientConfig, Verify};
use moka::{Expiry, future::Cache};
use serde_json::Value;
use tokio::sync::{Mutex, OnceCell};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_CACHED_DOCUMENTS: u64 = 1024;

#[derive(Clone)]
struct CachedDocument {
    value: Value,
    ttl: Duration,
}

struct DocumentTtl;

impl Expiry<String, CachedDocument> for DocumentTtl {
    fn expire_after_create(
        &self,
        _: &String,
        document: &CachedDocument,
        _: Instant,
    ) -> Option<Duration> {
        Some(document.ttl)
    }

    fn expire_after_update(
        &self,
        _: &String,
        document: &CachedDocument,
        _: Instant,
        _: Option<Duration>,
    ) -> Option<Duration> {
        Some(document.ttl)
    }
}

/// An answer the identity provider gave that cannot be used: a status that is not a success, or a
/// body that is not JSON. Python raises on both, so the request goes to Python rather than falling
/// back to signing keys cached before, which only an outage may do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unusable;

pub type Fetched = Result<Option<Value>, Unusable>;

/// The client config JWT documents are fetched over. Python fetches them with a plain `httpx`
/// client, which verifies certificates whatever `ssl_verify` says, so a provider opt-out never
/// reaches the signing keys.
pub fn client_config(config: &HttpClientConfig) -> HttpClientConfig {
    let verify = match &config.verify {
        Verify::Disabled => Verify::BuiltInRoots,
        verify => verify.clone(),
    };
    HttpClientConfig {
        verify,
        ..config.clone()
    }
}

pub struct JwtIo {
    client: litellm_http::Client,
    documents: Cache<String, CachedDocument>,
    in_flight: Mutex<HashMap<String, Arc<OnceCell<Fetched>>>>,
}

impl JwtIo {
    pub fn new(client: litellm_http::Client) -> Self {
        Self {
            client,
            documents: Cache::builder()
                .max_capacity(MAX_CACHED_DOCUMENTS)
                .expire_after(DocumentTtl)
                .build(),
            in_flight: Mutex::new(HashMap::new()),
        }
    }

    pub async fn fetch_json(&self, url: &str) -> Fetched {
        let cell = {
            let mut in_flight = self.in_flight.lock().await;
            Arc::clone(
                in_flight
                    .entry(url.to_owned())
                    .or_insert_with(|| Arc::new(OnceCell::new())),
            )
        };

        let result = cell.get_or_init(|| self.get(url)).await.clone();

        let mut in_flight = self.in_flight.lock().await;
        if in_flight
            .get(url)
            .is_some_and(|existing| Arc::ptr_eq(existing, &cell))
        {
            in_flight.remove(url);
        }

        result
    }

    async fn get(&self, url: &str) -> Fetched {
        let request = self.client.get(url).timeout(REQUEST_TIMEOUT);
        let response = match request.send().await {
            Ok(response) => response,
            Err(error) => {
                tracing::debug!(
                    url,
                    error = %litellm_error::Report::new(&error),
                    "a JWT document was not fetched"
                );
                return Ok(None);
            }
        };
        let status = response.status();
        if !status.is_success() {
            tracing::debug!(url, %status, "a JWT document fetch was answered with a failure");
            return Err(Unusable);
        }
        response.json::<Value>().await.map(Some).map_err(|error| {
            tracing::debug!(
                url,
                error = %litellm_error::Report::new(&error),
                "a JWT document was not JSON"
            );
            Unusable
        })
    }

    pub async fn cached_document(&self, key: &str) -> Option<Value> {
        self.documents.get(key).await.map(|document| document.value)
    }

    pub async fn cache_document(&self, key: &str, value: &Value, ttl: Duration) {
        if ttl.is_zero() {
            self.documents.invalidate(key).await;
            return;
        }
        let document = CachedDocument {
            value: value.clone(),
            ttl,
        };
        self.documents.insert(key.to_owned(), document).await;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::{Json, Router, extract::State, http::StatusCode, routing::get};
    use rstest::rstest;
    use serde_json::json;
    use tokio::net::TcpListener;

    use super::*;

    async fn serve(app: Router) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        litellm_call_scope::spawn_detached(async move {
            let _ = axum::serve(listener, app).await;
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn fetch_json_coalesces_concurrent_requests() {
        let hits = Arc::new(AtomicUsize::new(0));
        let app = Router::new()
            .route(
                "/jwks.json",
                get(|State(hits): State<Arc<AtomicUsize>>| async move {
                    hits.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    Json(json!({"keys": []}))
                }),
            )
            .with_state(hits.clone());
        let base = serve(app).await;
        let url = format!("{base}/jwks.json");
        let io = Arc::new(JwtIo::new(litellm_http::Client::for_test()));

        let (a, b, c) = tokio::join!(
            io.fetch_json(&url),
            io.fetch_json(&url),
            io.fetch_json(&url),
        );

        assert_eq!(a, Ok(Some(json!({"keys": []}))));
        assert_eq!(b, Ok(Some(json!({"keys": []}))));
        assert_eq!(c, Ok(Some(json!({"keys": []}))));
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_later_fetch_asks_the_provider_again() {
        let hits = Arc::new(AtomicUsize::new(0));
        let app = Router::new()
            .route(
                "/jwks.json",
                get(|State(hits): State<Arc<AtomicUsize>>| async move {
                    match hits.fetch_add(1, Ordering::SeqCst) {
                        0 => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({}))),
                        _ => (StatusCode::OK, Json(json!({"keys": []}))),
                    }
                }),
            )
            .with_state(hits.clone());
        let url = format!("{}/jwks.json", serve(app).await);
        let io = JwtIo::new(litellm_http::Client::for_test());

        let answers = (io.fetch_json(&url).await, io.fetch_json(&url).await);

        assert_eq!(
            (answers, hits.load(Ordering::SeqCst)),
            ((Err(Unusable), Ok(Some(json!({"keys": []})))), 2)
        );
    }

    #[rstest]
    #[case::not_found(StatusCode::NOT_FOUND, "{}")]
    #[case::server_error(StatusCode::BAD_GATEWAY, "{}")]
    #[case::not_json(StatusCode::OK, "<html></html>")]
    #[tokio::test]
    async fn an_answer_the_provider_gave_that_is_not_a_document_is_unusable(
        #[case] status: StatusCode,
        #[case] body: &'static str,
    ) {
        let app = Router::new().route("/jwks.json", get(move || async move { (status, body) }));
        let base = serve(app).await;
        let io = JwtIo::new(litellm_http::Client::for_test());

        assert_eq!(
            io.fetch_json(&format!("{base}/jwks.json")).await,
            Err(Unusable)
        );
    }

    #[tokio::test]
    async fn a_provider_that_cannot_be_reached_is_an_outage() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/jwks.json", listener.local_addr().unwrap());
        drop(listener);
        let io = JwtIo::new(litellm_http::Client::for_test());

        assert_eq!(io.fetch_json(&url).await, Ok(None));
    }

    #[rstest]
    #[case::verification_off(Verify::Disabled, Verify::BuiltInRoots)]
    #[case::built_in_roots(Verify::BuiltInRoots, Verify::BuiltInRoots)]
    #[case::own_bundle(Verify::CaBundle("ca.pem".into()), Verify::CaBundle("ca.pem".into()))]
    fn jwt_documents_are_always_fetched_over_verified_tls(
        #[case] provider: Verify,
        #[case] expected: Verify,
    ) {
        let config = HttpClientConfig {
            verify: provider,
            ..litellm_http::Resolution::from(&litellm_http::HttpSettings::default()).config
        };

        assert_eq!(client_config(&config).verify, expected);
    }

    #[tokio::test]
    async fn a_document_lives_for_its_own_ttl_and_a_zero_ttl_forgets_it() {
        let io = JwtIo::new(litellm_http::Client::for_test());
        let doc = json!({"keys": [{"kid": "k1"}]});

        io.cache_document("jwks:1", &doc, Duration::from_secs(60))
            .await;
        io.cache_document("jwks:2", &doc, Duration::from_millis(20))
            .await;
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert_eq!(
            (
                io.cached_document("jwks:1").await,
                io.cached_document("jwks:2").await
            ),
            (Some(doc.clone()), None)
        );

        io.cache_document("jwks:1", &doc, Duration::from_secs(60))
            .await;
        assert_eq!(io.cached_document("jwks:1").await, Some(doc.clone()));

        io.cache_document("jwks:1", &doc, Duration::from_millis(20))
            .await;
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert_eq!(io.cached_document("jwks:1").await, None);

        io.cache_document("jwks:3", &doc, Duration::from_secs(60))
            .await;
        io.cache_document("jwks:3", &doc, Duration::ZERO).await;
        assert_eq!(io.cached_document("jwks:3").await, None);
    }
}
