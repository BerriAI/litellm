//! An identity provider as a gateway test holds it: one Ed25519 key, the JWKS that publishes it
//! over HTTP, and tokens it signs.

// Shared across test targets; each target uses a different subset.
#![allow(dead_code)]

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use axum::http::StatusCode;
use serde_json::{Map, Value, json};

pub const KID: &str = "k1";

const PRIVATE_PEM: &str = "-----BEGIN PRIVATE KEY-----
MC4CAQAwBQYDK2VwBCIEIEjkuYWkj6THWjxVRFL36QT1XS2a1KjdV89JMvLev7r2
-----END PRIVATE KEY-----
";
const PUBLIC_X: &str = "iO5qGwVLMzVNKsLll1PGbMPQzI_aTO9tZuoe8AEWCo4";

/// `claims` signed with the provider's key, as a compact JWS.
pub fn sign(claims: &Value) -> String {
    let header = jsonwebtoken::Header {
        kid: Some(KID.to_owned()),
        ..jsonwebtoken::Header::new(jsonwebtoken::Algorithm::EdDSA)
    };
    let key = jsonwebtoken::EncodingKey::from_ed_pem(PRIVATE_PEM.as_bytes()).unwrap();
    jsonwebtoken::encode(&header, claims, &key).unwrap()
}

/// `claims` with `iat` now and `exp` `expires_in` seconds from now, signed.
pub fn issue(claims: &Map<String, Value>, expires_in: i64) -> String {
    let now = jiff::Timestamp::now().as_second();
    let stamped = claims
        .clone()
        .into_iter()
        .chain([
            ("iat".to_owned(), json!(now)),
            ("exp".to_owned(), json!(now + expires_in)),
        ])
        .collect::<Map<_, _>>();
    sign(&Value::Object(stamped))
}

/// The provider's JWKS endpoint, and how many times it was fetched.
pub struct Jwks {
    /// What `JWT_PUBLIC_KEY_URL` names.
    pub url: String,
    fetches: Arc<AtomicUsize>,
}

impl Jwks {
    pub fn fetches(&self) -> usize {
        self.fetches.load(Ordering::SeqCst)
    }
}

/// Serves the provider's JWKS.
pub async fn serve_jwks() -> Jwks {
    answering(StatusCode::OK).await
}

/// A provider that answers 404 where its JWKS should be.
pub async fn serve_no_jwks() -> Jwks {
    answering(StatusCode::NOT_FOUND).await
}

/// A JWKS URL nothing listens on.
pub async fn unreachable_jwks() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    format!("http://{}/jwks.json", listener.local_addr().unwrap())
}

async fn answering(status: StatusCode) -> Jwks {
    let jwks = json!({"keys": [{"kty": "OKP", "kid": KID, "crv": "Ed25519", "x": PUBLIC_X}]});
    let fetches = Arc::new(AtomicUsize::new(0));
    let counted = fetches.clone();
    let app = axum::Router::new().route(
        "/jwks.json",
        axum::routing::get(move || {
            counted.fetch_add(1, Ordering::SeqCst);
            let jwks = jwks.clone();
            async move { (status, axum::Json(jwks)) }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    litellm_call_scope::spawn_detached(async move {
        let _ = axum::serve(listener, app).await;
    });
    Jwks {
        url: format!("http://{addr}/jwks.json"),
        fetches,
    }
}
