use crate::Error;
use http::HeaderMap;
use litellm_http::Client;
use litellm_traces::Tenant;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use subtle::ConstantTimeEq;

pub const SNAPSHOT_TTL: Duration = Duration::from_secs(90);
const MAX_KEYS: usize = 10_000;
const MAX_SNAPSHOT_BYTES: usize = 8 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Credential {
    pub token_hash: String,
    pub tenant: Tenant,
    pub expires_at: Option<u64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    pub issued_at: u64,
    pub keys: Vec<Credential>,
}

struct ActiveSnapshot {
    received: Instant,
    issued_at: u64,
    expires_at: u64,
    keys: HashMap<String, Credential>,
}

#[derive(Default)]
pub struct Credentials(RwLock<Option<ActiveSnapshot>>);

pub fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn bearer(headers: &HeaderMap) -> Result<&str, Error> {
    let value = headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .ok_or(Error::Unauthorized)?;
    let (scheme, token) = value.split_once(' ').ok_or(Error::Unauthorized)?;
    if !scheme.eq_ignore_ascii_case("bearer") || token.is_empty() || token.len() > 512 {
        return Err(Error::Unauthorized);
    }
    Ok(token)
}

pub fn authorize_service(headers: &HeaderMap, expected: &str) -> Result<(), Error> {
    let supplied = Sha256::digest(bearer(headers)?.as_bytes());
    let expected = Sha256::digest(expected.as_bytes());
    if bool::from(supplied.ct_eq(&expected)) {
        Ok(())
    } else {
        Err(Error::Unauthorized)
    }
}

impl Credentials {
    pub fn replace(&self, snapshot: Snapshot) -> Result<(), Error> {
        let now = unix_seconds();
        if snapshot.keys.len() > MAX_KEYS
            || snapshot.issued_at > now.saturating_add(5)
            || snapshot.issued_at.saturating_add(SNAPSHOT_TTL.as_secs()) <= now
        {
            return Err(Error::Unavailable);
        }
        if snapshot.keys.iter().any(|key| {
            key.token_hash.len() != 64 || !key.token_hash.bytes().all(|b| b.is_ascii_hexdigit())
        }) {
            return Err(Error::Unavailable);
        }
        let count = snapshot.keys.len();
        let keys: HashMap<_, _> = snapshot
            .keys
            .into_iter()
            .map(|key| (key.token_hash.clone(), key))
            .collect();
        if keys.len() != count {
            return Err(Error::Unavailable);
        }
        let mut current = self.0.write().map_err(|_| Error::Unavailable)?;
        if current
            .as_ref()
            .is_some_and(|active| active.issued_at > snapshot.issued_at)
        {
            return Err(Error::Unavailable);
        }
        *current = Some(ActiveSnapshot {
            received: Instant::now(),
            issued_at: snapshot.issued_at,
            expires_at: snapshot.issued_at + SNAPSHOT_TTL.as_secs(),
            keys,
        });
        Ok(())
    }

    pub fn clear(&self) {
        if let Ok(mut snapshot) = self.0.write() {
            *snapshot = None;
        }
    }

    pub fn ready(&self) -> bool {
        self.0.read().ok().is_some_and(|snapshot| {
            snapshot.as_ref().is_some_and(|snapshot| {
                snapshot.received.elapsed() < SNAPSHOT_TTL && snapshot.expires_at > unix_seconds()
            })
        })
    }

    pub fn tenant(&self, headers: &HeaderMap) -> Result<Tenant, Error> {
        let token = bearer(headers)?;
        let hash = format!("{:x}", Sha256::digest(token.as_bytes()));
        let guard = self.0.read().map_err(|_| Error::Unavailable)?;
        let snapshot = guard.as_ref().ok_or(Error::Unavailable)?;
        let now = unix_seconds();
        if snapshot.received.elapsed() >= SNAPSHOT_TTL || snapshot.expires_at <= now {
            return Err(Error::Unavailable);
        }
        let pending = token
            .strip_prefix("lens-trace-")
            .and_then(|value| value.split_once('-'))
            .and_then(|(issued, _)| issued.parse::<u64>().ok())
            .is_some_and(|issued| issued >= snapshot.issued_at && issued <= now.saturating_add(5));
        let key = snapshot.keys.get(&hash).ok_or(if pending {
            Error::CredentialsPending
        } else {
            Error::Unauthorized
        })?;
        if key.expires_at.is_some_and(|expiry| expiry <= now) {
            return Err(Error::Unauthorized);
        }
        Ok(key.tenant.clone())
    }
}

pub async fn refresh(
    credentials: &Credentials,
    client: &Client,
    url: &url::Url,
    token: &str,
) -> Result<(), Error> {
    let mut response = client
        .get(url.clone())
        .bearer_auth(token)
        .timeout(Duration::from_secs(5))
        .send()
        .await?;
    if response.status() == http::StatusCode::UNAUTHORIZED
        || response.status() == http::StatusCode::FORBIDDEN
    {
        credentials.clear();
        return Err(Error::Unauthorized);
    }
    if !response.status().is_success() {
        return Err(Error::Unavailable);
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if body.len() + chunk.len() > MAX_SNAPSHOT_BYTES {
            return Err(Error::TooLarge);
        }
        body.extend_from_slice(&chunk);
    }
    credentials.replace(serde_json::from_slice(&body).map_err(|_| Error::Unavailable)?)
}

pub async fn refresh_loop(
    credentials: Arc<Credentials>,
    client: Client,
    url: url::Url,
    token: String,
) {
    loop {
        if refresh(&credentials, &client, &url, &token).await.is_err() {
            tracing::warn!("Lens ingestion credential refresh failed");
        }
        tokio::time::sleep(Duration::from_secs(30)).await;
    }
}
