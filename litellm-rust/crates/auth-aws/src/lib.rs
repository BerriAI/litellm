use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::fmt;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::pin::Pin;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aws_credential_types::Credentials as SdkCredentials;
use aws_credential_types::provider::ProvideCredentials;
use aws_sigv4::http_request::{
    SignableBody, SignableRequest, SigningParams, SigningSettings, sign,
};
use aws_sigv4::sign::v4;
use aws_smithy_runtime_api::client::identity::Identity;
use sha2::{Digest, Sha256};

mod error;

pub use error::Error;

const CREDENTIAL_FETCH_LOCK_STRIPES: usize = 64;

#[derive(Clone)]
pub struct Credentials(SdkCredentials);

impl Credentials {
    pub fn new(
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
        session_token: Option<String>,
        expires_after: Option<SystemTime>,
        provider_name: &'static str,
    ) -> Self {
        Self(SdkCredentials::new(
            access_key_id,
            secret_access_key,
            session_token,
            expires_after,
            provider_name,
        ))
    }

    pub fn access_key_id(&self) -> &str {
        self.0.access_key_id()
    }

    pub fn session_token(&self) -> Option<&str> {
        self.0.session_token()
    }
}

impl fmt::Debug for Credentials {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Credentials([REDACTED])")
    }
}

struct CredentialCache {
    entries: HashMap<CredentialScope, (Credentials, Duration)>,
    expirations: BinaryHeap<Reverse<(Duration, CredentialScope)>>,
    max_entries: usize,
}

impl Default for CredentialCache {
    fn default() -> Self {
        Self::new(200)
    }
}

impl CredentialCache {
    fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            expirations: BinaryHeap::new(),
            max_entries: max_entries.max(1),
        }
    }

    fn get(&mut self, key: &CredentialScope, now: Duration) -> Option<Credentials> {
        let (credentials, expiration) = self.entries.get(key)?;
        if *expiration > now {
            return Some(credentials.clone());
        }
        self.entries.remove(key);
        None
    }

    fn insert(
        &mut self,
        key: CredentialScope,
        credentials: Credentials,
        ttl: Duration,
        now: Duration,
    ) {
        while let Some(Reverse((expiration, key))) = self.expirations.peek().cloned() {
            if self.entries.get(&key).map(|(_, current)| *current) != Some(expiration) {
                self.expirations.pop();
            } else if expiration <= now || self.entries.len() >= self.max_entries {
                self.expirations.pop();
                self.entries.remove(&key);
            } else {
                break;
            }
        }
        let expiration = now + ttl;
        self.entries.insert(key.clone(), (credentials, expiration));
        self.expirations.push(Reverse((expiration, key)));
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CredentialScope([u8; 32]);

impl CredentialScope {
    pub fn from_optional_values<'a>(
        namespace: &str,
        values: impl IntoIterator<Item = Option<&'a str>>,
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(namespace.len().to_le_bytes());
        hasher.update(namespace.as_bytes());
        for value in values {
            match value {
                Some(value) => {
                    hasher.update([1]);
                    hasher.update(value.len().to_le_bytes());
                    hasher.update(value.as_bytes());
                }
                None => hasher.update([0]),
            }
        }
        Self(hasher.finalize().into())
    }
}

impl fmt::Debug for CredentialScope {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("CredentialScope([REDACTED])")
    }
}

pub trait Clock: Send + Sync {
    fn now(&self) -> Duration;
}

#[derive(Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Duration {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
    }
}

pub struct CredentialState<R, C = SystemClock> {
    runtime: R,
    clock: C,
    cache: Mutex<CredentialCache>,
    fetch_locks: Box<[tokio::sync::Mutex<()>]>,
}

impl<R> CredentialState<R, SystemClock> {
    pub fn new(runtime: R, max_entries: usize) -> Self {
        Self::with_clock(runtime, max_entries, SystemClock)
    }
}

impl<R, C> CredentialState<R, C>
where
    C: Clock,
{
    pub fn with_clock(runtime: R, max_entries: usize, clock: C) -> Self {
        Self {
            runtime,
            clock,
            cache: Mutex::new(CredentialCache::new(max_entries)),
            fetch_locks: (0..CREDENTIAL_FETCH_LOCK_STRIPES)
                .map(|_| tokio::sync::Mutex::new(()))
                .collect(),
        }
    }

    pub fn runtime(&self) -> &R {
        &self.runtime
    }

    pub async fn get_or_acquire<E, F, Fut>(
        &self,
        scope: CredentialScope,
        ttl: Duration,
        acquire: F,
    ) -> Result<Credentials, E>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Credentials, E>>,
    {
        let mut stripe_hasher = std::collections::hash_map::DefaultHasher::new();
        scope.hash(&mut stripe_hasher);
        let stripe = stripe_hasher.finish() as usize % self.fetch_locks.len();
        let _fetch_guard = self.fetch_locks[stripe].lock().await;

        if let Some(credentials) = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&scope, self.clock.now())
        {
            return Ok(credentials);
        }

        let credentials = acquire().await?;
        self.cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(scope, credentials.clone(), ttl, self.clock.now());
        Ok(credentials)
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct AssumeRoleRequest {
    pub role: String,
    pub session_name: String,
    pub region: Option<String>,
    pub endpoint: Option<String>,
    pub source_credentials: Option<Credentials>,
    pub external_id: Option<String>,
}

impl fmt::Debug for AssumeRoleRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AssumeRoleRequest")
            .field("role", &self.role)
            .field("session_name", &self.session_name)
            .field("region", &self.region)
            .field("endpoint", &self.endpoint)
            .field("source_credentials", &self.source_credentials.is_some())
            .field("external_id", &self.external_id.is_some())
            .finish()
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct WebIdentityRequest {
    pub token: String,
    pub role: String,
    pub session_name: String,
    pub region: Option<String>,
    pub endpoint: Option<String>,
}

impl fmt::Debug for WebIdentityRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WebIdentityRequest")
            .field("token", &"[REDACTED]")
            .field("role", &self.role)
            .field("session_name", &self.session_name)
            .field("region", &self.region)
            .field("endpoint", &self.endpoint)
            .finish()
    }
}

pub type CredentialFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, Error>> + Send + 'a>>;

pub trait CredentialRuntime: Send + Sync {
    fn profile<'a>(&'a self, name: &'a str) -> CredentialFuture<'a, Credentials>;
    fn ambient(&self) -> CredentialFuture<'_, Credentials>;
    fn assume_role(&self, request: AssumeRoleRequest) -> CredentialFuture<'_, Credentials>;
    fn web_identity(&self, request: WebIdentityRequest) -> CredentialFuture<'_, Credentials>;
    fn caller_identity(
        &self,
        region: Option<String>,
        endpoint: Option<String>,
    ) -> CredentialFuture<'_, Option<String>>;
}

#[derive(Default)]
pub struct NativeCredentialRuntime;

impl CredentialRuntime for NativeCredentialRuntime {
    fn profile<'a>(&'a self, name: &'a str) -> CredentialFuture<'a, Credentials> {
        Box::pin(profile_credentials(name))
    }

    fn ambient(&self) -> CredentialFuture<'_, Credentials> {
        Box::pin(default_credentials())
    }

    fn assume_role(&self, request: AssumeRoleRequest) -> CredentialFuture<'_, Credentials> {
        Box::pin(assume_role_credentials(request))
    }

    fn web_identity(&self, request: WebIdentityRequest) -> CredentialFuture<'_, Credentials> {
        Box::pin(web_identity_credentials(request))
    }

    fn caller_identity(
        &self,
        region: Option<String>,
        endpoint: Option<String>,
    ) -> CredentialFuture<'_, Option<String>> {
        Box::pin(caller_identity(region, endpoint))
    }
}

impl PartialEq for Credentials {
    fn eq(&self, other: &Self) -> bool {
        self.0.access_key_id() == other.0.access_key_id()
            && self.0.secret_access_key() == other.0.secret_access_key()
            && self.0.session_token() == other.0.session_token()
    }
}

impl Eq for Credentials {}

pub fn static_credentials(
    access_key_id: impl Into<String>,
    secret_access_key: impl Into<String>,
) -> Credentials {
    Credentials::new(
        access_key_id,
        secret_access_key,
        None,
        None,
        "litellm-static",
    )
}

pub fn session_credentials(
    access_key_id: impl Into<String>,
    secret_access_key: impl Into<String>,
    session_token: impl Into<String>,
    provider_name: &'static str,
) -> Credentials {
    Credentials::new(
        access_key_id,
        secret_access_key,
        Some(session_token.into()),
        None,
        provider_name,
    )
}

pub async fn profile_credentials(name: &str) -> Result<Credentials, Error> {
    let provider = aws_config::profile::ProfileFileCredentialsProvider::builder()
        .profile_name(name)
        .build();
    provider
        .provide_credentials()
        .await
        .map(Credentials)
        .map_err(|error| Error::new(format!("AWS profile credentials failed: {error}")))
}

pub async fn default_credentials() -> Result<Credentials, Error> {
    let provider = aws_config::default_provider::credentials::DefaultCredentialsChain::builder()
        .build()
        .await;
    provider
        .provide_credentials()
        .await
        .map(Credentials)
        .map_err(|error| Error::new(format!("AWS default credentials failed: {error}")))
}

fn sdk_loader(region: Option<String>, endpoint: Option<String>) -> aws_config::ConfigLoader {
    let loader = aws_config::defaults(aws_config::BehaviorVersion::latest());
    let loader = match region {
        Some(region) => loader.region(aws_types::region::Region::new(region)),
        None => loader,
    };
    match endpoint {
        Some(endpoint) => loader.endpoint_url(endpoint),
        None => loader,
    }
}

pub async fn assume_role_credentials(request: AssumeRoleRequest) -> Result<Credentials, Error> {
    let mut loader = sdk_loader(request.region, request.endpoint);
    if let Some(credentials) = request.source_credentials {
        loader = loader.credentials_provider(credentials.0);
    }
    let sdk_config = loader.load().await;
    let builder = aws_config::sts::AssumeRoleProvider::builder(request.role)
        .session_name(request.session_name);
    let builder = match request.external_id {
        Some(id) => builder.external_id(id),
        None => builder,
    };
    builder
        .configure(&sdk_config)
        .build()
        .await
        .provide_credentials()
        .await
        .map(Credentials)
        .map_err(|error| Error::new(format!("AWS role credentials failed: {error}")))
}

pub async fn web_identity_credentials(request: WebIdentityRequest) -> Result<Credentials, Error> {
    let sdk_config = sdk_loader(request.region, request.endpoint).load().await;
    let response = aws_sdk_sts::Client::new(&sdk_config)
        .assume_role_with_web_identity()
        .role_arn(request.role)
        .role_session_name(request.session_name)
        .web_identity_token(request.token)
        .send()
        .await
        .map_err(|error| Error::new(format!("AWS web identity credentials failed: {error}")))?;
    let credentials = response
        .credentials()
        .ok_or_else(|| Error::new("AWS web identity response had no credentials"))?;
    let expiration = SystemTime::try_from(*credentials.expiration())
        .map_err(|error| Error::new(format!("AWS web identity expiration was invalid: {error}")))?;
    Ok(Credentials::new(
        credentials.access_key_id(),
        credentials.secret_access_key(),
        Some(credentials.session_token().to_string()),
        Some(expiration),
        "litellm-web-identity",
    ))
}

pub async fn caller_identity(
    region: Option<String>,
    endpoint: Option<String>,
) -> Result<Option<String>, Error> {
    let sdk_config = sdk_loader(region, endpoint).load().await;
    match aws_sdk_sts::Client::new(&sdk_config)
        .get_caller_identity()
        .send()
        .await
    {
        Ok(response) => Ok(response.arn().map(str::to_string)),
        Err(_) => Ok(None),
    }
}

pub fn role_identity(arn: &str) -> Option<(&str, &str, &str)> {
    let mut parts = arn.splitn(6, ':');
    let ("arn", partition, _, _, account, resource) = (
        parts.next()?,
        parts.next()?,
        parts.next()?,
        parts.next()?,
        parts.next()?,
        parts.next()?,
    ) else {
        return None;
    };
    let role = if let Some(role) = resource.strip_prefix("role/") {
        role.rsplit('/').next()?
    } else {
        resource.strip_prefix("assumed-role/")?.split('/').next()?
    };
    Some((partition, account, role))
}

pub fn same_role_arns(target: &str, caller: &str) -> bool {
    role_identity(target) == role_identity(caller)
}

pub struct SigV4Request<'a> {
    pub method: &'a str,
    pub uri: &'a str,
    pub body: &'a [u8],
    pub headers: &'a BTreeMap<String, String>,
    pub region: &'a str,
    pub service: &'a str,
    pub signing_time: SystemTime,
}

pub fn sign_v4(
    request: SigV4Request<'_>,
    credentials: &Credentials,
) -> Result<BTreeMap<String, String>, Error> {
    let identity: Identity = credentials.0.clone().into();
    let params = v4::SigningParams::builder()
        .identity(&identity)
        .region(request.region)
        .name(request.service)
        .time(request.signing_time)
        .settings(SigningSettings::default())
        .build()
        .map(SigningParams::from)
        .map_err(|error| Error::new(format!("AWS signing parameters failed: {error}")))?;
    let header_refs = request
        .headers
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_str()));
    let signable = SignableRequest::new(
        request.method,
        request.uri,
        header_refs,
        SignableBody::Bytes(request.body),
    )
    .map_err(|error| Error::new(format!("AWS signable request failed: {error}")))?;
    let (instructions, _) = sign(signable, &params)
        .map_err(|error| Error::new(format!("AWS request signing failed: {error}")))?
        .into_parts();
    Ok(instructions
        .headers()
        .map(|(name, value)| {
            let normalized_name = match name {
                "authorization" => "Authorization",
                "x-amz-date" => "X-Amz-Date",
                "x-amz-security-token" => "X-Amz-Security-Token",
                _ => name,
            };
            (normalized_name.to_string(), value.to_string())
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    struct FixtureRuntime {
        effects: Arc<Mutex<Vec<&'static str>>>,
    }

    impl CredentialRuntime for FixtureRuntime {
        fn profile<'a>(&'a self, _name: &'a str) -> CredentialFuture<'a, Credentials> {
            self.effects.lock().unwrap().push("profile");
            Box::pin(std::future::ready(Ok(static_credentials(
                "profile", "secret",
            ))))
        }

        fn ambient(&self) -> CredentialFuture<'_, Credentials> {
            self.effects.lock().unwrap().push("ambient");
            Box::pin(std::future::ready(Ok(static_credentials(
                "ambient", "secret",
            ))))
        }

        fn assume_role(&self, _request: AssumeRoleRequest) -> CredentialFuture<'_, Credentials> {
            self.effects.lock().unwrap().push("assume-role");
            Box::pin(std::future::ready(Ok(static_credentials("role", "secret"))))
        }

        fn web_identity(&self, _request: WebIdentityRequest) -> CredentialFuture<'_, Credentials> {
            self.effects.lock().unwrap().push("web-identity");
            Box::pin(std::future::ready(Ok(static_credentials("web", "secret"))))
        }

        fn caller_identity(
            &self,
            _region: Option<String>,
            _endpoint: Option<String>,
        ) -> CredentialFuture<'_, Option<String>> {
            self.effects.lock().unwrap().push("caller-identity");
            Box::pin(std::future::ready(Ok(None)))
        }
    }

    fn signing_request<'a>(
        uri: &'a str,
        body: &'a [u8],
        headers: &'a BTreeMap<String, String>,
        service: &'a str,
    ) -> SigV4Request<'a> {
        SigV4Request {
            method: "POST",
            uri,
            body,
            headers,
            region: "us-east-1",
            service,
            signing_time: SystemTime::UNIX_EPOCH + Duration::from_secs(1_704_164_645),
        }
    }

    #[test]
    fn cache_expiry_and_bounds_are_clock_driven() {
        let mut cache = CredentialCache::new(1);
        let first = CredentialScope::from_optional_values("test", [Some("first")]);
        let second = CredentialScope::from_optional_values("test", [Some("second")]);
        cache.insert(
            first.clone(),
            static_credentials("ak1", "sk1"),
            Duration::from_secs(10),
            Duration::ZERO,
        );
        assert_eq!(
            cache
                .get(&first, Duration::from_secs(9))
                .unwrap()
                .access_key_id(),
            "ak1"
        );
        cache.insert(
            second.clone(),
            static_credentials("ak2", "sk2"),
            Duration::from_secs(10),
            Duration::ZERO,
        );
        assert!(cache.get(&first, Duration::ZERO).is_none());
        assert!(cache.get(&second, Duration::from_secs(11)).is_none());
    }

    #[tokio::test]
    async fn credential_state_coordinates_concurrent_misses() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let state = CredentialState::new((), 1);
        let acquisitions = AtomicUsize::new(0);
        let scope = CredentialScope::from_optional_values("test", [Some("identity")]);
        let acquire = || async {
            acquisitions.fetch_add(1, Ordering::SeqCst);
            tokio::task::yield_now().await;
            Ok::<_, ()>(static_credentials("ak", "sk"))
        };
        let (first, second) = tokio::join!(
            state.get_or_acquire(scope.clone(), Duration::from_secs(10), acquire),
            state.get_or_acquire(scope, Duration::from_secs(10), acquire),
        );

        assert_eq!(first.unwrap(), second.unwrap());
        assert_eq!(acquisitions.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn credential_state_reacquires_at_the_injected_expiry_boundary() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

        struct TestClock(Arc<AtomicU64>);

        impl Clock for TestClock {
            fn now(&self) -> Duration {
                Duration::from_secs(self.0.load(Ordering::SeqCst))
            }
        }

        let now = Arc::new(AtomicU64::new(0));
        let state = CredentialState::with_clock((), 1, TestClock(now.clone()));
        let acquisitions = AtomicUsize::new(0);
        let scope = CredentialScope::from_optional_values("test", [Some("identity")]);
        let acquire = || async {
            acquisitions.fetch_add(1, Ordering::SeqCst);
            Ok::<_, ()>(static_credentials("ak", "sk"))
        };

        state
            .get_or_acquire(scope.clone(), Duration::from_secs(10), acquire)
            .await
            .unwrap();
        now.store(9, Ordering::SeqCst);
        state
            .get_or_acquire(scope.clone(), Duration::from_secs(10), acquire)
            .await
            .unwrap();
        now.store(10, Ordering::SeqCst);
        state
            .get_or_acquire(scope, Duration::from_secs(10), acquire)
            .await
            .unwrap();

        assert_eq!(acquisitions.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn credential_scope_does_not_expose_key_material() {
        let scope = CredentialScope::from_optional_values(
            "test",
            [Some("visible-id"), Some("never-print-secret"), None],
        );
        let debug = format!("{scope:?}");
        assert!(!debug.contains("visible-id"));
        assert!(!debug.contains("never-print-secret"));
    }

    #[test]
    fn signing_matches_the_bedrock_golden_vector() {
        let uri = "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.titan-text-express-v1/invoke";
        let body = br#"{"input":"hello"}"#;
        let headers = BTreeMap::from([("Content-Type".into(), "application/json".into())]);
        let credentials = Credentials::new(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            Some("session-token".into()),
            None,
            "test",
        );
        let signed = sign_v4(
            signing_request(uri, body, &headers, "bedrock"),
            &credentials,
        )
        .expect("signature");
        assert_eq!(
            signed.get("X-Amz-Date").map(String::as_str),
            Some("20240102T030405Z")
        );
        assert_eq!(
            signed.get("Authorization").map(String::as_str),
            Some(
                "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20240102/us-east-1/bedrock/aws4_request, SignedHeaders=content-type;host;x-amz-date;x-amz-security-token, Signature=55c027ef47527d3ad63f1735f9d099efdbc99f296ff914bd94e727e24ec0e464"
            )
        );
    }

    #[test]
    fn signer_is_generic_over_method_and_service() {
        let uri = "https://sts.amazonaws.com/?Action=GetCallerIdentity&Version=2011-06-15";
        let headers = BTreeMap::new();
        let credentials = static_credentials("AKIDEXAMPLE", "secret");
        let request = SigV4Request {
            method: "GET",
            uri,
            body: b"",
            headers: &headers,
            region: "us-east-1",
            service: "sts",
            signing_time: SystemTime::UNIX_EPOCH,
        };
        let signed = sign_v4(request, &credentials).expect("signature");
        assert!(signed["Authorization"].contains("/sts/aws4_request"));
    }

    #[test]
    fn credentials_are_redacted() {
        let credentials = static_credentials("visible-id", "never-print-secret");
        let debug = format!("{credentials:?}");
        assert!(!debug.contains("visible-id"));
        assert!(!debug.contains("never-print-secret"));
    }

    #[test]
    fn mechanism_inputs_are_redacted() {
        let assume_role = AssumeRoleRequest {
            role: "role".into(),
            session_name: "session".into(),
            region: None,
            endpoint: None,
            source_credentials: Some(static_credentials("visible-id", "secret")),
            external_id: Some("external-secret".into()),
        };
        let web_identity = WebIdentityRequest {
            token: "identity-secret".into(),
            role: "role".into(),
            session_name: "session".into(),
            region: None,
            endpoint: None,
        };
        let debug = format!("{assume_role:?} {web_identity:?}");
        for secret in ["visible-id", "secret", "external-secret", "identity-secret"] {
            assert!(!debug.contains(secret));
        }
    }

    #[test]
    fn role_matching_is_partition_account_and_role_aware() {
        assert!(same_role_arns(
            "arn:aws:iam::123456789012:role/path/demo",
            "arn:aws:sts::123456789012:assumed-role/demo/session"
        ));
        assert!(!same_role_arns(
            "arn:aws:iam::123456789012:role/demo",
            "arn:aws-cn:iam::123456789012:role/demo"
        ));
    }

    #[tokio::test]
    async fn credential_io_is_injectable_for_policy_consumers() {
        let effects = Arc::new(Mutex::new(Vec::new()));
        let runtime = FixtureRuntime {
            effects: effects.clone(),
        };

        assert_eq!(
            runtime.profile("demo").await.unwrap().access_key_id(),
            "profile"
        );
        assert_eq!(runtime.ambient().await.unwrap().access_key_id(), "ambient");
        assert_eq!(
            runtime
                .caller_identity(Some("us-east-1".into()), None)
                .await
                .unwrap(),
            None
        );
        assert_eq!(
            effects.lock().unwrap().as_slice(),
            &["profile", "ambient", "caller-identity"]
        );
    }
}
