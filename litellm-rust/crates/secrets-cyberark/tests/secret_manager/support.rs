use super::*;

pub(super) const TOKEN_JSON: &str = r#"{"protected":"p","payload":"q","signature":"s"}"#;

#[derive(Deserialize)]
pub(super) struct ParityFixture {
    pub(super) account: String,
    pub(super) username: String,
    pub(super) api_key: String,
    pub(super) authenticate_path: String,
    pub(super) token_json: String,
    pub(super) authorization_header: String,
    pub(super) policy_path: String,
    pub(super) secrets: Vec<ParitySecret>,
}

#[derive(Deserialize)]
pub(super) struct ParitySecret {
    pub(super) name: String,
    pub(super) path: String,
    pub(super) policy_body: String,
}

#[derive(Debug)]
pub(super) struct RawPath(pub(super) String);

impl Match for RawPath {
    fn matches(&self, request: &Request) -> bool {
        request.url.path() == self.0
    }
}

#[fixture]
pub(super) fn parity_fixture() -> ParityFixture {
    serde_json::from_str(include_str!("../fixtures/parity.json")).unwrap()
}

#[fixture]
pub(super) fn client_identity_directory() -> tempfile::TempDir {
    let identity = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("client.crt"), identity.cert.pem()).unwrap();
    std::fs::write(
        directory.path().join("client.key"),
        identity.signing_key.serialize_pem(),
    )
    .unwrap();
    directory
}

pub(super) fn from_environment(
    environment: Arc<dyn Lookup + Send + Sync>,
    enterprise_enabled: bool,
) -> Result<CyberArkSecretManager, Error> {
    CyberArkSecretManager::new(
        &HttpClientPool::new(Arc::new(PublicDnsResolver)),
        &Resolution::from(&HttpSettings::default()).config,
        environment,
        enterprise_enabled,
    )
}

pub(super) fn manager(server: &MockServer, ttl: Duration) -> CyberArkSecretManager {
    CyberArkSecretManager::with_client(
        litellm_http::Client::plain_for_test(),
        server.uri().parse().unwrap(),
        "acct".into(),
        "admin".into(),
        SecretValue::new("k3y"),
        Some(ttl),
    )
}

pub(super) async fn mount_auth(server: &MockServer, expected: u64) {
    Mock::given(method("POST"))
        .and(path("/authn/acct/admin/authenticate"))
        .and(body_string("k3y"))
        .respond_with(ResponseTemplate::new(200).set_body_string(TOKEN_JSON))
        .expect(expected)
        .mount(server)
        .await;
}
