use super::*;

impl CyberArkSecretManager {
    pub fn with_client(
        client: reqwest::Client,
        endpoint: reqwest::Url,
        account: String,
        username: String,
        api_key: SecretValue,
        refresh_interval: Option<Duration>,
    ) -> Self {
        let endpoint = normalize_endpoint(endpoint);
        let ttl = refresh_interval
            .filter(|interval| !interval.is_zero())
            .unwrap_or(DEFAULT_REFRESH_INTERVAL);
        let token = Cache::builder()
            .time_to_live(ttl.min(MAX_TOKEN_LIFETIME))
            .build();
        let secrets = SecretCache::new(200, ttl);
        Self {
            client,
            endpoint,
            account,
            username,
            api_key,
            token,
            secrets,
            authentication_lock: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    pub fn new(
        environment: Arc<dyn Lookup + Send + Sync>,
        enterprise_enabled: bool,
    ) -> Result<Self, Error> {
        let api_key = environment.get(CYBERARK_API_KEY).unwrap_or_default();
        let cert = environment.get(CYBERARK_CLIENT_CERT).unwrap_or_default();
        let key = environment.get(CYBERARK_CLIENT_KEY).unwrap_or_default();
        if api_key.is_empty() && (cert.is_empty() || key.is_empty()) {
            return Err(Error::MissingCredentials);
        }
        if !enterprise_enabled {
            return Err(Error::EnterpriseRequired);
        }
        let verify = environment
            .get(CYBERARK_SSL_VERIFY)
            .map(|value| !value.trim().eq_ignore_ascii_case("false"))
            .unwrap_or(true);
        let mut builder = reqwest::Client::builder();
        if !verify {
            litellm_tracing::warn!(
                "CyberArk SSL verification is disabled. This is insecure and should only be used for testing with self-signed certificates."
            );
            builder = builder.danger_accept_invalid_certs(true);
        }
        if !cert.is_empty() && !key.is_empty() {
            let certificate = fs::read(cert).map_err(|_| Error::ClientCertificate)?;
            let private_key = fs::read(key).map_err(|_| Error::ClientCertificate)?;
            let identity = reqwest::Identity::from_pem(&[certificate, private_key].concat())
                .map_err(|_| Error::ClientCertificate)?;
            builder = builder.identity(identity);
        }
        let client = builder.build()?;
        let endpoint = reqwest::Url::parse(
            &environment
                .get(CYBERARK_API_BASE)
                .unwrap_or_else(|| DEFAULT_API_BASE.to_owned()),
        )
        .map_err(|_| Error::Endpoint)?;
        let account = environment
            .get(CYBERARK_ACCOUNT)
            .unwrap_or_else(|| DEFAULT_ACCOUNT.to_owned());
        let username = environment
            .get(CYBERARK_USERNAME)
            .unwrap_or_else(|| DEFAULT_USERNAME.to_owned());
        let refresh_interval = environment
            .get(CYBERARK_REFRESH_INTERVAL)
            .map(|value| {
                value
                    .parse::<u64>()
                    .map(Duration::from_secs)
                    .map_err(|_| Error::RefreshInterval)
            })
            .transpose()?;
        Ok(Self::with_client(
            client,
            endpoint,
            account,
            username,
            SecretValue::new(api_key),
            refresh_interval,
        ))
    }

    pub(super) fn authentication_url(&self) -> Result<reqwest::Url, Error> {
        self.endpoint
            .join(&format!(
                "authn/{}/{}/authenticate",
                self.account,
                utf8_percent_encode(&self.username, SECRET_NAME_SAFE)
            ))
            .map_err(|_| Error::Endpoint)
    }

    pub(super) async fn authenticate(
        &self,
        context: &CyberarkOperationContext,
    ) -> Result<SecretValue, Error> {
        if let Some(token) = self.token.get(&()).await {
            return Ok(token);
        }
        let _guard = self.authentication_lock.lock().await;
        if let Some(token) = self.token.get(&()).await {
            return Ok(token);
        }
        let url = self.authentication_url()?;
        let response = with_timeout(
            self.client.post(url).body(self.api_key.expose().to_owned()),
            context,
        )
        .send()
        .await?;
        if !response.status().is_success() {
            return Err(Error::AuthStatus(response.status().as_u16()));
        }
        let token = SecretValue::new(STANDARD.encode(response.text().await?));
        self.token.insert((), token.clone()).await;
        Ok(token)
    }

    pub(super) async fn authorization_header(
        &self,
        context: &CyberarkOperationContext,
    ) -> Result<String, Error> {
        Ok(format!(
            "Token token=\"{}\"",
            self.authenticate(context).await?.expose()
        ))
    }
}

fn normalize_endpoint(mut endpoint: reqwest::Url) -> reqwest::Url {
    if !endpoint.path().ends_with('/') {
        endpoint.set_path(&format!("{}/", endpoint.path()));
    }
    endpoint
}
