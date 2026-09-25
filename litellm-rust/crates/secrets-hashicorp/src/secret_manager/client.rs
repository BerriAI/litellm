use super::*;

impl HashicorpVault {
    pub(super) async fn client_for_location(
        &self,
        location: &SecretLocation,
    ) -> Result<Arc<VaultClient>, Error> {
        let client = self.vault_client().await?;
        if client.settings.namespace == location.namespace {
            return Ok(client);
        }
        self.build_client(location.namespace.as_deref(), &client.settings.token)
            .map(Arc::new)
    }

    pub(super) async fn vault_client(&self) -> Result<Arc<VaultClient>, Error> {
        let mut cached = self.auth_client.lock().await;
        if let Some(entry) = cached.as_ref()
            && entry
                .expires_at
                .is_none_or(|expires_at| expires_at > Instant::now())
        {
            return Ok(entry.client.clone());
        }

        let (client, expires_at): (VaultClient, Option<Instant>) =
            match (self.config.approle.as_ref(), self.config.tls_cert.as_ref()) {
                (Some(approle), _) => {
                    let login_client: VaultClient =
                        self.build_client(self.config.login_namespace(), "")?;
                    let auth = approle::login(
                        &login_client,
                        &approle.mount_path,
                        &approle.role_id,
                        approle.secret_id.expose(),
                    )
                    .await
                    .map_err(|error| map_api_error(error, ErrorContext::Login))?;
                    (
                        self.build_client(self.config.secret_namespace(), &auth.client_token)?,
                        token_expiry(auth.lease_duration),
                    )
                }
                (None, Some(tls)) => {
                    let login_client: VaultClient =
                        self.build_client(self.config.login_namespace(), "")?;
                    let endpoint: CertLoginRequest = CertLoginRequest::new(tls.role.as_deref());
                    let auth = api::auth(&login_client, endpoint)
                        .await
                        .map_err(|error| map_api_error(error, ErrorContext::Login))?;
                    (
                        self.build_client(self.config.secret_namespace(), &auth.client_token)?,
                        token_expiry(auth.lease_duration),
                    )
                }
                (None, None) => {
                    let token: SecretValue =
                        self.config.token.clone().ok_or(Error::NoAuthConfigured)?;
                    (
                        self.build_client(self.config.secret_namespace(), token.expose())?,
                        None,
                    )
                }
            };
        let client: Arc<VaultClient> = Arc::new(client);
        *cached = Some(CachedClient {
            client: client.clone(),
            expires_at,
        });
        Ok(client)
    }

    pub(super) fn build_client(
        &self,
        namespace: Option<&str>,
        token: &str,
    ) -> Result<VaultClient, Error> {
        let settings = VaultClientSettingsBuilder::default()
            .address(&self.config.address)
            .token(token.to_owned())
            .namespace(namespace.map(str::to_owned))
            .identity(identity_for(self.config.tls_cert.as_ref())?)
            .ca_certs(Vec::new())
            .verify(true)
            .build()
            .map_err(|message| Error::ClientSettings {
                message: message.to_string(),
            })?;
        VaultClient::new(settings).map_err(Error::Client)
    }
}

fn identity_for(tls: Option<&TlsCertAuth>) -> Result<Option<Identity>, Error> {
    tls.map(|tls| {
        let cert: Vec<u8> = std::fs::read(&tls.cert_path).map_err(|source| Error::TlsIdentity {
            path: tls.cert_path.clone(),
            message: source.to_string(),
        })?;
        let key: Vec<u8> = std::fs::read(&tls.key_path).map_err(|source| Error::TlsIdentity {
            path: tls.key_path.clone(),
            message: source.to_string(),
        })?;
        Identity::from_pem(&[cert.as_slice(), key.as_slice()].concat()).map_err(|source| {
            Error::TlsIdentity {
                path: tls.cert_path.clone(),
                message: source.to_string(),
            }
        })
    })
    .transpose()
}

fn token_expiry(lease_duration: u64) -> Option<Instant> {
    (lease_duration > 0).then(|| Instant::now() + Duration::from_secs(lease_duration))
}
