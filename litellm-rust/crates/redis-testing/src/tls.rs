use std::path::PathBuf;

use rcgen::{BasicConstraints, CertificateParams, CertifiedIssuer, DnType, IsCa, KeyPair};
use tempfile::TempDir;
use testcontainers_modules::testcontainers::{
    ContainerAsync, GenericImage, ImageExt,
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
};

use crate::{
    Error,
    ports::{HOST, start_on_free_ports, wait_for_listener},
    standalone::REDIS_TAG,
};

pub const SERVER_NAME: &str = "localhost";

pub struct TlsFiles {
    _directory: TempDir,
    pub ca: PathBuf,
    pub client_cert: PathBuf,
    pub client_key: PathBuf,
}

struct Generated {
    files: TlsFiles,
    ca_pem: String,
    server_cert: String,
    server_key: String,
}

pub struct RedisTlsServer {
    _container: ContainerAsync<GenericImage>,
    port: u16,
    files: TlsFiles,
}

impl RedisTlsServer {
    pub async fn start(require_client_certificate: bool) -> Result<Self, Error> {
        start_on_free_ports(|[port]| Self::start_on(port, require_client_certificate)).await
    }

    async fn start_on(port: u16, require_client_certificate: bool) -> Result<Self, Error> {
        let generated = generate()?;
        let auth_clients = match require_client_certificate {
            true => "yes",
            false => "no",
        };
        let container = GenericImage::new("redis", REDIS_TAG)
            .with_wait_for(WaitFor::message_on_stdout("Ready to accept connections"))
            .with_copy_to("/tls/ca.crt", generated.ca_pem.into_bytes())
            .with_copy_to("/tls/server.crt", generated.server_cert.into_bytes())
            .with_copy_to("/tls/server.key", generated.server_key.into_bytes())
            .with_cmd([
                "redis-server",
                "--port",
                "0",
                "--tls-port",
                &port.to_string(),
                "--tls-cert-file",
                "/tls/server.crt",
                "--tls-key-file",
                "/tls/server.key",
                "--tls-ca-cert-file",
                "/tls/ca.crt",
                "--tls-auth-clients",
                auth_clients,
            ])
            .with_mapped_port(port, port.tcp())
            .start()
            .await?;
        wait_for_listener(port).await?;
        Ok(Self {
            _container: container,
            port,
            files: generated.files,
        })
    }

    pub fn host(&self) -> &'static str {
        HOST
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn files(&self) -> &TlsFiles {
        &self.files
    }
}

fn generate() -> Result<Generated, Error> {
    let named = |names: Vec<String>, common_name: &str| -> Result<CertificateParams, Error> {
        let mut params = CertificateParams::new(names)?;
        params
            .distinguished_name
            .push(DnType::CommonName, common_name);
        Ok(params)
    };
    let mut ca_params = named(Vec::new(), "redis-testing CA")?;
    ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    let ca = CertifiedIssuer::self_signed(ca_params, KeyPair::generate()?)?;
    let server_key = KeyPair::generate()?;
    let server =
        named(vec![SERVER_NAME.to_owned()], "redis-testing server")?.signed_by(&server_key, &ca)?;
    let client_key = KeyPair::generate()?;
    let client =
        named(vec!["client".to_owned()], "redis-testing client")?.signed_by(&client_key, &ca)?;
    let directory = tempfile::tempdir().map_err(Error::Files)?;
    let write = |name: &str, contents: String| -> Result<PathBuf, Error> {
        let path = directory.path().join(name);
        std::fs::write(&path, contents).map_err(Error::Files)?;
        Ok(path)
    };
    let ca_path = write("ca.pem", ca.pem())?;
    let client_cert = write("client.pem", client.pem())?;
    let client_key_path = write("client.key", client_key.serialize_pem())?;
    Ok(Generated {
        ca_pem: ca.pem(),
        server_cert: server.pem(),
        server_key: server_key.serialize_pem(),
        files: TlsFiles {
            _directory: directory,
            ca: ca_path,
            client_cert,
            client_key: client_key_path,
        },
    })
}
