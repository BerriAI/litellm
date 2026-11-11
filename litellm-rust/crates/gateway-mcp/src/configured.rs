use std::{
    collections::{BTreeMap, HashMap},
    num::NonZeroUsize,
    sync::Arc,
    time::Duration,
};

use base64::{Engine, engine::general_purpose::STANDARD};
use futures_util::future::try_join_all;
use http::{HeaderName, HeaderValue};
use litellm_config::{McpAuth, McpServer, McpTransport};
use litellm_secret_value::SecretValue;
use litellm_secrets::source::SecretSource;
use rmcp::{
    RoleClient, ServiceExt,
    service::RunningService,
    transport::{
        StreamableHttpClientTransport, TokioChildProcess,
        streamable_http_client::{StreamableHttpClient, StreamableHttpClientTransportConfig},
    },
};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::{ConfigFault, ConnectError, Limits, NativeGateway, Registry, Server, ServerInfo};

pub struct ConfiguredGateway {
    pub operations: Arc<NativeGateway>,
    services: Vec<RunningService<RoleClient, ()>>,
}

fn invalid(server: &str, fault: ConfigFault) -> ConnectError {
    ConnectError::Configuration {
        server: server.into(),
        fault,
    }
}

fn timeout(name: &str, config: &McpServer) -> Result<Duration, ConnectError> {
    let duration = Duration::try_from_secs_f64(config.timeout.unwrap_or(30.0))
        .map_err(|source| invalid(name, ConfigFault::InvalidTimeout(source)))?;
    if duration.is_zero() {
        return Err(invalid(name, ConfigFault::ZeroTimeout));
    }
    Ok(duration)
}

fn info(name: &str, config: &McpServer) -> ServerInfo {
    let identity = format!(
        "{name}|{}|{}|{}|{}",
        config.url.as_ref().map_or("", SecretValue::expose),
        <&'static str>::from(config.transport),
        config.auth_type.map_or("", <&'static str>::from),
        config.alias.as_deref().unwrap_or("")
    );
    ServerInfo {
        server_id: config
            .server_id
            .clone()
            .unwrap_or_else(|| format!("{:x}", Sha256::digest(identity))[..32].into()),
        server_name: name.into(),
        alias: config.alias.clone(),
    }
}

fn validate(name: &str, config: &McpServer) -> Result<(), ConnectError> {
    timeout(name, config)?;
    if name.trim().is_empty()
        || config
            .server_id
            .as_ref()
            .is_some_and(|id| id.trim().is_empty())
        || config
            .alias
            .as_ref()
            .is_some_and(|alias| alias.trim().is_empty())
    {
        return Err(invalid(name, ConfigFault::BlankIdentifier));
    }
    if let Some(field) = config.unsupported.keys().next() {
        return Err(invalid(
            name,
            ConfigFault::UnsupportedSetting(field.clone()),
        ));
    }
    if config
        .max_concurrent_requests
        .is_some_and(|limit| limit == 0 || limit > tokio::sync::Semaphore::MAX_PERMITS)
    {
        return Err(invalid(name, ConfigFault::ConcurrencyOutOfRange));
    }
    if config.mcp_info.contains_key("protocol_version")
        || config.mcp_info.contains_key("mcp_server_cost_info")
    {
        return Err(invalid(name, ConfigFault::UnsupportedMcpInfo));
    }
    if !matches!(
        config.auth_type,
        None | Some(
            McpAuth::None
                | McpAuth::ApiKey
                | McpAuth::BearerToken
                | McpAuth::Basic
                | McpAuth::Authorization
                | McpAuth::Token
        )
    ) {
        return Err(invalid(name, ConfigFault::UnsupportedAuth));
    }
    match config.transport {
        McpTransport::Sse => {
            return Err(invalid(name, ConfigFault::SseTransport));
        }
        McpTransport::Http => {
            if config.url.is_none()
                || config.command.is_some()
                || !config.args.is_empty()
                || !config.env.is_empty()
            {
                return Err(invalid(name, ConfigFault::HttpShape));
            }
        }
        McpTransport::Stdio => {
            if config
                .command
                .as_ref()
                .is_none_or(|command| command.trim().is_empty())
                || config.url.is_some()
            {
                return Err(invalid(name, ConfigFault::StdioShape));
            }
            if config.authentication_token.is_some()
                || !config.static_headers.is_empty()
                || config.upstream_token_header.is_some()
                || !matches!(config.auth_type, None | Some(McpAuth::None))
            {
                return Err(invalid(name, ConfigFault::StdioCredentials));
            }
        }
    }
    if matches!(config.auth_type, None | Some(McpAuth::None)) {
        if config.authentication_token.is_some() || config.upstream_token_header.is_some() {
            return Err(invalid(name, ConfigFault::TokenWithoutAuth));
        }
    } else if config.authentication_token.is_none() {
        return Err(invalid(name, ConfigFault::AuthWithoutToken));
    }
    Ok(())
}

async fn resolve(
    name: &str,
    value: &SecretValue,
    secrets: &dyn SecretSource,
) -> Result<String, ConnectError> {
    match value.expose().strip_prefix("os.environ/") {
        Some(key) => secrets
            .get_secret_str(key)
            .await?
            .map(|secret| secret.expose().to_owned())
            .ok_or_else(|| invalid(name, ConfigFault::MissingSecret)),
        None => Ok(value.expose().to_owned()),
    }
}

fn header(
    server: &str,
    name: &str,
    value: &str,
) -> Result<(HeaderName, HeaderValue), ConnectError> {
    let key = HeaderName::try_from(name)
        .map_err(|source| invalid(server, ConfigFault::HeaderName(source)))?;
    if matches!(
        key.as_str(),
        "host"
            | "content-length"
            | "content-type"
            | "transfer-encoding"
            | "connection"
            | "accept"
            | "mcp-session-id"
            | "mcp-protocol-version"
            | "last-event-id"
    ) {
        return Err(invalid(server, ConfigFault::FramingHeader));
    }
    let mut value = HeaderValue::try_from(value)
        .map_err(|source| invalid(server, ConfigFault::HeaderValue(source)))?;
    value.set_sensitive(true);
    Ok((key, value))
}

fn scheme(value: &str, prefix: &str) -> String {
    let bare = value
        .get(..prefix.len())
        .filter(|part| part.eq_ignore_ascii_case(prefix))
        .and_then(|_| value.get(prefix.len()..))
        .and_then(|suffix| suffix.strip_prefix(' '))
        .unwrap_or(value);
    format!("{prefix} {bare}")
}

fn basic(value: &str) -> String {
    let normalized = scheme(value, "Basic");
    let credentials = normalized.strip_prefix("Basic ").unwrap_or(value);
    let encoded = if STANDARD.decode(credentials).is_ok() {
        credentials.to_owned()
    } else {
        STANDARD.encode(credentials)
    };
    format!("Basic {encoded}")
}

async fn headers(
    name: &str,
    config: &McpServer,
    secrets: &dyn SecretSource,
) -> Result<HashMap<HeaderName, HeaderValue>, ConnectError> {
    let entries = try_join_all(config.static_headers.iter().map(|(key, value)| async move {
        header(name, key, &resolve(name, value, secrets).await?)
    }))
    .await?;
    let auth = match (&config.authentication_token, config.auth_type) {
        (Some(token), Some(kind)) => {
            let token = resolve(name, token, secrets).await?;
            if token.trim().is_empty() {
                return Err(invalid(name, ConfigFault::EmptyCredential));
            }
            let (slot, value) = match kind {
                McpAuth::ApiKey => ("x-api-key", token),
                McpAuth::BearerToken => ("authorization", scheme(&token, "Bearer")),
                McpAuth::Token => ("authorization", scheme(&token, "token")),
                McpAuth::Basic => ("authorization", basic(&token)),
                McpAuth::Authorization => ("authorization", token),
                _ => return Err(invalid(name, ConfigFault::UnsupportedAuth)),
            };
            Some(header(
                name,
                config.upstream_token_header.as_deref().unwrap_or(slot),
                &value,
            )?)
        }
        _ => None,
    };
    Ok(entries.into_iter().chain(auth).collect())
}

impl ConfiguredGateway {
    pub async fn connect<C: StreamableHttpClient>(
        configs: &BTreeMap<String, McpServer>,
        client: C,
        secrets: &dyn SecretSource,
        shutdown: CancellationToken,
    ) -> Result<Self, ConnectError> {
        for (name, config) in configs {
            validate(name, config)?;
        }
        let connected = try_join_all(configs.iter().map(|(name, config)| {
            connect_one(
                name,
                config,
                client.clone(),
                secrets,
                shutdown.child_token(),
            )
        }))
        .await?;
        let (servers, services, limits) = connected.into_iter().fold(
            (Vec::new(), Vec::new(), BTreeMap::new()),
            |(mut servers, mut services, mut limits), (server, running, limit)| {
                limits.insert(server.info.server_id.clone(), limit);
                servers.push(server);
                services.push(running);
                (servers, services, limits)
            },
        );
        let registry = Registry::new(servers)?;
        Ok(Self {
            operations: Arc::new(
                NativeGateway::new(Arc::new(registry), Duration::from_secs(30)).with_limits(limits),
            ),
            services,
        })
    }

    pub async fn close(self) {
        futures_util::future::join_all(self.services.into_iter().map(|service| service.cancel()))
            .await;
    }
}

async fn connect_one<C: StreamableHttpClient>(
    name: &str,
    config: &McpServer,
    client: C,
    secrets: &dyn SecretSource,
    shutdown: CancellationToken,
) -> Result<(Server, RunningService<RoleClient, ()>, Limits), ConnectError> {
    let duration = timeout(name, config)?;
    let running = tokio::select! {
        result = tokio::time::timeout(duration, connect_service(name, config, client, secrets, shutdown.clone())) => result.map_err(|_| ConnectError::Timeout(name.into()))??,
        () = shutdown.cancelled() => return Err(ConnectError::Cancelled),
    };
    let server = Server {
        info: info(name, config),
        peer: running.peer().clone(),
        allowed_tools: config
            .allowed_tools
            .as_deref()
            .filter(|tools| !tools.is_empty())
            .map(Arc::from),
    };
    let limits = Limits::new(
        duration,
        config.max_concurrent_requests.and_then(NonZeroUsize::new),
    );
    Ok((server, running, limits))
}

async fn connect_service<C: StreamableHttpClient>(
    name: &str,
    config: &McpServer,
    client: C,
    secrets: &dyn SecretSource,
    shutdown: CancellationToken,
) -> Result<RunningService<RoleClient, ()>, ConnectError> {
    let initialized = match config.transport {
        McpTransport::Http => {
            let configured = config
                .url
                .as_ref()
                .ok_or_else(|| invalid(name, ConfigFault::HttpShape))?;
            let url = resolve(name, configured, secrets).await?;
            let parsed = url::Url::parse(&url)
                .map_err(|source| invalid(name, ConfigFault::InvalidUrl(source)))?;
            if !matches!(parsed.scheme(), "http" | "https")
                || parsed.host_str().is_none()
                || !parsed.username().is_empty()
                || parsed.password().is_some()
                || parsed.fragment().is_some()
            {
                return Err(invalid(name, ConfigFault::UrlShape));
            }
            let transport = StreamableHttpClientTransport::with_client(
                client,
                StreamableHttpClientTransportConfig::with_uri(url)
                    .custom_headers(headers(name, config, secrets).await?),
            );
            ().serve_with_ct(transport, shutdown).await
        }
        McpTransport::Stdio => {
            let environment = try_join_all(config.env.iter().map(|(key, value)| async move {
                Ok::<_, ConnectError>((key, resolve(name, value, secrets).await?))
            }))
            .await?;
            let program = config
                .command
                .as_deref()
                .ok_or_else(|| invalid(name, ConfigFault::StdioShape))?;
            let mut command = tokio::process::Command::new(program);
            command
                .args(&config.args)
                .envs(environment)
                .kill_on_drop(true);
            ().serve_with_ct(TokioChildProcess::new(command)?, shutdown)
                .await
        }
        McpTransport::Sse => return Err(invalid(name, ConfigFault::SseTransport)),
    };
    initialized.map_err(|source| ConnectError::Initialize {
        server: name.into(),
        source: Box::new(source),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    struct Secrets;

    impl SecretSource for Secrets {
        fn get_secret_str<'a>(
            &'a self,
            name: &'a str,
        ) -> futures_util::future::BoxFuture<'a, Result<Option<SecretValue>, litellm_secrets::Error>>
        {
            Box::pin(async move { Ok((name == "KEY").then(|| SecretValue::new("resolved-token"))) })
        }
    }

    #[rstest]
    #[case::api_key("api_key", "token-value", "x-api-key", "token-value")]
    #[case::bearer("bearer_token", "token-value", "authorization", "Bearer token-value")]
    #[case::prefixed_bearer(
        "bearer_token",
        "bEaReR token-value",
        "authorization",
        "Bearer token-value"
    )]
    #[case::token("token", "token token-value", "authorization", "token token-value")]
    #[case::basic(
        "basic",
        "user:password",
        "authorization",
        "Basic dXNlcjpwYXNzd29yZA=="
    )]
    #[case::prefixed_basic(
        "basic",
        "Basic dXNlcjpwYXNzd29yZA==",
        "authorization",
        "Basic dXNlcjpwYXNzd29yZA=="
    )]
    #[case::authorization(
        "authorization",
        "Custom token-value",
        "authorization",
        "Custom token-value"
    )]
    #[case::secret_reference("api_key", "os.environ/KEY", "x-api-key", "resolved-token")]
    #[tokio::test]
    async fn builds_configured_auth_header(
        #[case] mode: &str,
        #[case] token: &str,
        #[case] slot: &str,
        #[case] expected: &str,
    ) {
        let config = litellm_config::Config::from_yaml(&format!(
            "mcp_servers:\n  test:\n    auth_type: {mode}\n    authentication_token: '{token}'\n"
        ))
        .unwrap();
        let built = headers("test", &config.mcp_servers["test"], &Secrets)
            .await
            .unwrap();
        let key = HeaderName::try_from(slot).unwrap();
        assert_eq!(built[&key], expected);
        assert!(built[&key].is_sensitive());
    }

    #[rstest]
    #[tokio::test]
    async fn explicit_credential_slot_overrides_static_header() {
        let config = litellm_config::Config::from_yaml("mcp_servers: {test: {auth_type: api_key, authentication_token: os.environ/KEY, upstream_token_header: x-custom, static_headers: {X-Custom: old, x-extra: extra}}}").unwrap();
        let built = headers("test", &config.mcp_servers["test"], &Secrets)
            .await
            .unwrap();
        assert_eq!(
            built[&HeaderName::from_static("x-custom")],
            "resolved-token"
        );
        assert_eq!(built[&HeaderName::from_static("x-extra")], "extra");
        assert_eq!(built.len(), 2);
    }
}
