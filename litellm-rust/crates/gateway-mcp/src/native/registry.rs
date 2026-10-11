use super::NativeGateway;
use crate::{Context, Error, GatewayFuture, ServerInfo};
use rmcp::{Peer, RoleClient};
use std::sync::Arc;

#[derive(Clone)]
pub struct Server {
    pub info: ServerInfo,
    pub peer: Peer<RoleClient>,
    pub allowed_tools: Option<Arc<[String]>>,
}

impl Server {
    pub(super) fn prefix(&self) -> String {
        self.info
            .alias
            .as_deref()
            .unwrap_or(&self.info.server_name)
            .replace(' ', "_")
    }

    pub(super) fn matches(&self, name: &str) -> bool {
        self.info.server_id == name
            || self.info.server_name == name
            || self.info.alias.as_deref() == Some(name)
    }

    pub(super) fn allows(&self, name: &str) -> bool {
        self.allowed_tools
            .as_ref()
            .is_none_or(|allowed| allowed.iter().any(|entry| entry == name))
    }
}

pub trait ServerResolver: Send + Sync + 'static {
    fn resolve<'a>(&'a self, context: &'a Context) -> GatewayFuture<'a, Arc<[Server]>>;
}

pub struct Registry(Arc<[Server]>);

impl Registry {
    pub fn new(servers: impl Into<Arc<[Server]>>) -> Result<Self, Error> {
        let servers = servers.into();
        for (index, server) in servers.iter().enumerate() {
            let identifiers = [&server.info.server_id, &server.info.server_name];
            if identifiers.iter().any(|value| value.is_empty()) || server.prefix().is_empty() {
                return Err(Error::Configuration(
                    "MCP server identifiers must not be empty".into(),
                ));
            }
            if servers[..index].iter().any(|previous| {
                previous.prefix() == server.prefix()
                    || identifiers.iter().any(|value| previous.matches(value))
                    || server
                        .info
                        .alias
                        .as_deref()
                        .is_some_and(|alias| previous.matches(alias))
            }) {
                return Err(Error::Configuration(
                    "MCP server identifiers and prefixes must be unique".into(),
                ));
            }
        }
        Ok(Self(servers))
    }
}

impl ServerResolver for Registry {
    fn resolve<'a>(&'a self, _: &'a Context) -> GatewayFuture<'a, Arc<[Server]>> {
        Box::pin(async { Ok(self.0.clone()) })
    }
}

impl NativeGateway {
    pub(super) async fn servers(&self, context: &Context) -> Result<Arc<[Server]>, Error> {
        let authorized = self.resolver.resolve(context).await?;
        let header = context
            .parts
            .headers
            .get("x-mcp-servers")
            .map(|value| {
                value
                    .to_str()
                    .map_err(|_| Error::InvalidRequest("Invalid x-mcp-servers header".into()))
            })
            .transpose()?;
        let scope = context.server.as_deref().or(header);
        let Some(scope) = scope else {
            return Ok(authorized);
        };
        let names: Vec<_> = scope
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .collect();
        if names.is_empty()
            || names
                .iter()
                .any(|name| !authorized.iter().any(|server| server.matches(name)))
        {
            return Err(Error::Forbidden);
        }
        let selected: Arc<[Server]> = authorized
            .iter()
            .filter(|server| names.iter().any(|name| server.matches(name)))
            .cloned()
            .collect();
        if let Some(header) = header {
            let requested: Vec<_> = header
                .split(',')
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .collect();
            if requested.is_empty()
                || requested
                    .iter()
                    .any(|name| !selected.iter().any(|server| server.matches(name)))
            {
                return Err(Error::Forbidden);
            }
            return Ok(selected
                .iter()
                .filter(|server| requested.iter().any(|name| server.matches(name)))
                .cloned()
                .collect());
        }
        Ok(selected)
    }
}
