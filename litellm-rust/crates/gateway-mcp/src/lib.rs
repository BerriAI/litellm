mod error;
mod legacy;
mod native;
mod relay;
mod routes;
mod server;
mod sessions;

use std::{collections::BTreeMap, future::Future, pin::Pin, sync::Arc};

use http::request::Parts;
use rmcp::{RoleServer, model::*, service::RequestContext};

pub use error::Error;
pub use native::{NativeGateway, Registry, Server, ServerResolver};
pub use relay::RelayClient;
pub use rmcp;
pub use routes::{HttpConfig, router};
pub use server::McpServer;
pub use sessions::SessionOwner;

pub type GatewayFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, Error>> + Send + 'a>>;

pub struct Context {
    pub parts: Parts,
    pub server: Option<String>,
    pub mcp: Option<RequestContext<RoleServer>>,
}

#[derive(Clone, Debug)]
pub enum Operation {
    ListTools(Option<PaginatedRequestParams>),
    CallTool(CallToolRequestParams),
    ListPrompts(Option<PaginatedRequestParams>),
    GetPrompt(GetPromptRequestParams),
    ListResources(Option<PaginatedRequestParams>),
    ListResourceTemplates(Option<PaginatedRequestParams>),
    ReadResource(ReadResourceRequestParams),
}

#[derive(serde::Serialize)]
pub struct RestTool {
    #[serde(flatten)]
    pub tool: Tool,
    pub mcp_info: ServerInfo,
}

#[derive(serde::Serialize)]
pub struct ToolCatalog {
    pub tools: Vec<RestTool>,
    pub server_outcomes: BTreeMap<String, ServerOutcome>,
}

#[derive(serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ServerOutcome {
    Ok { tool_count: usize },
    Timeout,
    Unreachable,
    UpstreamError,
    Internal,
}

#[derive(Clone, serde::Serialize)]
pub struct ServerInfo {
    pub server_id: String,
    pub server_name: String,
    pub alias: Option<String>,
}

pub trait Operations: Send + Sync + 'static {
    fn authorize(&self, context: Context) -> GatewayFuture<'_, ()>;
    fn execute(&self, operation: Operation, context: Context) -> GatewayFuture<'_, ServerResult>;
    fn rest_tools(&self, context: Context) -> GatewayFuture<'_, ToolCatalog>;
}

pub trait OperationAuthorizer: Send + Sync {
    fn authorize<'a>(
        &'a self,
        server: &'a ServerInfo,
        request: Option<&'a ClientRequest>,
    ) -> GatewayFuture<'a, ()>;
}

#[derive(Clone)]
pub struct Authorization(pub Arc<dyn OperationAuthorizer>);
