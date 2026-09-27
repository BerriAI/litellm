use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, Query, Request, State},
    http::request::Parts,
    middleware::{self, Next},
    response::Response,
    routing::{get, post},
};
use rmcp::{model::*, transport::streamable_http_server::StreamableHttpServerConfig};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::{Context, Error, McpServer, Operation, Operations, ToolCatalog, server::ServerScope};

pub struct HttpConfig {
    pub allowed_hosts: Vec<String>,
    pub allowed_origins: Vec<String>,
    pub cancellation_token: CancellationToken,
    pub server_info: ServerConfig,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            allowed_hosts: vec!["localhost".into(), "127.0.0.1".into(), "::1".into()],
            allowed_origins: Vec::new(),
            cancellation_token: CancellationToken::new(),
            server_info: ServerConfig::new(
                ServerCapabilities::builder()
                    .enable_tools()
                    .enable_prompts()
                    .enable_resources()
                    .build(),
            )
            .with_server_info(Implementation::new("litellm-mcp-server", "1.0.0")),
        }
    }
}

pub fn router(operations: Arc<dyn Operations>, config: HttpConfig) -> Router {
    let server = McpServer::new(operations.clone()).with_info(config.server_info.clone());
    let legacy = crate::legacy::router(operations.clone(), server.clone(), &config);
    let transport_config = StreamableHttpServerConfig::default()
        .with_legacy_session_mode(false)
        .with_allowed_hosts(config.allowed_hosts)
        .with_allowed_origins(config.allowed_origins)
        .enforce_origin_validation()
        .with_cancellation_token(config.cancellation_token);
    let service = crate::sessions::transport(server, transport_config);
    let scoped = Router::new()
        .route_service("/mcp/{server}", service.clone())
        .route_service("/mcp/{server}/mcp", service.clone())
        .route_service("/{server}/mcp", service.clone())
        .route_service("/{server}/mcp/", service.clone())
        .route_layer(middleware::from_fn(scope_server));
    Router::new()
        .route_service("/mcp", service.clone())
        .route_service("/mcp/", service)
        .merge(scoped)
        .route(
            "/mcp/enabled",
            get(|| async { Json(Enabled { enabled: true }) }),
        )
        .route("/mcp-rest/tools/list", get(rest_list))
        .route("/mcp-rest/tools/call", post(rest_call))
        .with_state(operations)
        .merge(legacy)
}

async fn scope_server(Path(server): Path<String>, mut request: Request, next: Next) -> Response {
    request.extensions_mut().insert(ServerScope(server));
    next.run(request).await
}

#[derive(Serialize)]
struct Enabled {
    enabled: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ListQuery {
    server_id: Option<String>,
    mcp_server_name: Option<String>,
}

#[derive(Serialize)]
struct ToolList {
    #[serde(flatten)]
    catalog: ToolCatalog,
    error: Option<String>,
    message: String,
}

async fn rest_list(
    State(operations): State<Arc<dyn Operations>>,
    Query(query): Query<ListQuery>,
    parts: Parts,
) -> Result<Json<ToolList>, Error> {
    let context = Context {
        parts,
        server: query.server_id.or(query.mcp_server_name),
        mcp: None,
    };
    let catalog = operations.rest_tools(context).await?;
    let failed = catalog.tools.is_empty()
        && catalog
            .server_outcomes
            .values()
            .any(|outcome| !matches!(outcome, crate::ServerOutcome::Ok { .. }));
    Ok(Json(ToolList {
        catalog,
        error: failed.then(|| "partial_failure".into()),
        message: if failed {
            "Failed to get tools from servers"
        } else {
            "Successfully retrieved tools"
        }
        .into(),
    }))
}

#[derive(Deserialize)]
struct ToolCall {
    server_id: Option<String>,
    name: Option<String>,
    arguments: Option<JsonObject>,
}

async fn rest_call(
    State(operations): State<Arc<dyn Operations>>,
    parts: Parts,
    Json(call): Json<ToolCall>,
) -> Result<Json<ServerResult>, Error> {
    let server = call
        .server_id
        .filter(|id| !id.is_empty())
        .ok_or_else(|| Error::InvalidRequest("server_id is required in request body".into()))?;
    let name = call
        .name
        .filter(|name| !name.is_empty())
        .ok_or_else(|| Error::InvalidRequest("name is required in request body".into()))?;
    let params =
        CallToolRequestParams::new(name).with_arguments(call.arguments.unwrap_or_default());
    let context = Context {
        parts,
        server: Some(server),
        mcp: None,
    };
    Ok(Json(
        operations
            .execute(Operation::CallTool(params), context)
            .await?,
    ))
}
