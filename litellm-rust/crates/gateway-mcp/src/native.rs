mod catalog;
mod registry;
pub use registry::{Registry, Server, ServerResolver};

use std::{sync::Arc, time::Duration};

use futures_util::future::join_all;
use rmcp::{model::*, service::PeerRequestOptions};

use crate::{Context, Error, GatewayFuture, Operation, Operations, ToolCatalog};

pub struct NativeGateway {
    resolver: Arc<dyn ServerResolver>,
    timeout: Duration,
}

impl NativeGateway {
    pub fn new(resolver: Arc<dyn ServerResolver>, timeout: Duration) -> Self {
        Self { resolver, timeout }
    }

    async fn send(
        &self,
        server: &Server,
        mut request: ClientRequest,
        context: &Context,
    ) -> Result<ServerResult, Error> {
        if context
            .mcp
            .as_ref()
            .is_some_and(|mcp| mcp.ct.is_cancelled())
        {
            return Err(Error::Cancelled);
        }
        if let Some(mcp) = &context.mcp {
            request.get_meta_mut().extend(
                mcp.meta
                    .iter()
                    .filter(|(key, _)| {
                        !matches!(
                            key.as_str(),
                            "progressToken"
                                | "io.modelcontextprotocol/protocolVersion"
                                | "io.modelcontextprotocol/clientInfo"
                                | "io.modelcontextprotocol/clientCapabilities"
                        )
                    })
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect::<JsonObject>()
                    .into(),
            );
        }
        let options = PeerRequestOptions::with_timeout(self.timeout);
        let handle = server
            .peer
            .send_request_with_option(request, options)
            .await?;
        let request_id = handle.id.clone();
        let cancelled = async {
            match &context.mcp {
                Some(mcp) => mcp.ct.cancelled().await,
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            response = handle.await_response() => Ok(response?),
            () = cancelled => {
                let notification = CancelledNotification::new(CancelledNotificationParam::new(Some(request_id), None));
                let _ = server.peer.send_notification(notification.into()).await;
                Err(Error::Cancelled)
            }
        }
    }

    async fn execute_native(
        &self,
        operation: Operation,
        context: Context,
    ) -> Result<ServerResult, Error> {
        let servers = self.servers(&context).await?;
        match operation {
            Operation::ListTools(params) => {
                reject_cursor(&params)?;
                let catalog = self.tools(&servers, &context, false).await?;
                let tools = catalog
                    .tools
                    .into_iter()
                    .map(|entry| {
                        let prefix = entry
                            .mcp_info
                            .alias
                            .as_deref()
                            .unwrap_or(&entry.mcp_info.server_name)
                            .replace(' ', "_");
                        let mut tool = entry.tool;
                        tool.name = format!("{prefix}-{}", tool.name).into();
                        tool
                    })
                    .collect();
                let mut result = ListToolsResult::with_all_items(tools);
                result.meta = Some(MetaObject(JsonObject::from_iter([(
                    "litellm.ai/server_outcomes".into(),
                    serde_json::json!(catalog.server_outcomes),
                )])));
                Ok(result.into())
            }
            Operation::CallTool(request) => {
                let catalog = self.tools(&servers, &context, false).await?;
                let candidates: Vec<_> = catalog
                    .tools
                    .iter()
                    .filter(|entry| {
                        let prefix = entry
                            .mcp_info
                            .alias
                            .as_deref()
                            .unwrap_or(&entry.mcp_info.server_name)
                            .replace(' ', "_");
                        request.name == format!("{prefix}-{}", entry.tool.name)
                            || request.name == entry.tool.name
                    })
                    .collect();
                let [tool] = candidates.as_slice() else {
                    return Err(Error::Forbidden);
                };
                let server = servers
                    .iter()
                    .find(|server| server.info.server_id == tool.mcp_info.server_id)
                    .ok_or(Error::Forbidden)?;
                let mut upstream = request;
                upstream.name = tool.tool.name.clone();
                match self
                    .send(server, CallToolRequest::new(upstream).into(), &context)
                    .await
                {
                    Err(Error::Upstream(rmcp::service::ServiceError::McpError(error))) => {
                        Ok(CallToolResult::error(vec![ContentBlock::text(error.message)]).into())
                    }
                    result => result,
                }
            }
            Operation::ListPrompts(params) => {
                reject_cursor(&params)?;
                let listings = join_all(servers.iter().map(|server| async {
                    if !server
                        .peer
                        .peer_info()
                        .is_some_and(|info| info.capabilities.prompts.is_some())
                    {
                        return Ok::<_, Error>(Vec::new());
                    }
                    let items = self
                        .pages(
                            server,
                            &context,
                            |params| {
                                ListPromptsRequest {
                                    params,
                                    ..Default::default()
                                }
                                .into()
                            },
                            |result| match result {
                                ServerResult::ListPromptsResult(result) => {
                                    Ok((result.prompts, result.next_cursor))
                                }
                                _ => Err(Error::UnexpectedResult),
                            },
                        )
                        .await?;
                    Ok(items
                        .into_iter()
                        .map(|mut item| {
                            item.name = format!("{}-{}", server.prefix(), item.name);
                            item
                        })
                        .collect::<Vec<_>>())
                }))
                .await;
                Ok(ListPromptsResult::with_all_items(collect_listings(listings)?).into())
            }
            Operation::GetPrompt(request) => {
                let (server, name) =
                    resolve_name(&servers, &request.name, context.server.is_some())?;
                let mut upstream = request;
                upstream.name = name;
                self.send(server, GetPromptRequest::new(upstream).into(), &context)
                    .await
            }
            Operation::ListResources(params) => {
                reject_cursor(&params)?;
                let listings = join_all(servers.iter().map(|server| async {
                    if !server
                        .peer
                        .peer_info()
                        .is_some_and(|info| info.capabilities.resources.is_some())
                    {
                        return Ok::<_, Error>(Vec::new());
                    }
                    let items = self
                        .pages(
                            server,
                            &context,
                            |params| {
                                ListResourcesRequest {
                                    params,
                                    ..Default::default()
                                }
                                .into()
                            },
                            |result| match result {
                                ServerResult::ListResourcesResult(result) => {
                                    Ok((result.resources, result.next_cursor))
                                }
                                _ => Err(Error::UnexpectedResult),
                            },
                        )
                        .await?;
                    Ok(items
                        .into_iter()
                        .map(|mut item| {
                            item.name = format!("{}-{}", server.prefix(), item.name);
                            item
                        })
                        .collect::<Vec<_>>())
                }))
                .await;
                Ok(ListResourcesResult::with_all_items(collect_listings(listings)?).into())
            }
            Operation::ListResourceTemplates(params) => {
                reject_cursor(&params)?;
                let listings = join_all(servers.iter().map(|server| async {
                    if !server
                        .peer
                        .peer_info()
                        .is_some_and(|info| info.capabilities.resources.is_some())
                    {
                        return Ok::<_, Error>(Vec::new());
                    }
                    let items = self
                        .pages(
                            server,
                            &context,
                            |params| {
                                ListResourceTemplatesRequest {
                                    params,
                                    ..Default::default()
                                }
                                .into()
                            },
                            |result| match result {
                                ServerResult::ListResourceTemplatesResult(result) => {
                                    Ok((result.resource_templates, result.next_cursor))
                                }
                                _ => Err(Error::UnexpectedResult),
                            },
                        )
                        .await?;
                    Ok(items
                        .into_iter()
                        .map(|mut item| {
                            item.name = format!("{}-{}", server.prefix(), item.name);
                            item
                        })
                        .collect::<Vec<_>>())
                }))
                .await;
                Ok(ListResourceTemplatesResult::with_all_items(collect_listings(listings)?).into())
            }
            Operation::ReadResource(request) => {
                if servers.is_empty() {
                    return Err(Error::Forbidden);
                }
                let [server] = servers.as_ref() else {
                    return Err(Error::InvalidRequest(
                        "read_resource requires exactly one allowed MCP server".into(),
                    ));
                };
                self.send(server, ReadResourceRequest::new(request).into(), &context)
                    .await
            }
        }
    }
}

impl Operations for NativeGateway {
    fn authorize(&self, context: Context) -> GatewayFuture<'_, ()> {
        Box::pin(async move {
            if self.servers(&context).await?.is_empty() {
                return Err(Error::Forbidden);
            }
            Ok(())
        })
    }

    fn execute(&self, operation: Operation, context: Context) -> GatewayFuture<'_, ServerResult> {
        Box::pin(self.execute_native(operation, context))
    }

    fn rest_tools(&self, context: Context) -> GatewayFuture<'_, ToolCatalog> {
        Box::pin(async move {
            let servers = self.servers(&context).await?;
            self.tools(&servers, &context, context.server.is_some())
                .await
        })
    }
}

fn reject_cursor(params: &Option<PaginatedRequestParams>) -> Result<(), Error> {
    match params.as_ref().and_then(|params| params.cursor.as_ref()) {
        Some(_) => Err(Error::InvalidRequest(
            "Gateway listings do not accept an upstream cursor".into(),
        )),
        None => Ok(()),
    }
}

fn resolve_name<'a>(
    servers: &'a [Server],
    name: &str,
    scoped: bool,
) -> Result<(&'a Server, String), Error> {
    let matched = servers
        .iter()
        .filter_map(|server| {
            name.strip_prefix(&format!("{}-", server.prefix()))
                .map(|bare| (server, bare.to_owned()))
        })
        .max_by_key(|(server, _)| server.prefix().len());
    match matched {
        Some(found) => Ok(found),
        None if scoped && servers.len() == 1 => Ok((&servers[0], name.to_owned())),
        None => Err(Error::Forbidden),
    }
}

fn collect_listings<T>(listings: Vec<Result<Vec<T>, Error>>) -> Result<Vec<T>, Error> {
    let healthy = listings
        .into_iter()
        .map(|result| match result {
            Err(Error::Cancelled) => Err(Error::Cancelled),
            Err(_) => Ok(Vec::new()),
            Ok(items) => Ok(items),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(healthy.into_iter().flatten().collect())
}
