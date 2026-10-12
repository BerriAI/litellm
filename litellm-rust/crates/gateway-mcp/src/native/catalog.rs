use super::{NativeGateway, Server};
use crate::{Context, Error, RestTool, ServerOutcome, ToolCatalog};
use futures_util::{TryStreamExt, future::join_all, stream};
use rmcp::model::*;
use std::collections::BTreeSet;

impl NativeGateway {
    pub(super) async fn pages<T, F, G>(
        &self,
        server: &Server,
        context: &Context,
        request: F,
        unpack: G,
    ) -> Result<Vec<T>, Error>
    where
        T: Send,
        F: Fn(Option<PaginatedRequestParams>) -> ClientRequest,
        G: Fn(ServerResult) -> Result<(Vec<T>, Option<String>), Error>,
    {
        let pages: Vec<Vec<T>> = stream::try_unfold(Some((None, BTreeSet::new())), |state| async {
            let Some((cursor, seen)) = state else {
                return Ok(None);
            };
            let params =
                cursor.map(|cursor| PaginatedRequestParams::default().with_cursor(Some(cursor)));
            let result = self.send(server, request(params), context).await?;
            let (items, next) = unpack(result)?;
            let state = match next {
                None => None,
                Some(cursor) if seen.contains(&cursor) || seen.len() >= 1000 => {
                    return Err(Error::InvalidRequest(
                        "MCP upstream returned invalid pagination".into(),
                    ));
                }
                Some(cursor) => Some((
                    Some(cursor.clone()),
                    seen.into_iter().chain([cursor]).collect(),
                )),
            };
            Ok(Some((items, state)))
        })
        .try_collect()
        .await?;
        Ok(pages.into_iter().flatten().collect())
    }

    pub(super) async fn tools(
        &self,
        servers: &[Server],
        context: &Context,
        strict: bool,
    ) -> Result<ToolCatalog, Error> {
        let catalogs = join_all(servers.iter().map(|server| async {
            if !server
                .peer
                .peer_info()
                .is_some_and(|info| info.capabilities.tools.is_some())
            {
                return Ok::<_, Error>(Vec::new());
            }
            let tools = self
                .pages(
                    server,
                    context,
                    |params| {
                        ListToolsRequest {
                            params,
                            ..Default::default()
                        }
                        .into()
                    },
                    |result| match result {
                        ServerResult::ListToolsResult(result) => {
                            Ok((result.tools, result.next_cursor))
                        }
                        _ => Err(Error::UnexpectedResult),
                    },
                )
                .await?;
            Ok(tools
                .into_iter()
                .filter(|tool| server.allows(&tool.name))
                .map(|tool| RestTool {
                    tool,
                    mcp_info: server.info.clone(),
                })
                .collect::<Vec<_>>())
        }))
        .await;
        let outcomes = servers
            .iter()
            .zip(&catalogs)
            .map(|(server, result)| {
                let outcome = match result {
                    Ok(tools) => ServerOutcome::Ok {
                        tool_count: tools.len(),
                    },
                    Err(Error::Upstream(rmcp::service::ServiceError::Timeout { .. })) => {
                        ServerOutcome::Timeout
                    }
                    Err(Error::Upstream(
                        rmcp::service::ServiceError::TransportClosed
                        | rmcp::service::ServiceError::TransportSend(_),
                    )) => ServerOutcome::Unreachable,
                    Err(Error::Upstream(_)) => ServerOutcome::UpstreamError,
                    Err(_) => ServerOutcome::Internal,
                };
                (server.prefix(), outcome)
            })
            .collect();
        let tools = catalogs
            .into_iter()
            .map(|result| match result {
                Err(error) if strict || matches!(error, Error::Cancelled) => Err(error),
                Err(_) => Ok(Vec::new()),
                Ok(tools) => Ok(tools),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ToolCatalog {
            tools: tools.into_iter().flatten().collect(),
            server_outcomes: outcomes,
        })
    }
}
