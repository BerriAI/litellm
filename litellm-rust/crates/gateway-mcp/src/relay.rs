use rmcp::{
    ClientHandler, ErrorData, Peer, RoleClient, RoleServer,
    model::*,
    service::{NotificationContext, RequestContext},
};

use crate::Error;

#[derive(Clone)]
pub struct RelayClient {
    downstream: Peer<RoleServer>,
    info: ClientConfig,
    progress_token: Option<ProgressToken>,
}

impl RelayClient {
    pub fn for_request(context: &RequestContext<RoleServer>) -> Self {
        Self {
            downstream: context.peer.clone(),
            info: ClientConfig::new(
                context.client_capabilities().unwrap_or_default(),
                Implementation::new("litellm-mcp-gateway", "1.0.0"),
            ),
            progress_token: context.meta.get_progress_token(),
        }
    }
}

impl ClientHandler for RelayClient {
    fn get_info(&self) -> ClientConfig {
        self.info.clone()
    }

    #[allow(
        deprecated,
        reason = "preserve sampling for legacy MCP clients supported by the Python gateway"
    )]
    async fn create_message(
        &self,
        params: CreateMessageRequestParams,
        _: RequestContext<RoleClient>,
    ) -> Result<CreateMessageResult, ErrorData> {
        self.downstream
            .create_message(params)
            .await
            .map_err(|error| Error::Upstream(error).into_mcp())
    }

    #[allow(
        deprecated,
        reason = "preserve roots for legacy MCP clients supported by the Python gateway"
    )]
    async fn list_roots(
        &self,
        _: RequestContext<RoleClient>,
    ) -> Result<ListRootsResult, ErrorData> {
        self.downstream
            .list_roots()
            .await
            .map_err(|error| Error::Upstream(error).into_mcp())
    }

    async fn create_elicitation(
        &self,
        params: ElicitRequestParams,
        _: RequestContext<RoleClient>,
    ) -> Result<ElicitResult, ErrorData> {
        self.downstream
            .create_elicitation(params)
            .await
            .map_err(|error| Error::Upstream(error).into_mcp())
    }

    async fn on_progress(
        &self,
        mut params: ProgressNotificationParam,
        _: NotificationContext<RoleClient>,
    ) {
        if let Some(token) = &self.progress_token {
            params.progress_token = token.clone();
            let _ = self.downstream.notify_progress(params).await;
        }
    }
}
