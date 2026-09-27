use litellm_gateway_mcp::{McpServer, NativeGateway, Registry, Server, ServerInfo};
use rmcp::{
    ErrorData, RoleServer, ServerHandler, ServiceExt,
    model::*,
    service::{PeerRequestOptions, RequestContext},
};
use rstest::rstest;
use std::{sync::Arc, time::Duration};
use tokio::sync::Notify;

struct Slow {
    entered: Arc<Notify>,
    cancelled: Arc<Notify>,
}

struct Finished(Arc<Notify>);
impl Drop for Finished {
    fn drop(&mut self) {
        self.0.notify_one();
    }
}

impl ServerHandler for Slow {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
    }

    async fn list_tools(
        &self,
        _: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult::with_all_items(vec![Tool::new(
            "wait",
            "Wait",
            Arc::new(JsonObject::new()),
        )]))
    }

    async fn call_tool(
        &self,
        _: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let _finished = Finished(self.cancelled.clone());
        self.entered.notify_one();
        context.ct.cancelled().await;
        Err(ErrorData::internal_error("cancelled", None))
    }
}

#[rstest]
#[case::client_cancel(false)]
#[case::upstream_timeout(true)]
#[tokio::test]
async fn cancellation_reaches_upstream(#[case] timeout: bool) {
    let entered = Arc::new(Notify::new());
    let cancelled = Arc::new(Notify::new());
    let upstream_server = Slow {
        entered: entered.clone(),
        cancelled: cancelled.clone(),
    };
    let (server, client) = tokio::io::duplex(65536);
    let upstream_task = tokio::spawn(async move {
        upstream_server
            .serve(server)
            .await
            .unwrap()
            .waiting()
            .await
            .unwrap()
    });
    let upstream = ().serve(client).await.unwrap();
    let registry = Registry::new(vec![Server {
        info: ServerInfo {
            server_id: "slow".into(),
            server_name: "slow".into(),
            alias: None,
        },
        peer: upstream.peer().clone(),
        allowed_tools: None,
    }])
    .unwrap();
    let duration = if timeout {
        Duration::from_millis(100)
    } else {
        Duration::from_secs(5)
    };
    let gateway = McpServer::new(Arc::new(NativeGateway::new(Arc::new(registry), duration)));
    let (server, client) = tokio::io::duplex(65536);
    let gateway_task = tokio::spawn(async move {
        gateway
            .serve(server)
            .await
            .unwrap()
            .waiting()
            .await
            .unwrap()
    });
    let client = ().serve(client).await.unwrap();
    let handle = client
        .send_request_with_option(
            CallToolRequest::new(CallToolRequestParams::new("slow-wait")).into(),
            PeerRequestOptions::with_timeout(Duration::from_secs(2)),
        )
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), entered.notified())
        .await
        .unwrap();
    if timeout {
        assert!(handle.await_response().await.is_err());
    } else {
        handle.cancel(None).await.unwrap();
    }
    tokio::time::timeout(Duration::from_secs(2), cancelled.notified())
        .await
        .unwrap();
    client.cancel().await.unwrap();
    upstream.cancel().await.unwrap();
    gateway_task.await.unwrap();
    upstream_task.await.unwrap();
}
