use litellm_gateway_mcp::{Limits, McpServer, NativeGateway, Registry, Server, ServerInfo};
use rmcp::{
    ErrorData, RoleServer, ServerHandler, ServiceExt,
    model::*,
    service::{PeerRequestOptions, RequestContext},
};
use rstest::rstest;
use std::{collections::BTreeMap, num::NonZeroUsize, sync::Arc, time::Duration};
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
#[case::client_cancel(false, false)]
#[case::upstream_timeout(true, false)]
#[case::configured_timeout(true, true)]
#[case::configured_concurrency(false, true)]
#[tokio::test]
async fn cancellation_reaches_upstream(#[case] timeout: bool, #[case] configured: bool) {
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
    let native = NativeGateway::new(
        Arc::new(registry),
        if configured {
            Duration::from_secs(5)
        } else {
            duration
        },
    );
    let native = if configured {
        native.with_limits(BTreeMap::from([(
            "slow".into(),
            Limits::new(duration, NonZeroUsize::new(1)),
        )]))
    } else {
        native
    };
    let gateway = McpServer::new(Arc::new(native));
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
    let queued = if configured && !timeout {
        let second = client
            .send_request_with_option(
                CallToolRequest::new(CallToolRequestParams::new("slow-wait")).into(),
                PeerRequestOptions::with_timeout(Duration::from_secs(5)),
            )
            .await
            .unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(100), entered.notified())
                .await
                .is_err()
        );
        Some(second)
    } else {
        None
    };
    if timeout {
        assert!(matches!(
            handle.await_response().await,
            Err(rmcp::ServiceError::McpError(_))
        ));
    } else {
        handle.cancel(None).await.unwrap();
    }
    tokio::time::timeout(Duration::from_secs(2), cancelled.notified())
        .await
        .unwrap();
    if let Some(queued) = queued {
        tokio::time::timeout(Duration::from_secs(2), entered.notified())
            .await
            .unwrap();
        queued.cancel(None).await.unwrap();
        tokio::time::timeout(Duration::from_secs(2), cancelled.notified())
            .await
            .unwrap();
    }
    client.cancel().await.unwrap();
    upstream.cancel().await.unwrap();
    gateway_task.await.unwrap();
    upstream_task.await.unwrap();
}
