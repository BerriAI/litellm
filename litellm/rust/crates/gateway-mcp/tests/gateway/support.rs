use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use axum::Router;
use litellm_gateway_mcp::{HttpConfig, NativeGateway, Registry, Server, ServerInfo, router};
use rmcp::{
    ErrorData, RoleClient, RoleServer, ServerHandler, ServiceExt,
    model::*,
    service::{RequestContext, RunningService},
};
use rstest::fixture;
use serde_json::json;

pub struct Upstream {
    pub calls: Arc<AtomicUsize>,
}

impl ServerHandler for Upstream {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_prompts()
                .enable_resources()
                .build(),
        )
    }

    async fn list_tools(
        &self,
        params: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        let second = params.and_then(|params| params.cursor).is_some();
        let names = if second {
            vec!["hidden", "fail"]
        } else {
            vec!["echo-value"]
        };
        let tools = names
            .into_iter()
            .map(|name| {
                Tool::new(
                    name,
                    name,
                    Arc::new(serde_json::from_value(json!({"type":"object"})).unwrap()),
                )
            })
            .collect();
        let mut result = ListToolsResult::with_all_items(tools);
        result.next_cursor = (!second).then(|| "page-2".into());
        Ok(result)
    }

    async fn call_tool(
        &self,
        params: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if params.name == "fail" {
            return Err(ErrorData::invalid_params("fixture tool failed", None));
        }
        Ok(CallToolResult::success(vec![ContentBlock::text(
            json!({"name":params.name, "arguments":params.arguments, "meta":context.meta})
                .to_string(),
        )])
        .into())
    }

    async fn list_prompts(
        &self,
        _: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListPromptsResult, ErrorData> {
        Ok(ListPromptsResult::with_all_items(vec![Prompt::new(
            "review-code",
            None::<String>,
            None,
        )]))
    }

    async fn get_prompt(
        &self,
        params: GetPromptRequestParams,
        _: RequestContext<RoleServer>,
    ) -> Result<GetPromptResponse, ErrorData> {
        Ok(serde_json::from_value::<GetPromptResult>(
            json!({"messages": [{"role":"user", "content":{"type":"text", "text":params.name}}]}),
        )
        .unwrap()
        .into())
    }

    async fn list_resources(
        &self,
        _: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, ErrorData> {
        Ok(ListResourcesResult::with_all_items(vec![Resource::new(
            "fixture://document",
            "document",
        )]))
    }

    async fn list_resource_templates(
        &self,
        _: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, ErrorData> {
        Ok(ListResourceTemplatesResult::with_all_items(vec![
            serde_json::from_value(json!({"name":"files", "uriTemplate":"fixture://{file}"}))
                .unwrap(),
        ]))
    }

    async fn read_resource(
        &self,
        params: ReadResourceRequestParams,
        _: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResponse, ErrorData> {
        Ok(serde_json::from_value::<ReadResourceResult>(
            json!({"contents":[{"uri":params.uri,"text":"fixture contents"}]}),
        )
        .unwrap()
        .into())
    }
}

pub struct Harness {
    pub gateway: Arc<NativeGateway>,
    pub app: Router,
    pub calls: Arc<AtomicUsize>,
    pub connections: Vec<RunningService<RoleClient, ()>>,
}

#[fixture]
pub async fn harness() -> Harness {
    let calls = Arc::new(AtomicUsize::new(0));
    let connections = futures_util::future::join_all((0..2).map(|_| {
        let calls = calls.clone();
        async move {
            let (server, client) = tokio::io::duplex(65536);
            tokio::spawn(async move {
                Upstream { calls }
                    .serve(server)
                    .await
                    .unwrap()
                    .waiting()
                    .await
                    .unwrap();
            });
            ().serve(client).await.unwrap()
        }
    }))
    .await;
    let servers: Vec<_> = connections
        .iter()
        .zip(["alpha", "alpha-beta"])
        .map(|(connection, name)| Server {
            info: ServerInfo {
                server_id: format!("id-{name}"),
                server_name: name.into(),
                alias: None,
            },
            peer: connection.peer().clone(),
            allowed_tools: Some(Arc::from(["echo-value".into(), "fail".into()])),
        })
        .collect();
    let gateway = Arc::new(NativeGateway::new(
        Arc::new(Registry::new(servers).unwrap()),
        Duration::from_secs(2),
    ));
    let app = router(gateway.clone(), HttpConfig::default());
    Harness {
        gateway,
        app,
        calls,
        connections,
    }
}

pub fn context(server: Option<&str>) -> litellm_gateway_mcp::Context {
    litellm_gateway_mcp::Context {
        parts: http::Request::new(()).into_parts().0,
        server: server.map(str::to_owned),
        mcp: None,
    }
}
