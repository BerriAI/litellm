use litellm_gateway_mcp::RelayClient;
use rmcp::{
    ClientHandler, ErrorData, RoleClient, RoleServer, ServerHandler, ServiceExt,
    model::*,
    service::{NotificationContext, RequestContext},
};
use rstest::rstest;
use serde_json::json;
use tokio::sync::mpsc;

struct Downstream(mpsc::UnboundedSender<ProgressNotificationParam>);

impl ClientHandler for Downstream {
    fn get_info(&self) -> ClientConfig {
        ClientConfig::new(
            ClientCapabilities::builder().enable_elicitation().build(),
            Implementation::new("fixture", "1"),
        )
    }

    async fn create_elicitation(
        &self,
        _: ElicitRequestParams,
        _: RequestContext<RoleClient>,
    ) -> Result<ElicitResult, ErrorData> {
        Ok(
            serde_json::from_value(json!({"action":"accept", "content":{"confirmed":true}}))
                .unwrap(),
        )
    }

    async fn on_progress(
        &self,
        params: ProgressNotificationParam,
        _: NotificationContext<RoleClient>,
    ) {
        self.0.send(params).unwrap();
    }
}

struct AsksForInput;

impl ServerHandler for AsksForInput {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
    }

    async fn call_tool(
        &self,
        _: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let progress =
            ProgressNotificationParam::new(context.meta.get_progress_token().unwrap(), 1.0);
        context.peer.notify_progress(progress).await.unwrap();
        let question = serde_json::from_value(json!({"message":"Confirm", "requestedSchema":{"type":"object","properties":{"confirmed":{"type":"boolean"}}}})).unwrap();
        let answer = context.peer.create_elicitation(question).await.unwrap();
        Ok(CallToolResult::success(vec![ContentBlock::text(
            serde_json::to_string(&answer).unwrap(),
        )])
        .into())
    }
}

struct Gateway;

impl ServerHandler for Gateway {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let (server, client) = tokio::io::duplex(65536);
        let worker = tokio::spawn(async move {
            AsksForInput
                .serve(server)
                .await
                .unwrap()
                .waiting()
                .await
                .unwrap()
        });
        let upstream = RelayClient::for_request(&context)
            .serve(client)
            .await
            .unwrap();
        let result = upstream.call_tool(request).await.unwrap();
        upstream.cancel().await.unwrap();
        worker.await.unwrap();
        Ok(result.into())
    }
}

#[rstest]
#[tokio::test]
async fn relays_elicitation_and_maps_progress_to_downstream_token() {
    let (server, client) = tokio::io::duplex(65536);
    let worker = tokio::spawn(async move {
        Gateway
            .serve(server)
            .await
            .unwrap()
            .waiting()
            .await
            .unwrap()
    });
    let (sender, mut progress) = mpsc::unbounded_channel();
    let client = Downstream(sender).serve(client).await.unwrap();
    let request = CallToolRequest::new(CallToolRequestParams::new("confirm"));
    let handle = client
        .send_request_with_option(request.into(), Default::default())
        .await
        .unwrap();
    let expected_token = handle.progress_token.clone();
    let result = handle.await_response().await.unwrap();
    let ServerResult::CallToolResult(result) = result else {
        panic!("expected tool result")
    };
    let content: serde_json::Value =
        serde_json::from_str(&result.content[0].as_text().unwrap().text).unwrap();
    assert_eq!(content["content"]["confirmed"], true);
    let update = progress.recv().await.unwrap();
    assert_eq!(update.progress_token, expected_token);
    assert_eq!(update.progress, 1.0);
    client.cancel().await.unwrap();
    worker.await.unwrap();
}
