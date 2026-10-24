use std::sync::Arc;

use axum::{extract::Request, middleware::Next, response::Response};
use rmcp::model::ClientRequest;

use litellm_gateway_auth::{AccessRequest, AuthenticatedRequest, McpAction};

use crate::{Authorization, GatewayFuture, OperationAuthorizer, ServerInfo, SessionOwner};

pub async fn bind_caller(
    identity: AuthenticatedRequest,
    mut request: Request,
    next: Next,
) -> Response {
    request
        .extensions_mut()
        .insert(SessionOwner(identity.caller().session_owner()));
    request
        .extensions_mut()
        .insert(Authorization(Arc::new(McpPolicy(identity))));
    next.run(request).await
}

struct McpPolicy(AuthenticatedRequest);

impl OperationAuthorizer for McpPolicy {
    fn authorize<'a>(
        &'a self,
        server: &'a ServerInfo,
        request: Option<&'a ClientRequest>,
    ) -> GatewayFuture<'a, ()> {
        Box::pin(async move {
            let action = match request {
                None => McpAction::Connect,
                Some(ClientRequest::ListToolsRequest(_)) => McpAction::ListTools,
                Some(ClientRequest::CallToolRequest(request)) => {
                    McpAction::CallTool(request.params.name.to_string())
                }
                Some(ClientRequest::ListPromptsRequest(_)) => McpAction::ListPrompts,
                Some(ClientRequest::GetPromptRequest(request)) => {
                    McpAction::GetPrompt(request.params.name.clone())
                }
                Some(ClientRequest::ListResourcesRequest(_)) => McpAction::ListResources,
                Some(ClientRequest::ListResourceTemplatesRequest(_)) => {
                    McpAction::ListResourceTemplates
                }
                Some(ClientRequest::ReadResourceRequest(request)) => {
                    McpAction::ReadResource(request.params.uri.clone())
                }
                Some(_) => return Err(crate::Error::Forbidden),
            };
            let access = AccessRequest::Mcp {
                server: server.server_id.clone(),
                action,
            };
            self.0
                .authorize(access.clone())
                .await?
                .consume(self.0.caller(), &access)?;
            Ok(())
        })
    }
}
