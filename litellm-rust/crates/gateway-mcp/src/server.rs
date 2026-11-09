use std::sync::Arc;

use http::{Request, request::Parts};
use rmcp::{ErrorData, RoleServer, ServerHandler, model::*, service::RequestContext};

use crate::{Context, Error, Operation, Operations};

#[derive(Clone)]
pub struct McpServer {
    operations: Arc<dyn Operations>,
    info: ServerConfig,
}

impl McpServer {
    pub fn new(operations: Arc<dyn Operations>) -> Self {
        Self {
            operations,
            info: ServerConfig::new(
                ServerCapabilities::builder()
                    .enable_tools()
                    .enable_prompts()
                    .enable_resources()
                    .build(),
            )
            .with_server_info(Implementation::new("litellm-mcp-server", "1.0.0")),
        }
    }

    pub fn with_info(self, info: ServerConfig) -> Self {
        Self { info, ..self }
    }

    async fn execute(
        &self,
        operation: Operation,
        mcp: RequestContext<RoleServer>,
    ) -> Result<ServerResult, ErrorData> {
        self.operations
            .execute(operation, context(mcp))
            .await
            .map_err(Error::into_mcp)
    }
}

#[derive(Clone)]
pub(crate) struct ServerScope(pub String);

macro_rules! listing {
    ($method:ident, $operation:ident, $result:ident) => {
        async fn $method(
            &self,
            request: Option<PaginatedRequestParams>,
            context: RequestContext<RoleServer>,
        ) -> Result<$result, ErrorData> {
            match self
                .execute(Operation::$operation(request), context)
                .await?
            {
                ServerResult::$result(result) => Ok(result),
                _ => Err(Error::UnexpectedResult.into_mcp()),
            }
        }
    };
}

impl ServerHandler for McpServer {
    fn get_info(&self) -> ServerConfig {
        self.info.clone()
    }

    async fn initialize(
        &self,
        request: InitializeRequestParams,
        mcp: RequestContext<RoleServer>,
    ) -> Result<InitializeResult, ErrorData> {
        mcp.peer.set_peer_info(request.clone());
        self.operations
            .authorize(context(mcp))
            .await
            .map_err(Error::into_mcp)?;
        self.negotiate_initialize(&request)
    }

    async fn discover(&self, mcp: RequestContext<RoleServer>) -> Result<DiscoverResult, ErrorData> {
        self.operations
            .authorize(context(mcp))
            .await
            .map_err(Error::into_mcp)?;
        Ok(DiscoverResult::from_server_info(
            self.supported_protocol_versions().into_owned(),
            self.get_info(),
        ))
    }

    listing!(list_tools, ListTools, ListToolsResult);
    listing!(list_prompts, ListPrompts, ListPromptsResult);
    listing!(list_resources, ListResources, ListResourcesResult);
    listing!(
        list_resource_templates,
        ListResourceTemplates,
        ListResourceTemplatesResult
    );

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        match self.execute(Operation::CallTool(request), context).await? {
            ServerResult::CallToolResult(result) => Ok(result.into()),
            ServerResult::InputRequiredResult(result) => {
                Ok(CallToolResponse::InputRequired(result))
            }
            _ => Err(Error::UnexpectedResult.into_mcp()),
        }
    }

    async fn get_prompt(
        &self,
        request: GetPromptRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<GetPromptResponse, ErrorData> {
        match self.execute(Operation::GetPrompt(request), context).await? {
            ServerResult::GetPromptResult(result) => Ok(result.into()),
            ServerResult::InputRequiredResult(result) => {
                Ok(GetPromptResponse::InputRequired(result))
            }
            _ => Err(Error::UnexpectedResult.into_mcp()),
        }
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResponse, ErrorData> {
        match self
            .execute(Operation::ReadResource(request), context)
            .await?
        {
            ServerResult::ReadResourceResult(result) => Ok(result.into()),
            ServerResult::InputRequiredResult(result) => {
                Ok(ReadResourceResponse::InputRequired(result))
            }
            _ => Err(Error::UnexpectedResult.into_mcp()),
        }
    }
}

fn context(mcp: RequestContext<RoleServer>) -> Context {
    let parts = mcp
        .extensions
        .get::<Parts>()
        .cloned()
        .unwrap_or_else(|| Request::new(()).into_parts().0);
    let server = parts
        .extensions
        .get::<ServerScope>()
        .map(|scope| scope.0.clone());
    Context {
        parts,
        server,
        mcp: Some(mcp),
    }
}
