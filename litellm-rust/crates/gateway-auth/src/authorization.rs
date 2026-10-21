use std::sync::Arc;

use crate::{AuthFuture, AuthenticatedCaller, Clock, Error, SharedCaller};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum McpAction {
    Connect,
    ListTools,
    CallTool(String),
    ListPrompts,
    GetPrompt(String),
    ListResources,
    ListResourceTemplates,
    ReadResource(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UiAction {
    SessionInfo,
    Logout,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AccessRequest {
    Route { method: String, path: String },
    Model { name: String, deployment: String },
    Mcp { server: String, action: McpAction },
    Ui(UiAction),
}

#[derive(Clone, Debug, Default)]
pub enum Permissions {
    All,
    #[default]
    None,
    Only(Arc<[AccessRequest]>),
}

impl Permissions {
    pub fn allows(&self, request: &AccessRequest) -> bool {
        match self {
            Self::All => true,
            Self::None => false,
            Self::Only(allowed) => allowed.contains(request),
        }
    }
}

pub trait Authorizer: Send + Sync {
    fn authorize<'a>(
        &'a self,
        caller: &'a AuthenticatedCaller,
        request: &'a AccessRequest,
    ) -> AuthFuture<'a, ()>;
}

pub struct NoAdditionalPolicy;

impl Authorizer for NoAdditionalPolicy {
    fn authorize<'a>(
        &'a self,
        _: &'a AuthenticatedCaller,
        _: &'a AccessRequest,
    ) -> AuthFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }
}

#[derive(Clone)]
pub struct AuthenticatedRequest {
    caller: SharedCaller,
    authorizer: Arc<dyn Authorizer>,
    clock: Arc<dyn Clock>,
}

impl AuthenticatedRequest {
    pub(crate) fn new(
        caller: SharedCaller,
        authorizer: Arc<dyn Authorizer>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            caller,
            authorizer,
            clock,
        }
    }

    pub fn caller(&self) -> &AuthenticatedCaller {
        &self.caller
    }

    pub async fn authorize(&self, request: AccessRequest) -> Result<AuthorizedOperation, Error> {
        if self
            .caller
            .authentication()
            .expires_at
            .is_some_and(|expiry| expiry <= self.clock.now())
        {
            return Err(Error::Expired);
        }
        if !self.caller.restrictions().allows(&request)
            || !self.caller.permissions().allows(&request)
        {
            return Err(Error::Forbidden);
        }
        self.authorizer.authorize(&self.caller, &request).await?;
        Ok(AuthorizedOperation {
            caller: self.caller.clone(),
            request,
        })
    }
}

pub struct AuthorizedOperation {
    caller: SharedCaller,
    request: AccessRequest,
}

impl AuthorizedOperation {
    pub fn caller(&self) -> &AuthenticatedCaller {
        &self.caller
    }
    pub fn request(&self) -> &AccessRequest {
        &self.request
    }

    pub fn consume(
        self,
        caller: &AuthenticatedCaller,
        request: &AccessRequest,
    ) -> Result<(), Error> {
        if std::ptr::eq(self.caller.as_ref(), caller) && &self.request == request {
            Ok(())
        } else {
            Err(Error::Forbidden)
        }
    }
}
