use std::sync::Arc;

use crate::{
    AuthFuture, AuthenticatedCaller, Clock, Error, SharedCaller,
    keys::{AllowedRoutes, KeyModels},
};

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
    Route {
        method: String,
        path: String,
    },
    /// `team_id` is the team owning the deployment (`model_info.team_id`), `None` when shared.
    Model {
        name: String,
        deployment: String,
        team_id: Option<String>,
    },
    Mcp {
        server: String,
        action: McpAction,
    },
    Ui(UiAction),
}

#[derive(Clone, Debug, Default)]
pub enum Permissions {
    All,
    #[default]
    None,
    Only(Arc<[AccessRequest]>),
    /// A virtual key: no MCP or UI access.
    VirtualKey(KeyGrant),
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KeyGrant {
    pub models: KeyModels,
    pub routes: AllowedRoutes,
    /// The key's team. A deployment owned by a team serves only that team's keys, as the
    /// proxy's `filter_team_based_models` does; a teamless key gets shared deployments only.
    pub team_id: Option<String>,
    pub profile: Arc<crate::keys::KeyProfile>,
}

impl KeyGrant {
    pub fn may_use_deployment(&self, owner: Option<&str>) -> bool {
        owner.is_none() || owner == self.team_id.as_deref()
    }
}

impl Permissions {
    pub fn allows(&self, request: &AccessRequest) -> bool {
        match (self, request) {
            (Self::All, _) => true,
            (Self::None, _) => false,
            (Self::Only(allowed), _) => allowed.contains(request),
            (Self::VirtualKey(grant), AccessRequest::Route { path, .. }) => {
                grant.routes.allows(path)
            }
            (Self::VirtualKey(grant), AccessRequest::Model { name, team_id, .. }) => {
                grant.models.check(name).is_ok() && grant.may_use_deployment(team_id.as_deref())
            }
            (Self::VirtualKey(_), AccessRequest::Mcp { .. } | AccessRequest::Ui(_)) => false,
        }
    }

    /// Whether routing may pick a deployment owned by `owner` for this caller. Only a virtual
    /// key is limited; every other caller is decided by [`Permissions::allows`] alone.
    pub fn may_use_deployment(&self, owner: Option<&str>) -> bool {
        match self {
            Self::VirtualKey(grant) => grant.may_use_deployment(owner),
            Self::All | Self::None | Self::Only(_) => true,
        }
    }

    fn denial(&self, request: &AccessRequest) -> Error {
        match (self, request) {
            (Self::VirtualKey(grant), AccessRequest::Route { path, .. }) => {
                Error::RouteNotAllowed {
                    allowed: grant.routes.python_repr(),
                    route: path.clone(),
                }
            }
            (Self::VirtualKey(grant), AccessRequest::Model { name, .. }) => {
                match grant.models.check(name) {
                    Err(by) => Error::ModelAccessDenied {
                        model: name.clone(),
                        by,
                    },
                    Ok(()) => Error::Forbidden,
                }
            }
            _ => Error::Forbidden,
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

    /// Whether routing may pick a deployment owned by `owner` (`model_info.team_id`) for this
    /// caller; [`AuthenticatedRequest::authorize`] still checks the one it picks.
    pub fn may_use_deployment(&self, owner: Option<&str>) -> bool {
        self.caller.restrictions().may_use_deployment(owner)
            && self.caller.permissions().may_use_deployment(owner)
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
        let restrictions = self.caller.restrictions();
        if !restrictions.allows(&request) {
            return Err(restrictions.denial(&request));
        }
        if !self.caller.permissions().allows(&request) {
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
