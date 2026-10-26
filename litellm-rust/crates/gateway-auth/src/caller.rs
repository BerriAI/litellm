use axum::{extract::FromRequestParts, http::request::Parts};

use crate::Error;

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
pub enum AccessRequest {
    Model { name: String, deployment: String },
    Mcp { server: String, action: McpAction },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Caller {
    Master,
}

impl Caller {
    pub fn authorize(&self, _: &AccessRequest) -> Result<(), Error> {
        match self {
            Self::Master => Ok(()),
        }
    }

    pub fn owner(&self) -> String {
        match self {
            Self::Master => "master".into(),
        }
    }
}

impl<S: Send + Sync> FromRequestParts<S> for Caller {
    type Rejection = Error;

    async fn from_request_parts(parts: &mut Parts, _: &S) -> Result<Self, Error> {
        parts
            .extensions
            .get::<Self>()
            .cloned()
            .ok_or(Error::MissingIdentity)
    }
}
