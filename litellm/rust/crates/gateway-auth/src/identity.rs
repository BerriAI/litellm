use std::{sync::Arc, time::SystemTime};

use crate::authorization::Permissions;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrincipalKind {
    Human,
    Service,
    System,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Principal {
    authority: String,
    subject: String,
    kind: PrincipalKind,
}

impl Principal {
    pub fn new(authority: String, subject: String, kind: PrincipalKind) -> Self {
        Self {
            authority,
            subject,
            kind,
        }
    }

    pub fn authority(&self) -> &str {
        &self.authority
    }
    pub fn subject(&self) -> &str {
        &self.subject
    }
    pub fn kind(&self) -> PrincipalKind {
        self.kind
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AuthenticationMethod {
    MasterKey,
    External(String),
}

#[derive(Clone, Debug)]
pub struct Authentication {
    pub method: AuthenticationMethod,
    pub verifier: String,
    pub credential_id: String,
    pub expires_at: Option<SystemTime>,
}

pub struct VerifiedIdentity {
    pub principal: Principal,
    pub authentication: Authentication,
    pub restrictions: Permissions,
}

pub struct ResolvedIdentity {
    pub principal: Principal,
    pub permissions: Permissions,
}

#[derive(Clone, Debug)]
pub struct AuthenticatedCaller {
    verified_principal: Principal,
    principal: Principal,
    authentication: Authentication,
    permissions: Permissions,
    restrictions: Permissions,
}

impl AuthenticatedCaller {
    pub(crate) fn new(verified: VerifiedIdentity, resolved: ResolvedIdentity) -> Self {
        Self {
            verified_principal: verified.principal,
            principal: resolved.principal,
            authentication: verified.authentication,
            permissions: resolved.permissions,
            restrictions: verified.restrictions,
        }
    }

    pub fn verified_principal(&self) -> &Principal {
        &self.verified_principal
    }

    pub fn principal(&self) -> &Principal {
        &self.principal
    }
    pub fn authentication(&self) -> &Authentication {
        &self.authentication
    }
    pub fn permissions(&self) -> &Permissions {
        &self.permissions
    }
    pub fn restrictions(&self) -> &Permissions {
        &self.restrictions
    }
}

pub type SharedCaller = Arc<AuthenticatedCaller>;
