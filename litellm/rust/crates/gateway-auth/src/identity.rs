use std::{sync::Arc, time::SystemTime};

use sha2::{Digest, Sha256};

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
    Session,
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

    pub fn session_owner(&self) -> String {
        let fields = [
            self.verified_principal.authority(),
            self.verified_principal.subject(),
            self.principal.authority(),
            self.principal.subject(),
            self.authentication.verifier.as_str(),
            self.authentication.credential_id.as_str(),
        ];
        let digest = fields.into_iter().fold(Sha256::new(), |mut digest, field| {
            digest.update((field.len() as u64).to_be_bytes());
            digest.update(field.as_bytes());
            digest
        });
        format!("{:x}", digest.finalize())
    }
}

pub type SharedCaller = Arc<AuthenticatedCaller>;
