use crate::{Error, codex_models, route_access, route_enable, route_scope};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct KeyFacts {
    pub proxy: Option<route_scope::Facts>,
    pub enabled: Option<route_enable::Facts>,
    pub virtual_route: Option<route_access::VirtualRouteFacts>,
    pub role_loaded: bool,
    pub role: Option<String>,
    pub admin: Option<route_scope::Facts>,
}

#[derive(Clone, Copy, Deserialize, strum::VariantArray)]
pub enum JwtRead {
    Returned,
    Exception,
}

#[derive(Deserialize)]
pub struct AccessFacts {
    pub key: KeyFacts,
    pub jwt_token_present: bool,
    pub jwt: Option<JwtRead>,
}

#[derive(Deserialize)]
pub struct Credentials {
    pub bearer_present: bool,
    pub gateway_present: bool,
    pub equal: bool,
}

#[derive(Deserialize)]
pub struct Facts {
    pub model: Option<String>,
    pub native: bool,
    pub credentials: Option<Credentials>,
    pub access: Option<AccessFacts>,
    pub routing: Option<codex_models::Facts>,
    pub catalog_loaded: bool,
    pub slugs: Option<Vec<String>>,
}

#[derive(Deserialize)]
pub enum Request {
    Key(Box<KeyFacts>),
    Access(Box<AccessFacts>),
    Turn(Box<Facts>),
    Jwt(JwtRead),
    JwtToken {
        claims_present: bool,
        token_present: bool,
    },
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    Complete { allowed: bool },
    Failure { message: String },
    ReadProxy,
    ReadEnabled,
    ReadVirtual,
    ReadRole,
    ReadAdmin,
    ReadCredentials,
    ReadAccess,
    ReadRouting,
    ReadJwt,
    ReadCatalog,
}

fn complete(allowed: bool) -> Decision {
    Decision::Complete { allowed }
}

fn scope(facts: &route_scope::Facts) -> Result<Result<bool, String>, Error> {
    match route_scope::decide(facts) {
        route_scope::Decision::Allow => Ok(Ok(true)),
        route_scope::Decision::Deny => Ok(Ok(false)),
        route_scope::Decision::Failure { message } => Ok(Err(message)),
        _ => Err(Error::InvalidAnswer),
    }
}

pub fn key(facts: &KeyFacts) -> Result<Decision, Error> {
    let Some(proxy) = &facts.proxy else {
        return Ok(Decision::ReadProxy);
    };
    match scope(proxy)? {
        Ok(true) => (),
        Ok(false) => return Ok(complete(false)),
        Err(message) => return Ok(Decision::Failure { message }),
    }
    let Some(enabled) = &facts.enabled else {
        return Ok(Decision::ReadEnabled);
    };
    match route_enable::decide(enabled)? {
        route_enable::Decision::Allow => (),
        route_enable::Decision::Deny { .. } | route_enable::Decision::PropagateHttp => {
            return Ok(complete(false));
        }
        _ => return Err(Error::InvalidAnswer),
    }
    let Some(virtual_route) = &facts.virtual_route else {
        return Ok(Decision::ReadVirtual);
    };
    if !route_access::virtual_route(virtual_route)?.allowed {
        return Ok(complete(false));
    }
    if !facts.role_loaded {
        return Ok(Decision::ReadRole);
    }
    if facts.role.as_deref() == Some("proxy_admin") {
        return Ok(complete(true));
    }
    let Some(admin) = &facts.admin else {
        return Ok(Decision::ReadAdmin);
    };
    Ok(match scope(admin)? {
        Ok(allowed) => complete(allowed),
        Err(message) => Decision::Failure { message },
    })
}

pub fn access(facts: &AccessFacts) -> Result<Decision, Error> {
    match key(&facts.key)? {
        Decision::Complete { allowed: true } => (),
        Decision::Complete { allowed: false } => return Ok(complete(false)),
        _ => return Err(Error::InvalidAnswer),
    }
    if !facts.jwt_token_present {
        return Ok(complete(true));
    }
    Ok(match facts.jwt {
        None => Decision::ReadJwt,
        Some(read) => jwt(read),
    })
}

fn jwt(read: JwtRead) -> Decision {
    complete(matches!(read, JwtRead::Returned))
}

pub fn turn(facts: &Facts) -> Result<Decision, Error> {
    if facts.native {
        let Some(credentials) = &facts.credentials else {
            return Ok(Decision::ReadCredentials);
        };
        if !credentials.bearer_present || !credentials.gateway_present || credentials.equal {
            return Ok(complete(false));
        }
    }
    let Some(access_facts) = &facts.access else {
        return Ok(Decision::ReadAccess);
    };
    let Some(model) = &facts.model else {
        return Ok(complete(false));
    };
    let Some(routing) = &facts.routing else {
        return Ok(Decision::ReadRouting);
    };
    match codex_models::decide(routing)? {
        codex_models::Decision::Complete { allowed: true, .. } => (),
        codex_models::Decision::Complete { allowed: false, .. } => return Ok(complete(false)),
        _ => return Err(Error::InvalidAnswer),
    }
    match access(access_facts)? {
        Decision::Complete { allowed: true } => (),
        other => return Ok(other),
    }
    if !facts.catalog_loaded {
        return Ok(Decision::ReadCatalog);
    }
    Ok(complete(
        facts
            .slugs
            .as_ref()
            .is_some_and(|slugs| !slugs.contains(model)),
    ))
}

pub fn decide(request: Request) -> Result<Decision, Error> {
    match request {
        Request::Key(facts) => key(&facts),
        Request::Access(facts) => access(&facts),
        Request::Turn(facts) => turn(&facts),
        Request::Jwt(read) => Ok(jwt(read)),
        Request::JwtToken {
            claims_present,
            token_present,
        } => Ok(complete(claims_present && !token_present)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::absent(None, Decision::ReadCredentials)]
    #[case::no_bearer(Some(Credentials { bearer_present: false, gateway_present: true, equal: false }), complete(false))]
    #[case::no_gateway(Some(Credentials { bearer_present: true, gateway_present: false, equal: false }), complete(false))]
    #[case::equal(Some(Credentials { bearer_present: true, gateway_present: true, equal: true }), complete(false))]
    #[case::separate(Some(Credentials { bearer_present: true, gateway_present: true, equal: false }), Decision::ReadAccess)]
    fn credentials_precede_route_reads(
        #[case] credentials: Option<Credentials>,
        #[case] expected: Decision,
    ) {
        let facts = Facts {
            model: Some("model".into()),
            native: true,
            credentials,
            access: None,
            routing: None,
            catalog_loaded: false,
            slugs: None,
        };
        assert_eq!(turn(&facts).unwrap(), expected);
    }

    #[rstest]
    #[case::returned(JwtRead::Returned, true)]
    #[case::exception(JwtRead::Exception, false)]
    fn jwt_wrapper_uses_only_validator_outcome(#[case] read: JwtRead, #[case] allowed: bool) {
        assert_eq!(jwt(read), complete(allowed));
    }
}
