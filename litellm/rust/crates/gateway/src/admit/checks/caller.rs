use litellm_gateway_auth::AuthenticationMethod;

use super::{Admit, Authenticated, Reason, Refused};

/// Each way `Auth` verifies a caller is matched by name, so one `Auth` learns does not compile here
/// until it is named native or not. A caller it does not verify is over the error's name.
pub(super) fn caller(_: &Admit, authenticated: &Authenticated) -> Option<Reason> {
    let over: &'static str = match authenticated {
        Ok(request) => match request.caller().authentication().method {
            AuthenticationMethod::MasterKey | AuthenticationMethod::VirtualKey => return None,
            AuthenticationMethod::External(_) => "external",
        },
        Err(error) => error.into(),
    };
    Some(Reason::new(Refused::Credential, over))
}
