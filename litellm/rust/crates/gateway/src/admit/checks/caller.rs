use litellm_gateway_auth_core::Caller;

use super::{Admit, Authentication, Reason, Refused};

/// Each caller the core hands back is matched by name, with no catch-all, so a caller the native
/// host learns to serve does not compile here until it is named native or not. A credential the
/// core did not authenticate is over the error's name.
pub(super) fn caller(_: &Admit, authentication: &Authentication) -> Option<Reason> {
    match authentication {
        Ok(caller) => match &**caller {
            Caller::MasterKey | Caller::Key(_) => None,
        },
        Err(error) => Some(Reason::new(
            Refused::Credential,
            <&'static str>::from(error),
        )),
    }
}
