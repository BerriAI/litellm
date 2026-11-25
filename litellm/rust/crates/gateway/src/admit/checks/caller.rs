use litellm_config::GeneralSettings;
use litellm_gateway_auth::Error;
use litellm_gateway_auth_core::Credential;

use super::{Admit, Authentication, Reason, Refused};

/// Each caller the core hands back is matched by name, with no catch-all, so a caller the native
/// host learns to serve does not compile here until it is named native or not. A credential the
/// core did not authenticate is over the error's name. One the core refused is denied when the
/// proxy takes nothing but a key, since the native app's own authentication refuses it again. One
/// the core could not judge is not: a key this server does not serve, or an authentication that
/// failed.
pub(super) fn caller(admit: &Admit, authentication: &Authentication) -> Option<Reason> {
    match authentication {
        Ok(caller) => match &caller.credential {
            Credential::MasterKey | Credential::Key { .. } => None,
        },
        Err(error) => {
            let over = <&'static str>::from(error);
            Some(match error {
                Error::InvalidToken | Error::InvalidApiKey | Error::Expired | Error::Forbidden
                    if takes_only_a_key(&admit.config.general_settings) =>
                {
                    Reason::denied(Refused::Credential, over)
                }
                Error::InvalidToken
                | Error::InvalidApiKey
                | Error::Expired
                | Error::Forbidden
                | Error::NotNative
                | Error::Unavailable(_)
                | Error::Unconfigured
                | Error::MissingIdentity => Reason::new(Refused::Credential, over),
            })
        }
    }
}

/// With JWT auth on or routes made public, the proxy may take a request the key path refuses
/// another way, which the native app does not. Every other credential path is a setting that
/// already sends each request to Python.
fn takes_only_a_key(general: &GeneralSettings) -> bool {
    !general.enable_jwt_auth && general.public_routes.is_none()
}
