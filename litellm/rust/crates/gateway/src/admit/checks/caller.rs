use litellm_callbacks_model_capacity::NAME as MODEL_CAPACITY;
use litellm_gateway_auth_core::{Caller, StoredKey};

use super::{Admit, Authentication, Reason, Refused};

/// `dynamic_rate_limiter_v3` shares a model's capacity among the teams that are calling it, and
/// the native callback tells it of no team, so a key in a team goes to Python under it.
fn team_under_model_capacity(admit: &Admit, team_id: Option<&str>) -> Option<Reason> {
    let callbacks = admit.config.litellm_settings.callbacks.iter().flatten();
    let shared = callbacks
        .into_iter()
        .any(|callback| callback.as_str() == Some(MODEL_CAPACITY));
    (shared && team_id.is_some()).then(|| Reason::new(Refused::Credential, "team_model_capacity"))
}

/// Each caller the core hands back is matched by name, with no catch-all, so a caller the native
/// host learns to serve does not compile here until it is named native or not. A credential the
/// core did not authenticate is over the error's name.
pub(super) fn caller(admit: &Admit, authentication: &Authentication) -> Option<Reason> {
    match authentication {
        Ok(caller) => match &**caller {
            Caller::MasterKey => None,
            Caller::Key(context) => {
                team_under_model_capacity(admit, context.key.record().team_id.as_deref())
            }
        },
        Err(error) => Some(Reason::new(
            Refused::Credential,
            <&'static str>::from(error),
        )),
    }
}
