use jiff::Timestamp;

const INVALID_VIRTUAL_KEY_ERROR_MESSAGE: &str = "LiteLLM Virtual Key expected";
const INVALID_API_KEY_ERROR_MESSAGE: &str = "Authentication Error, Invalid API key provided.";
const JWT_AUTH_DISABLED_HINT: &str = " This key has the structure of a JWT, but JWT auth is not enabled on this proxy, so it \
was treated as a virtual key. Set `enable_jwt_auth: true` under `general_settings` in your proxy config to authenticate \
with JWTs.";

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum KeyDenial {
    #[error("Authentication Error, No api key passed in.")]
    MissingKey,
    #[error("Authentication Error, Malformed API Key passed in. Ensure Key has `Bearer ` prefix.")]
    MalformedKey,
    #[error("'allow_user_auth' not set or set to False")]
    UserAuthDisabled,
    #[error(
        "Authentication Error - Expired Key. Key Expiry time {} and current time {}",
        utc(*expires),
        utc(*now)
    )]
    Expired {
        expires: Timestamp,
        now: Timestamp,
        key: String,
    },
    #[error("Authentication Error, Tried to access route={route}, which is only for MASTER KEY")]
    MasterKeyOnlyRoute { route: String },
    #[error("No connected db.")]
    NoDb,
    #[error(
        "{INVALID_VIRTUAL_KEY_ERROR_MESSAGE}. Received={received}, expected to start with 'sk-'.{}",
        if *jwt_hint { JWT_AUTH_DISABLED_HINT } else { "" }
    )]
    NotVirtualKey { received: String, jwt_hint: bool },
    #[error("{INVALID_API_KEY_ERROR_MESSAGE}")]
    NotFound,
    #[error("Authentication Error, Key is blocked. Update via `/key/unblock` if you're an admin.")]
    Blocked,
    #[error(
        "Authentication Error, User={user_id} has been deactivated via SCIM. Keys owned by this user cannot be used."
    )]
    UserDeactivated { user_id: String },
    #[error("Invalid API key, no token associated")]
    NoToken,
}

fn utc(time: Timestamp) -> String {
    let precision = match time.subsec_nanosecond() {
        0 => 0,
        nanos if nanos % 1_000_000 == 0 => 3,
        nanos if nanos % 1_000 == 0 => 6,
        _ => 9,
    };
    format!("{time:.precision$}")
}
