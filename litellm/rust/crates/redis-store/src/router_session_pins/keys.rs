use litellm_python_compat::{Value, json, number::hex_int};
use sha2::{Digest, Sha256};

use crate::Error;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PinKey {
    /// Deployment affinity by caller.
    Deployment { model_group: String, caller: String },
    /// Deployment affinity by session, scoped to the caller when there is one.
    DeploymentSession {
        model_group: String,
        caller: Option<String>,
        session_id: String,
    },
    /// The complexity router's session pin, whose caller is already the API key hash.
    TierSession {
        router: String,
        caller: Option<String>,
        session_id: String,
    },
    /// The complexity router's model within a tier. `tiers` hashes the router's tier config, and
    /// `replay_fallback` is the model a replayed turn was pinned to.
    TierModel {
        router: String,
        tiers: String,
        caller: Option<String>,
        session_id: String,
        tier: String,
        replay_fallback: Option<String>,
    },
}

impl PinKey {
    pub fn spelling(&self) -> Result<String, Error> {
        Ok(match self {
            Self::Deployment {
                model_group,
                caller,
            } => format!("deployment_affinity:v1:{model_group}:{}", hashed(caller)),
            Self::DeploymentSession {
                model_group,
                caller,
                session_id,
            } => {
                let scope = caller
                    .as_deref()
                    .map_or_else(|| "unscoped".to_owned(), hashed);
                format!("deployment_affinity:v1:session:{model_group}:{scope}:{session_id}")
            }
            Self::TierSession {
                router,
                caller,
                session_id,
            } => {
                let scope = caller.as_deref().filter(|caller| !caller.is_empty());
                format!(
                    "complexity_router_session_affinity:v1:{router}:{}:{session_id}",
                    scope.unwrap_or("unscoped")
                )
            }
            Self::TierModel {
                router,
                tiers,
                caller,
                session_id,
                tier,
                replay_fallback,
            } => {
                let text = |value: &str| Value::Str(value.to_owned());
                let identity = [
                    text(router),
                    text(tiers),
                    caller.as_deref().map_or(Value::None, text),
                    text(session_id),
                    text(tier),
                ]
                .into_iter()
                .chain(
                    replay_fallback
                        .as_deref()
                        .map(|model| Value::List(vec![text("replay_fallback"), text(model)])),
                )
                .collect();
                format!(
                    "complexity_router_tier_model_affinity:v1:{}",
                    sha256_hex(&json::dumps(&Value::List(identity))?)
                )
            }
        })
    }
}

/// `_hash_user_key`: a caller that already reads as a SHA-256 hex digest to `int(text, 16)` keeps
/// its text lowercased, anything else is hashed.
fn hashed(caller: &str) -> String {
    match caller.chars().count() == 64 && hex_int(caller).is_some() {
        true => caller.to_lowercase(),
        false => sha256_hex(caller),
    }
}

fn sha256_hex(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::PinKey;

    fn text(value: &str) -> String {
        value.to_owned()
    }

    // Expected values are main's DeploymentAffinityCheck.get_affinity_cache_key and
    // get_session_affinity_cache_key, and ComplexityRouter's session and tier model keys
    #[rstest]
    #[case::caller_hashed(
        PinKey::Deployment { model_group: text("gpt-4o"), caller: text("sk-caller") },
        "deployment_affinity:v1:gpt-4o:8b6712790a2089c67aa97a2d80022df18cc65c7814350e33baebe79aab508891"
    )]
    #[case::caller_already_a_digest_is_lowercased(
        PinKey::Deployment { model_group: text("gpt-4o"), caller: "A".repeat(64) },
        "deployment_affinity:v1:gpt-4o:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    )]
    #[case::caller_int_reads_as_hex_is_kept(
        PinKey::Deployment { model_group: text("gpt-4o"), caller: format!("0x{}", "f".repeat(62)) },
        "deployment_affinity:v1:gpt-4o:0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    )]
    #[case::user_id_caller(
        PinKey::Deployment { model_group: text("gpt-4o"), caller: text("user_id:u-1") },
        "deployment_affinity:v1:gpt-4o:a8ae9346b4f3e606434b5bf77a2ba2ea81313df4283d69d85bcef9e93a78a4f8"
    )]
    #[case::session_without_caller(
        PinKey::DeploymentSession { model_group: text("gpt-4o"), caller: None, session_id: text("sess-1") },
        "deployment_affinity:v1:session:gpt-4o:unscoped:sess-1"
    )]
    #[case::session_with_caller(
        PinKey::DeploymentSession { model_group: text("gpt-4o"), caller: Some(text("sk-caller")), session_id: text("sess-1") },
        "deployment_affinity:v1:session:gpt-4o:8b6712790a2089c67aa97a2d80022df18cc65c7814350e33baebe79aab508891:sess-1"
    )]
    #[case::session_with_empty_caller_is_hashed(
        PinKey::DeploymentSession { model_group: text("gpt-4o"), caller: Some(String::new()), session_id: text("sess-1") },
        "deployment_affinity:v1:session:gpt-4o:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855:sess-1"
    )]
    #[case::tier_session(
        PinKey::TierSession { router: text("smart"), caller: Some(text("key-hash")), session_id: text("sess-1") },
        "complexity_router_session_affinity:v1:smart:key-hash:sess-1"
    )]
    #[case::tier_session_with_empty_caller_is_unscoped(
        PinKey::TierSession { router: text("smart"), caller: Some(String::new()), session_id: text("sess-1") },
        "complexity_router_session_affinity:v1:smart:unscoped:sess-1"
    )]
    #[case::tier_model(
        PinKey::TierModel {
            router: text("smart"),
            tiers: text("cfg"),
            caller: Some(text("sk-caller")),
            session_id: text("sess-1"),
            tier: text("SIMPLE"),
            replay_fallback: None,
        },
        "complexity_router_tier_model_affinity:v1:d93f3c0fe46c4ef59f1e45d1b2aa37455b2663b05deb14d4afb3339bc2094f51"
    )]
    #[case::tier_model_replay_without_caller(
        PinKey::TierModel {
            router: text("smart"),
            tiers: text("cfg"),
            caller: None,
            session_id: text("séss"),
            tier: text("COMPLEX"),
            replay_fallback: Some(text("m-a")),
        },
        "complexity_router_tier_model_affinity:v1:c83437032b0b0d7d8229f7aaf444096a83510ee630e5a510e4770b0f6f03a8d3"
    )]
    fn a_pin_key_is_spelled_as_main_spells_it(#[case] key: PinKey, #[case] expected: &str) {
        assert_eq!(key.spelling().unwrap(), expected);
    }
}
