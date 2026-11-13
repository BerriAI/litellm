use std::collections::HashSet;

use jiff::Timestamp;

use crate::{
    AuthHost, Error,
    host::{KeyRow, Rows},
};

mod error;
mod gate;
mod lookup;
mod token;
mod validate;

pub use error::KeyDenial;
use gate::{Gate, Precheck};
use lookup::Lookup;
use token::Shown;
pub use token::{KeyHash, MasterKey, Token};

const UI_TEAM_ID: &str = "litellm-dashboard";

#[derive(Clone, Debug, Default)]
pub struct KeySettings {
    /// [`HostFacts::master_key`](crate::HostFacts::master_key)
    pub master_key: Option<MasterKey>,
    /// [`HostFacts::has_db`](crate::HostFacts::has_db)
    pub has_db: bool,
    /// [`general_settings.allow_user_auth`](litellm_config::GeneralSettings::allow_user_auth)
    pub allow_user_auth: bool,
    /// [`HostFacts::ui_login`](crate::HostFacts::ui_login)
    pub ui_login: bool,
    /// [`general_settings.enable_jwt_auth`](litellm_config::GeneralSettings::enable_jwt_auth), so a JWT
    /// sent while it is off gets a hint in the denial.
    pub jwt_auth: bool,
    /// [`HostFacts::proxy_budget`](crate::HostFacts::proxy_budget)
    pub proxy_budget: bool,
    /// The `master_key_only_routes` group of [`RouteGroups`](crate::RouteGroups)
    pub master_key_only_routes: HashSet<String>,
    /// `MINIMUM_CUSTOM_KEY_LENGTH`: shorter keys are never echoed back in an error, not even in part.
    pub min_custom_key_length: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct KeyRequest<'a> {
    pub route: &'a str,
    /// [`Request::end_user_id`](crate::Request::end_user_id)
    pub end_user_id: Option<&'a str>,
    pub now: Timestamp,
    /// `RouteChecks.is_llm_api_route(route)`, where the key's max budget applies.
    pub is_llm_api_route: bool,
}

/// What the key path authenticates.
#[derive(Debug)]
pub enum Credential<'a, K> {
    /// The token the request presented, if it presented one.
    Presented(Option<&'a Token>),
    /// A key the caller already read, which a JWT mapped to that key authenticates as.
    Resolved(Keyed<K>),
}

/// The key fields the core decides on.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KeyRecord {
    pub token: Option<KeyHash>,
    pub blocked: bool,
    pub expires: Option<Timestamp>,
    pub is_proxy_admin: bool,
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub team_object_permission_id: Option<String>,
    pub project_id: Option<String>,
    /// The end user the key itself names, which wins over the request's.
    pub end_user_id: Option<String>,
    pub last_refreshed_at: Option<Timestamp>,
}

/// A host's key row with the fields the core decides on.
#[derive(Clone, Debug, PartialEq)]
pub struct Keyed<K> {
    pub record: KeyRecord,
    pub row: K,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Owner<U> {
    /// Deactivated through SCIM.
    pub deactivated: bool,
    pub row: U,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CachedTeam<T> {
    pub refreshed_at: Option<Timestamp>,
    pub row: T,
}

/// The team values a virtual key carries into the request.
#[derive(Clone, Debug, PartialEq)]
pub enum TeamGrant<R: Rows> {
    NoTeam,
    Loaded(R::Team),
    /// The UI team, or a team `get_team_object` did not report: the key's own copy, with the
    /// object permission it names.
    KeyCopy(Option<R::ObjectPermission>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct KeyContext<R: Rows> {
    pub key: R::Key,
    pub end_user: R::EndUser,
    pub owner: Option<R::User>,
    pub team: TeamGrant<R>,
    pub project: Option<R::Project>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum KeyIdentity<R: Rows> {
    /// No master key is configured, so any token, the one a pass-through endpoint read included, is let through.
    NoMasterKey {
        token: Option<Token>,
    },
    UserAuthRoute,
    CachedAdmin {
        key: R::Key,
        end_user: R::EndUser,
    },
    /// `hash` is the master key's hash, which the host caches the admin identity under.
    MasterKey {
        hash: KeyHash,
        end_user: R::EndUser,
    },
    VirtualKey(Box<KeyContext<R>>),
}

/// An error, with the key the request had resolved before it, which the host labels its failure with.
#[derive(Clone, Debug, PartialEq)]
pub struct Refused<R: Rows> {
    pub error: Box<Error>,
    pub key: Option<R::Key>,
}

impl<R: Rows> From<Error> for Refused<R> {
    fn from(error: Error) -> Self {
        Self {
            error: Box::new(error),
            key: None,
        }
    }
}

pub type KeyOutcome<R> = Result<KeyIdentity<R>, Refused<R>>;

fn refused<R: Rows>(error: impl Into<Error>, key: Option<&R::Key>) -> Refused<R> {
    Refused {
        error: Box::new(error.into()),
        key: key.cloned(),
    }
}

pub async fn authenticate<H: AuthHost>(
    host: &H,
    request: &KeyRequest<'_>,
    settings: &KeySettings,
    credential: Credential<'_, KeyRow<H>>,
) -> KeyOutcome<H::Rows> {
    let checked = gate::precheck::<H::Rows>(request.route, credential, settings)
        .map_err(|denial| refused(denial, None))?;
    let (presented, master_key, found) = match checked {
        Precheck::Decided(identity) => return Ok(identity),
        Precheck::Presented { token, master_key } => {
            let found = lookup::presented(host, token, request.end_user_id, settings).await?;
            (Some(token), master_key, found)
        }
        Precheck::Resolved {
            hash,
            key,
            master_key,
        } => {
            let found = lookup::resolved(host, hash, *key, request.end_user_id, settings).await?;
            (None, master_key, found)
        }
    };
    let Lookup {
        hash,
        end_user,
        key,
    } = found;
    let shown = presented.map_or(Shown::Stored(&hash), Shown::Presented);
    let key = match key {
        Some(admin) if admin.record.is_proxy_admin => {
            return match validate::expired(&admin.record, request.now, shown, settings) {
                Some(denial) => {
                    host.evict_key(&hash).await?;
                    Err(refused(denial, Some(&admin.row)))
                }
                None => Ok(KeyIdentity::CachedAdmin {
                    key: admin.row,
                    end_user,
                }),
            };
        }
        key => lookup::with_cached_team(host, key).await?,
    };
    let row = key.as_ref().map(|key| key.row.clone());
    match gate::gate(key, presented, master_key, request.route, settings) {
        Gate::MasterKey => Ok(KeyIdentity::MasterKey { hash, end_user }),
        Gate::Denied(denial) => Err(refused(denial, row.as_ref())),
        Gate::Held(key) => validate::validate(host, key, end_user, request, settings, shown).await,
        Gate::Unread => match host.db_key(&hash, &end_user).await? {
            Some(key) => {
                let shown = Shown::Stored(&hash);
                validate::validate(host, key, end_user, request, settings, shown).await
            }
            None => Err(refused(KeyDenial::NotFound, None)),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rstest::{fixture, rstest};

    use super::*;
    use crate::{
        Error,
        testing::{Labels, ScriptedHost},
    };

    const MASTER: &str = "sk-master";
    const NOW: Timestamp = Timestamp::constant(1_000, 0);

    fn at(second: i64) -> Timestamp {
        Timestamp::constant(second, 0)
    }

    #[fixture]
    fn settings() -> KeySettings {
        KeySettings {
            master_key: Some(MasterKey::new(MASTER)),
            has_db: true,
            ui_login: true,
            master_key_only_routes: HashSet::from(["/global/spend/reset".to_owned()]),
            min_custom_key_length: 16,
            ..KeySettings::default()
        }
    }

    fn virtual_key() -> KeyRecord {
        KeyRecord {
            token: Some(KeyHash::new("hashed")),
            ..KeyRecord::default()
        }
    }

    fn over_budget() -> crate::BudgetExceeded {
        crate::BudgetExceeded {
            current_cost: 2.0,
            max_budget: 1.0,
            message: "Budget has been exceeded!".to_owned(),
            subject: None,
            entity_type: Some("key".to_owned()),
            entity_id: None,
        }
    }

    async fn run_with(
        host: &ScriptedHost,
        token: &str,
        route: &str,
        settings: &KeySettings,
    ) -> KeyOutcome<Labels> {
        let token = Token::new(token);
        let request = KeyRequest {
            route,
            end_user_id: None,
            now: NOW,
            is_llm_api_route: route == "/chat/completions",
        };
        authenticate(
            host,
            &request,
            settings,
            Credential::Presented(Some(&token)),
        )
        .await
    }

    async fn run_resolved(
        host: &ScriptedHost,
        record: KeyRecord,
        settings: &KeySettings,
    ) -> KeyOutcome<Labels> {
        let request = KeyRequest {
            route: "/chat/completions",
            end_user_id: None,
            now: NOW,
            is_llm_api_route: true,
        };
        let resolved = Keyed {
            record,
            row: "mapped".to_owned(),
        };
        authenticate(host, &request, settings, Credential::Resolved(resolved)).await
    }

    /// The outcome with a denial's key dropped; `denials_carry_the_resolved_key` checks the key.
    async fn run(
        host: &ScriptedHost,
        token: &str,
        route: &str,
        settings: &KeySettings,
    ) -> Result<KeyIdentity<Labels>, Error> {
        run_with(host, token, route, settings)
            .await
            .map_err(|refused| *refused.error)
    }

    fn context(outcome: Result<KeyIdentity<Labels>, Error>) -> KeyContext<Labels> {
        match outcome {
            Ok(KeyIdentity::VirtualKey(context)) => *context,
            other => panic!("expected a virtual key, got {other:?}"),
        }
    }

    #[rstest]
    #[tokio::test]
    async fn without_master_key_every_token_is_let_through(settings: KeySettings) {
        let settings = KeySettings {
            master_key: None,
            ..settings
        };
        let host = ScriptedHost::default();
        assert_eq!(
            run(&host, "", "/chat/completions", &settings).await,
            Ok(KeyIdentity::NoMasterKey {
                token: Some(Token::new(""))
            })
        );
        assert!(host.calls().is_empty());
    }

    #[rstest]
    #[tokio::test]
    async fn missing_token_is_refused(settings: KeySettings) {
        let host = ScriptedHost::default();
        let request = KeyRequest {
            route: "/chat/completions",
            end_user_id: None,
            now: NOW,
            is_llm_api_route: true,
        };
        let outcome = authenticate(&host, &request, &settings, Credential::Presented(None)).await;
        assert_eq!(
            outcome.map_err(|refused| *refused.error),
            Err(KeyDenial::MissingKey.into())
        );
    }

    #[rstest]
    #[tokio::test]
    async fn empty_token_is_malformed(settings: KeySettings) {
        let host = ScriptedHost::default();
        assert_eq!(
            run(&host, "", "/chat/completions", &settings).await,
            Err(KeyDenial::MalformedKey.into())
        );
    }

    #[rstest]
    #[case::allowed(true, Ok(KeyIdentity::UserAuthRoute))]
    #[case::disabled(false, Err(KeyDenial::UserAuthDisabled.into()))]
    #[tokio::test]
    async fn user_auth_route(
        settings: KeySettings,
        #[case] allow_user_auth: bool,
        #[case] expected: Result<KeyIdentity<Labels>, Error>,
    ) {
        let settings = KeySettings {
            allow_user_auth,
            ..settings
        };
        let host = ScriptedHost::default();
        assert_eq!(run(&host, "sk-1", "/user/auth", &settings).await, expected);
        assert!(host.calls().is_empty());
    }

    #[rstest]
    #[tokio::test]
    async fn master_key_carries_the_end_user(settings: KeySettings) {
        let host = ScriptedHost::default();
        assert_eq!(
            run(&host, MASTER, "/chat/completions", &settings).await,
            Ok(KeyIdentity::MasterKey {
                hash: Token::new(MASTER).hash(),
                end_user: "end-user".to_owned()
            })
        );
        assert_eq!(
            host.calls(),
            ["prefetch_identity - unread", "end_user -", "cached_key"]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn end_user_over_budget_is_refused_before_the_key_is_read(settings: KeySettings) {
        let host = ScriptedHost {
            end_user_over_budget: Some(over_budget()),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(Error::BudgetExceeded(over_budget()))
        );
        assert_eq!(host.calls(), ["prefetch_identity - unread", "end_user -"]);
    }

    #[rstest]
    #[case::the_request_names_it(
        None,
        Some("from-body"),
        false,
        &["validated_end_user_id from-body", "prefetch_identity from-body unread", "end_user from-body"]
    )]
    #[case::the_resolved_key_names_it_first(
        Some("from-key"),
        Some("from-body"),
        false,
        &["validated_end_user_id from-key", "prefetch_identity from-key resolved", "end_user from-key"]
    )]
    #[case::an_empty_id_on_the_key_leaves_the_requests(
        Some(""),
        Some("from-body"),
        false,
        &["validated_end_user_id from-body", "prefetch_identity from-body resolved", "end_user from-body"]
    )]
    #[case::an_id_the_host_drops_is_not_read(
        None,
        Some("from-body"),
        true,
        &["validated_end_user_id from-body", "prefetch_identity - unread", "end_user -"]
    )]
    #[tokio::test]
    async fn the_end_user_is_named_validated_then_read(
        settings: KeySettings,
        #[case] key_end_user: Option<&str>,
        #[case] requested: Option<&str>,
        #[case] end_user_unknown: bool,
        #[case] expected: &[&str],
    ) {
        let host = ScriptedHost {
            end_user_unknown,
            ..ScriptedHost::default()
        };
        let token = Token::new("sk-1");
        let request = KeyRequest {
            route: "/chat/completions",
            end_user_id: requested,
            now: NOW,
            is_llm_api_route: true,
        };
        let credential = match key_end_user {
            None => Credential::Presented(Some(&token)),
            Some(end_user_id) => Credential::Resolved(Keyed {
                record: KeyRecord {
                    end_user_id: Some(end_user_id.to_owned()),
                    ..virtual_key()
                },
                row: "mapped".to_owned(),
            }),
        };
        let _ = authenticate(&host, &request, &settings, credential).await;
        assert_eq!(&host.calls()[..expected.len()], expected);
    }

    #[rstest]
    #[tokio::test]
    async fn without_a_database_no_identity_is_prefetched(settings: KeySettings) {
        let settings = KeySettings {
            has_db: false,
            ..settings
        };
        let host = ScriptedHost::default();
        let _ = run(&host, MASTER, "/chat/completions", &settings).await;
        assert_eq!(host.calls().first().unwrap(), "end_user -");
    }

    #[rstest]
    #[case::fresh(None)]
    #[case::not_yet_expired(Some(at(1_001)))]
    #[tokio::test]
    async fn cached_admin_key_carries_its_row(
        settings: KeySettings,
        #[case] expires: Option<Timestamp>,
    ) {
        let host = ScriptedHost {
            cached: Some(KeyRecord {
                is_proxy_admin: true,
                expires,
                ..virtual_key()
            }),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "sk-admin", "/chat/completions", &settings).await,
            Ok(KeyIdentity::CachedAdmin {
                key: "cached".to_owned(),
                end_user: "end-user".to_owned()
            })
        );
    }

    #[rstest]
    #[tokio::test]
    async fn expired_cached_admin_key_is_evicted_then_refused(settings: KeySettings) {
        let host = ScriptedHost {
            cached: Some(KeyRecord {
                is_proxy_admin: true,
                expires: Some(at(999)),
                ..virtual_key()
            }),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "abc", "/chat/completions", &settings).await,
            Err(KeyDenial::Expired {
                expires: at(999),
                now: NOW,
                key: "sk-...".to_owned(),
            }
            .into())
        );
        assert_eq!(
            host.calls().last().unwrap(),
            "evict ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn expired_resolved_admin_key_is_evicted_under_its_stored_hash(settings: KeySettings) {
        let host = ScriptedHost::default();
        let record = KeyRecord {
            is_proxy_admin: true,
            expires: Some(at(999)),
            ..virtual_key()
        };
        let outcome = run_resolved(&host, record, &settings).await;
        assert!(matches!(
            outcome.map_err(|refused| *refused.error),
            Err(Error::Key(KeyDenial::Expired { .. }))
        ));
        assert_eq!(host.calls().last().unwrap(), "evict hashed");
    }

    #[rstest]
    #[tokio::test]
    async fn master_key_only_route_refuses_virtual_keys(settings: KeySettings) {
        let host = ScriptedHost::default();
        assert_eq!(
            run(&host, "sk-1", "/global/spend/reset", &settings).await,
            Err(KeyDenial::MasterKeyOnlyRoute {
                route: "/global/spend/reset".to_owned()
            }
            .into())
        );
        assert!(matches!(
            run(&host, MASTER, "/global/spend/reset", &settings).await,
            Ok(KeyIdentity::MasterKey { .. })
        ));
    }

    #[rstest]
    #[tokio::test]
    async fn without_db_only_the_master_key_works(settings: KeySettings) {
        let settings = KeySettings {
            has_db: false,
            ..settings
        };
        let host = ScriptedHost::default();
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(KeyDenial::NoDb.into())
        );
    }

    #[rstest]
    #[case::random("not-a-key", false, Err(KeyDenial::NotVirtualKey { received: "not-****-key".to_owned(), jwt_hint: false }.into()))]
    #[case::jwt_with_jwt_auth_off("a.b.c", false, Err(KeyDenial::NotVirtualKey { received: "****".to_owned(), jwt_hint: true }.into()))]
    #[case::jwt_with_jwt_auth_on("a.b.c", true, Err(KeyDenial::NotVirtualKey { received: "****".to_owned(), jwt_hint: false }.into()))]
    #[tokio::test]
    async fn non_virtual_key_tokens(
        settings: KeySettings,
        #[case] token: &str,
        #[case] jwt_auth: bool,
        #[case] expected: Result<KeyIdentity<Labels>, Error>,
    ) {
        let settings = KeySettings {
            jwt_auth,
            ui_login: false,
            ..settings
        };
        let host = ScriptedHost::default();
        assert_eq!(
            run(&host, token, "/chat/completions", &settings).await,
            expected
        );
        assert_eq!(
            host.calls(),
            ["prefetch_identity - unread", "end_user -", "cached_key"]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn virtual_key_missing_from_the_database_is_not_found(settings: KeySettings) {
        let host = ScriptedHost::default();
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(KeyDenial::NotFound.into())
        );
        assert_eq!(
            host.calls(),
            [
                "prefetch_identity - unread",
                "end_user -",
                "cached_key",
                "db_key end-user"
            ]
        );
    }

    #[rstest]
    #[case::non_virtual_key_tries_ui_login("ui-session-blob", true, &["prefetch_identity - unread", "end_user -", "cached_key", "ui_key"])]
    #[case::ui_login_off("ui-session-blob", false, &["prefetch_identity - unread", "end_user -", "cached_key"])]
    #[case::virtual_key_skips_ui_login("sk-1", true, &["prefetch_identity - unread", "end_user -", "cached_key", "db_key end-user"])]
    #[tokio::test]
    async fn ui_login_key_is_only_tried_for_non_virtual_keys(
        settings: KeySettings,
        #[case] token: &str,
        #[case] ui_login: bool,
        #[case] expected_prefix: &[&str],
    ) {
        let settings = KeySettings {
            ui_login,
            ..settings
        };
        let host = ScriptedHost::default();
        let _ = run(&host, token, "/chat/completions", &settings).await;
        assert_eq!(&host.calls()[..expected_prefix.len()], expected_prefix);
    }

    #[rstest]
    #[tokio::test]
    async fn virtual_key_runs_every_check_in_order_and_keeps_every_row(settings: KeySettings) {
        let settings = KeySettings {
            proxy_budget: true,
            ..settings
        };
        let host = ScriptedHost {
            db: Some(KeyRecord {
                user_id: Some("u".to_owned()),
                team_id: Some("t".to_owned()),
                project_id: Some("p".to_owned()),
                ..virtual_key()
            }),
            key_owner: Some(false),
            key_team_found: true,
            ..ScriptedHost::default()
        };
        assert_eq!(
            context(run(&host, "sk-1", "/chat/completions", &settings).await),
            KeyContext {
                key: "db+prepared".to_owned(),
                end_user: "end-user".to_owned(),
                owner: Some("owner u".to_owned()),
                team: TeamGrant::Loaded("team t".to_owned()),
                project: Some("project p".to_owned()),
            }
        );
        assert_eq!(
            host.calls(),
            [
                "prefetch_identity - unread",
                "end_user -",
                "cached_key",
                "db_key end-user",
                "temp_budget db",
                "normalize db+prepared",
                "prefetch db+prepared end-user",
                "key_owner u",
                "key_skips_budget_checks db+prepared",
                "team_member_budget db+prepared owner u",
                "key_budgets db+prepared owner u llm",
                "key_team t",
                "project p",
                "proxy_budget_alert db+prepared",
            ]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn blocked_key_is_refused_before_any_request_work(settings: KeySettings) {
        let host = ScriptedHost {
            cached: Some(KeyRecord {
                blocked: true,
                ..virtual_key()
            }),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(KeyDenial::Blocked.into())
        );
        assert_eq!(
            host.calls(),
            ["prefetch_identity - unread", "end_user -", "cached_key"]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn deactivated_owner_is_refused(settings: KeySettings) {
        let host = ScriptedHost {
            cached: Some(KeyRecord {
                user_id: Some("u".to_owned()),
                ..virtual_key()
            }),
            key_owner: Some(true),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(KeyDenial::UserDeactivated {
                user_id: "u".to_owned()
            }
            .into())
        );
        assert_eq!(host.calls().last().unwrap(), "key_owner u");
    }

    #[rstest]
    #[tokio::test]
    async fn the_key_budget_step_denies_with_what_the_host_reported(settings: KeySettings) {
        let host = ScriptedHost {
            cached: Some(virtual_key()),
            key_over_budget: Some(over_budget()),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(Error::BudgetExceeded(over_budget()))
        );
        assert_eq!(
            host.calls().last().unwrap(),
            "key_budgets cached+prepared - llm"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_free_request_runs_no_budget_step(settings: KeySettings) {
        let host = ScriptedHost {
            cached: Some(virtual_key()),
            key_skips_budget_checks: true,
            team_member_over_budget: Some(over_budget()),
            key_over_budget: Some(over_budget()),
            ..ScriptedHost::default()
        };
        assert!(
            run(&host, "sk-1", "/chat/completions", &settings)
                .await
                .is_ok()
        );
        assert!(
            !host
                .calls()
                .iter()
                .any(|call| call.starts_with("team_member_budget")
                    || call.starts_with("key_budgets"))
        );
    }

    #[rstest]
    #[tokio::test]
    async fn the_team_member_budget_denies_before_expiry(settings: KeySettings) {
        let host = ScriptedHost {
            cached: Some(KeyRecord {
                expires: Some(at(999)),
                ..virtual_key()
            }),
            team_member_over_budget: Some(over_budget()),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(Error::BudgetExceeded(over_budget()))
        );
    }

    #[rstest]
    #[tokio::test]
    async fn expiry_is_checked_after_the_team_member_budget(settings: KeySettings) {
        let host = ScriptedHost {
            cached: Some(KeyRecord {
                expires: Some(at(999)),
                user_id: Some("u".to_owned()),
                team_id: Some("t".to_owned()),
                ..virtual_key()
            }),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(KeyDenial::Expired {
                expires: at(999),
                now: NOW,
                key: "sk-...".to_owned(),
            }
            .into())
        );
        assert!(
            host.calls()
                .last()
                .unwrap()
                .starts_with("team_member_budget")
        );
    }

    #[rstest]
    #[case::no_team(None, None, true, TeamGrant::NoTeam)]
    #[case::ui_team(Some("litellm-dashboard"), None, true, TeamGrant::KeyCopy(None))]
    #[case::unresolvable_team(Some("t"), None, false, TeamGrant::KeyCopy(None))]
    #[case::unresolvable_team_with_permission(Some("t"), Some("op"), false, TeamGrant::KeyCopy(Some("permission op".to_owned())))]
    #[case::resolved_team(Some("t"), Some("op"), true, TeamGrant::Loaded("team t".to_owned()))]
    #[tokio::test]
    async fn team_falls_back_to_the_key_copy(
        settings: KeySettings,
        #[case] team_id: Option<&str>,
        #[case] team_object_permission_id: Option<&str>,
        #[case] key_team_found: bool,
        #[case] expected: TeamGrant<Labels>,
    ) {
        let host = ScriptedHost {
            cached: Some(KeyRecord {
                team_id: team_id.map(str::to_owned),
                team_object_permission_id: team_object_permission_id.map(str::to_owned),
                ..virtual_key()
            }),
            key_team_found,
            ..ScriptedHost::default()
        };
        assert_eq!(
            context(run(&host, "sk-1", "/chat/completions", &settings).await).team,
            expected
        );
    }

    #[rstest]
    #[tokio::test]
    async fn key_without_token_is_refused_last(settings: KeySettings) {
        let host = ScriptedHost {
            cached: Some(KeyRecord::default()),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(KeyDenial::NoToken.into())
        );
    }

    #[rstest]
    #[case::team_newer(Some(at(2)), Some(at(1)), "cached+team+prepared")]
    #[case::team_older(Some(at(1)), Some(at(2)), "cached+prepared")]
    #[case::key_never_refreshed(Some(at(2)), None, "cached+prepared")]
    #[case::team_never_refreshed(None, Some(at(1)), "cached+prepared")]
    #[tokio::test]
    async fn cached_team_values_reach_a_stale_key(
        settings: KeySettings,
        #[case] team_refreshed_at: Option<Timestamp>,
        #[case] key_refreshed_at: Option<Timestamp>,
        #[case] key: &str,
    ) {
        let host = ScriptedHost {
            cached: Some(KeyRecord {
                team_id: Some("t".to_owned()),
                last_refreshed_at: key_refreshed_at,
                ..virtual_key()
            }),
            team_refreshed_at,
            key_team_found: true,
            ..ScriptedHost::default()
        };
        assert_eq!(
            context(run(&host, "sk-1", "/chat/completions", &settings).await).key,
            key
        );
    }

    #[rstest]
    #[tokio::test]
    async fn failed_cache_read_falls_through_to_the_database(settings: KeySettings) {
        let host = ScriptedHost {
            cached: Some(virtual_key()),
            failing: HashMap::from([("cached_key", Error::HostFailed)]),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(KeyDenial::NotFound.into())
        );
    }

    #[rstest]
    #[case::cached_team("cached_team", None, "cached+prepared")]
    #[case::deactivated_owner("key_owner", Some(true), "cached+team+prepared")]
    #[tokio::test]
    async fn failed_lookups_are_treated_as_absent(
        settings: KeySettings,
        #[case] failing: &'static str,
        #[case] key_owner: Option<bool>,
        #[case] key: &str,
    ) {
        let host = ScriptedHost {
            cached: Some(KeyRecord {
                user_id: Some("u".to_owned()),
                team_id: Some("t".to_owned()),
                last_refreshed_at: Some(at(1)),
                ..virtual_key()
            }),
            team_refreshed_at: Some(at(2)),
            key_owner,
            failing: HashMap::from([(failing, Error::HostFailed)]),
            ..ScriptedHost::default()
        };
        let context = context(run(&host, "sk-1", "/chat/completions", &settings).await);
        assert_eq!((context.key.as_str(), context.owner), (key, None));
    }

    #[rstest]
    #[case::prefetch_identity("prefetch_identity")]
    #[case::end_user("end_user")]
    #[case::db_key("db_key")]
    #[case::key_team("key_team")]
    #[case::project("project")]
    #[tokio::test]
    async fn other_host_failures_end_the_call(
        settings: KeySettings,
        #[case] failing: &'static str,
    ) {
        let host = ScriptedHost {
            db: Some(KeyRecord {
                team_id: Some("t".to_owned()),
                project_id: Some("p".to_owned()),
                ..virtual_key()
            }),
            failing: HashMap::from([(failing, Error::HostFailed)]),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(Error::HostFailed)
        );
    }

    #[rstest]
    #[tokio::test]
    async fn resolved_key_skips_the_lookups(settings: KeySettings) {
        let host = ScriptedHost::default();
        let outcome = run_resolved(&host, virtual_key(), &settings).await;
        assert_eq!(
            context(outcome.map_err(|refused| *refused.error)).key,
            "mapped+prepared"
        );
        assert_eq!(
            &host.calls()[..5],
            [
                "prefetch_identity - resolved",
                "end_user -",
                "temp_budget mapped",
                "normalize mapped+prepared",
                "prefetch mapped+prepared end-user"
            ]
        );
    }

    #[rstest]
    #[case::not_found_has_no_key("sk-1", None, None)]
    #[case::blocked_cached_key("sk-1", Some(KeyRecord { blocked: true, ..virtual_key() }), Some("cached"))]
    #[case::expired_key_after_prepare("sk-1", Some(KeyRecord { expires: Some(at(999)), ..virtual_key() }), Some("cached+prepared"))]
    #[tokio::test]
    async fn denials_carry_the_resolved_key(
        settings: KeySettings,
        #[case] token: &str,
        #[case] cached: Option<KeyRecord>,
        #[case] key: Option<&str>,
    ) {
        let host = ScriptedHost {
            cached,
            ..ScriptedHost::default()
        };
        let refused = run_with(&host, token, "/chat/completions", &settings)
            .await
            .unwrap_err();
        assert_eq!(refused.key.as_deref(), key);
    }
}
