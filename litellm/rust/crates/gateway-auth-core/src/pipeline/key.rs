use crate::{
    AuthHost,
    authn::key::{self, Credential, KeyRequest, KeySettings, Prechecked, Verified},
    authz::RoutePolicy,
    host::KeyRow,
    identity::{
        self,
        key::{KeyContext, KeyOutcome},
    },
    outcome::Refused,
};

/// The key path: what the credential settles alone, the route policy, the key and whether it is
/// still usable, who holds it, then their budgets.
pub(crate) async fn authenticate<H: AuthHost>(
    host: &H,
    request: &KeyRequest<'_>,
    settings: &KeySettings,
    policy: &RoutePolicy<'_>,
    credential: Credential<'_, KeyRow<H>>,
) -> Result<KeyOutcome<H::Rows>, Refused<H::Rows>> {
    let checked = key::precheck::<H::Rows>(request.route, credential, settings)
        .map_err(|denial| Refused::new(denial, None))?;
    let verified = match checked {
        Prechecked::Decided(verified) => verified,
        Prechecked::Named(named) => {
            policy
                .key_credential(named.is_master_key(), request.route)
                .map_err(|denial| Refused::new(denial, None))?;
            key::verify(host, named, request, settings).await?
        }
    };
    match verified {
        Verified::NoMasterKey { token } => Ok(KeyOutcome::NoMasterKey { token }),
        Verified::UserAuthRoute => Ok(KeyOutcome::UserAuthRoute),
        Verified::MasterKey { hash, end_user } => Ok(KeyOutcome::MasterKey { hash, end_user }),
        Verified::CachedAdmin { key, end_user } => Ok(KeyOutcome::CachedAdmin { key, end_user }),
        Verified::VirtualKey { key, end_user } => {
            let principal = identity::key::principal(host, key, end_user).await?;
            let admitted = crate::admission::key(host, principal, request, settings).await?;
            Ok(KeyOutcome::VirtualKey(Box::new(KeyContext::from(admitted))))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use jiff::Timestamp;
    use rstest::{fixture, rstest};

    use super::*;
    use crate::{
        Belongs, Error, KeyDenial, KeyHash, MasterKey, TeamGrant, Token,
        authn::{
            jwt::JwtSettings,
            key::{KeyRecord, Keyed},
        },
        identity::key::{ObjectPermissionRecord, ProjectRecord},
        testing::{Labels, ScriptedHost, end_user},
    };

    /// The key path with no JWT route rules, which it never reads.
    async fn authenticate(
        host: &ScriptedHost,
        request: &KeyRequest<'_>,
        settings: &KeySettings,
        credential: Credential<'_, String>,
    ) -> Result<KeyOutcome<Labels>, Refused<Labels>> {
        let jwt = JwtSettings::default();
        super::authenticate(
            host,
            request,
            settings,
            &RoutePolicy::new(settings, &jwt),
            credential,
        )
        .await
    }

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
    ) -> Result<KeyOutcome<Labels>, Refused<Labels>> {
        let token = Token::new(token);
        let request = KeyRequest {
            route,
            end_user_id: None,
            now: NOW,
            runs_inference_checks: route == "/chat/completions",
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
    ) -> Result<KeyOutcome<Labels>, Refused<Labels>> {
        let request = KeyRequest {
            route: "/chat/completions",
            end_user_id: None,
            now: NOW,
            runs_inference_checks: true,
        };
        let resolved = Keyed {
            record,
            row: "mapped".to_owned(),
        };
        let credential = Credential::Resolved {
            key: resolved,
            end_user_id: None,
        };
        authenticate(host, &request, settings, credential).await
    }

    /// The outcome with a denial's key dropped; `denials_carry_the_resolved_key` checks the key.
    async fn run(
        host: &ScriptedHost,
        token: &str,
        route: &str,
        settings: &KeySettings,
    ) -> Result<KeyOutcome<Labels>, Error> {
        run_with(host, token, route, settings)
            .await
            .map_err(|refused| *refused.error)
    }

    fn context(outcome: Result<KeyOutcome<Labels>, Error>) -> KeyContext<Labels> {
        match outcome {
            Ok(KeyOutcome::VirtualKey(context)) => *context,
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
            Ok(KeyOutcome::NoMasterKey {
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
            runs_inference_checks: true,
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
    #[tokio::test]
    async fn a_resolved_key_stored_under_an_empty_hash_is_malformed(settings: KeySettings) {
        let host = ScriptedHost::default();
        let record = KeyRecord {
            token: Some(KeyHash::new("")),
            ..virtual_key()
        };
        assert_eq!(
            run_resolved(&host, record, &settings)
                .await
                .map_err(|refused| *refused.error),
            Err(KeyDenial::MalformedKey.into())
        );
    }

    #[rstest]
    #[case::allowed(true, Ok(KeyOutcome::UserAuthRoute))]
    #[case::disabled(false, Err(KeyDenial::UserAuthDisabled.into()))]
    #[tokio::test]
    async fn user_auth_route(
        settings: KeySettings,
        #[case] allow_user_auth: bool,
        #[case] expected: Result<KeyOutcome<Labels>, Error>,
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
            Ok(KeyOutcome::MasterKey {
                hash: Token::new(MASTER).hash(),
                end_user: end_user("end-user")
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
    #[case::the_resolved_credential_names_it_first(
        Some("from-jwt"),
        Some("from-body"),
        false,
        &["validated_end_user_id from-jwt", "prefetch_identity from-jwt resolved", "end_user from-jwt"]
    )]
    #[case::an_empty_id_on_the_credential_leaves_the_requests(
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
        #[case] credential_end_user: Option<&str>,
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
            runs_inference_checks: true,
        };
        let credential = match credential_end_user {
            None => Credential::Presented(Some(&token)),
            Some(end_user_id) => Credential::Resolved {
                key: Keyed {
                    record: virtual_key(),
                    row: "mapped".to_owned(),
                },
                end_user_id: Some(end_user_id),
            },
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
            Ok(KeyOutcome::CachedAdmin {
                key: "cached".to_owned(),
                end_user: end_user("end-user")
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
    /// Refused before anything is read: the check used to run after the end user and the key.
    async fn master_key_only_route_refuses_virtual_keys(settings: KeySettings) {
        let host = ScriptedHost::default();
        assert_eq!(
            run(&host, "sk-1", "/global/spend/reset", &settings).await,
            Err(KeyDenial::MasterKeyOnlyRoute {
                route: "/global/spend/reset".to_owned()
            }
            .into())
        );
        assert!(host.calls().is_empty());
        assert!(matches!(
            run(&host, MASTER, "/global/spend/reset", &settings).await,
            Ok(KeyOutcome::MasterKey { .. })
        ));
    }

    #[rstest]
    #[case::never_configured(false, KeyDenial::NoDbConfigured)]
    #[case::configured_but_not_connected(true, KeyDenial::NoDb)]
    #[tokio::test]
    async fn without_db_only_the_master_key_works(
        settings: KeySettings,
        #[case] db_configured: bool,
        #[case] denial: KeyDenial,
    ) {
        let settings = KeySettings {
            has_db: false,
            db_configured,
            ..settings
        };
        let host = ScriptedHost::default();
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(denial.into())
        );
        assert!(matches!(
            run(&host, MASTER, "/chat/completions", &settings).await,
            Ok(KeyOutcome::MasterKey { .. })
        ));
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
        #[case] expected: Result<KeyOutcome<Labels>, Error>,
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
            Err(KeyDenial::NotFound {
                hash: Token::new("sk-1").hash()
            }
            .into())
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
                belongs: Belongs {
                    user: Some("owner u".to_owned()),
                    team: TeamGrant::Loaded("team t".to_owned()),
                    project: Some(ProjectRecord {
                        project_alias: Some("project p".to_owned()),
                        metadata: None,
                    }),
                    end_user: Some(end_user("end-user")),
                    ..Belongs::default()
                },
            }
        );
        assert_eq!(
            host.calls(),
            [
                "prefetch_identity - unread",
                "end_user -",
                "cached_key",
                "db_key end-user",
                "prefetch db end-user",
                "key_owner u",
                "key_team t",
                "project p",
                "temp_budget db",
                "normalize db+prepared",
                "key_skips_budget_checks db+prepared",
                "team_member_budget db+prepared owner u",
                "key_budgets db+prepared owner u llm",
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

    /// An expired key is refused before its owner, team, project or budgets are read: expiry
    /// used to run after the team member budget, whose denial won.
    #[rstest]
    #[case::over_team_member_budget(None, Some(over_budget()))]
    #[case::deactivated_owner(Some(true), None)]
    #[tokio::test]
    async fn expiry_is_checked_before_any_identity_read(
        settings: KeySettings,
        #[case] key_owner: Option<bool>,
        #[case] team_member_over_budget: Option<crate::BudgetExceeded>,
    ) {
        let host = ScriptedHost {
            cached: Some(KeyRecord {
                expires: Some(at(999)),
                user_id: Some("u".to_owned()),
                team_id: Some("t".to_owned()),
                ..virtual_key()
            }),
            key_owner,
            team_member_over_budget,
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
        assert_eq!(
            host.calls(),
            [
                "prefetch_identity - unread",
                "end_user -",
                "cached_key",
                "cached_team t"
            ]
        );
    }

    /// The key's team and project are read before its budgets, so a failed read ends the call
    /// before a budget could deny it.
    #[rstest]
    #[case::team("key_team")]
    #[case::project("project")]
    #[tokio::test]
    async fn identity_reads_run_before_the_budgets(
        settings: KeySettings,
        #[case] failing: &'static str,
    ) {
        let host = ScriptedHost {
            cached: Some(KeyRecord {
                team_id: Some("t".to_owned()),
                project_id: Some("p".to_owned()),
                ..virtual_key()
            }),
            key_team_found: true,
            key_over_budget: Some(over_budget()),
            failing: HashMap::from([(failing, Error::HostFailed)]),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(Error::HostFailed)
        );
        assert!(
            !host
                .calls()
                .iter()
                .any(|call| call.starts_with("key_budgets"))
        );
    }

    #[rstest]
    #[case::no_team(None, None, true, TeamGrant::NoTeam)]
    #[case::ui_team(Some("litellm-dashboard"), None, true, TeamGrant::KeyCopy(None))]
    #[case::unresolvable_team(Some("t"), None, false, TeamGrant::KeyCopy(None))]
    #[case::unresolvable_team_with_permission(Some("t"), Some("op"), false, TeamGrant::KeyCopy(Some(Box::new(ObjectPermissionRecord { object_permission_id: "op".to_owned(), ..ObjectPermissionRecord::default() }))))]
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
            context(run(&host, "sk-1", "/chat/completions", &settings).await)
                .belongs
                .team,
            expected
        );
    }

    #[rstest]
    #[tokio::test]
    async fn key_without_token_is_refused_before_any_identity_read(settings: KeySettings) {
        let host = ScriptedHost {
            cached: Some(KeyRecord {
                user_id: Some("u".to_owned()),
                ..KeyRecord::default()
            }),
            key_owner: Some(false),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(KeyDenial::NoToken.into())
        );
        assert_eq!(
            host.calls(),
            ["prefetch_identity - unread", "end_user -", "cached_key"]
        );
    }

    #[rstest]
    #[case::team_newer(Some(at(2)), Some(at(1)), "cached+team+prepared")]
    #[case::team_older(Some(at(1)), Some(at(2)), "cached+prepared")]
    #[case::refreshed_together(Some(at(1)), Some(at(1)), "cached+prepared")]
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
            Err(KeyDenial::NotFound {
                hash: Token::new("sk-1").hash()
            }
            .into())
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
        assert_eq!((context.key.as_str(), context.belongs.user), (key, None));
    }

    #[rstest]
    #[case::cached_key("cached_key")]
    #[case::cached_team("cached_team")]
    #[case::key_owner("key_owner")]
    #[tokio::test]
    async fn a_read_the_host_declines_is_not_absent(
        settings: KeySettings,
        #[case] declining: &'static str,
    ) {
        let host = ScriptedHost {
            cached: Some(KeyRecord {
                user_id: Some("u".to_owned()),
                team_id: Some("t".to_owned()),
                last_refreshed_at: Some(at(1)),
                ..virtual_key()
            }),
            team_refreshed_at: Some(at(2)),
            failing: HashMap::from([(declining, Error::HostDeclined)]),
            ..ScriptedHost::default()
        };
        assert_eq!(
            run(&host, "sk-1", "/chat/completions", &settings).await,
            Err(Error::HostDeclined)
        );
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
                "prefetch mapped end-user",
                "temp_budget mapped",
                "normalize mapped+prepared"
            ]
        );
    }

    #[rstest]
    #[case::not_found_has_no_key("sk-1", None, None)]
    #[case::blocked_cached_key("sk-1", Some(KeyRecord { blocked: true, ..virtual_key() }), Some("cached"))]
    #[case::expired_key_before_prepare("sk-1", Some(KeyRecord { expires: Some(at(999)), ..virtual_key() }), Some("cached"))]
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
