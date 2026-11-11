use std::{
    collections::{HashMap, HashSet},
    sync::Mutex,
    time::Duration,
};

use litellm_config::LitellmRole;
use serde_json::Value;

use crate::{
    AuthHost, BudgetExceeded, Error, Rows,
    authorize::Membership,
    budgets::{BudgetAlert, MemberBudget, NamedBudget, RecordedBudget, SpendCounter},
    key_budget::KeyBudgetAlert,
    jwt::{
        AgentIdentityFailure, AliasRead, DelegatedTeam, FinishedJwt, IdentityRead, KeyMappingRow,
        KeyMappingWrite, ManagedAgent, MappedJwt, MappedKey, ModelAccess, Org, RegisteredAgent,
        TeamRead, User, UserLookup, UserRead,
    },
    key::{CachedTeam, KeyRecord, Keyed, Owner},
};

/// Rows the scripted host labels by where they came from, so a test can tell which row an
/// identity carries.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Labels;

impl Rows for Labels {
    type Key = String;
    type User = String;
    type Team = String;
    type EndUser = String;
    type ObjectPermission = String;
    type Project = String;
    type Membership = String;
    type Org = String;
    type Agent = String;
}

#[derive(Default)]
pub(crate) struct ScriptedHost {
    pub(crate) end_user_over_budget: Option<BudgetExceeded>,
    pub(crate) cached: Option<KeyRecord>,
    pub(crate) ui: Option<KeyRecord>,
    pub(crate) db: Option<KeyRecord>,
    pub(crate) refreshed: KeyRecord,
    pub(crate) team_refreshed_at: Option<f64>,
    /// The owner, and whether it is deactivated.
    pub(crate) key_owner: Option<bool>,
    pub(crate) key_skips_budget_checks: bool,
    pub(crate) team_member_max_budget: Option<f64>,
    pub(crate) global_proxy_spend: Option<f64>,
    pub(crate) key_over_budget: Option<BudgetExceeded>,
    pub(crate) key_team_found: bool,
    /// Host calls that fail, by the name they record under.
    pub(crate) failing: HashMap<&'static str, Error>,
    pub(crate) teams: HashMap<&'static str, Result<TeamRead<String>, Error>>,
    pub(crate) aliases: HashMap<&'static str, Result<AliasRead<String>, Error>>,
    pub(crate) membership_fails: bool,
    /// Team rows whose model access the host allows.
    pub(crate) model_access: HashSet<&'static str>,
    pub(crate) jwt_user_over_budget: Option<BudgetExceeded>,
    /// Bodies by URL; `None` is a transport outage, an unscripted URL a host error.
    pub(crate) fetched: HashMap<String, Option<Value>>,
    pub(crate) userinfo: Option<Value>,
    pub(crate) json_cache: Mutex<HashMap<String, Value>>,
    pub(crate) custom_validate_passes: bool,
    pub(crate) users: HashMap<&'static str, User<String>>,
    pub(crate) orgs: HashMap<&'static str, Org<String>>,
    /// Mapped hashes by `"{field}={value} {issuer or -}"`.
    pub(crate) mappings: HashMap<String, String>,
    pub(crate) keys: HashMap<&'static str, MappedKey<String>>,
    pub(crate) reusable_key: Option<&'static str>,
    /// `Created("minted")` when unset.
    pub(crate) key_mapping_write: Option<KeyMappingWrite>,
    pub(crate) managed: Option<ManagedAgent<String>>,
    pub(crate) agent_failure: Option<AgentIdentityFailure>,
    pub(crate) agents: HashMap<&'static str, RegisteredAgent>,
    pub(crate) delegated_team: Option<&'static str>,
    pub(crate) unpriced: Vec<String>,
    /// Team organizations by team id.
    pub(crate) team_organizations: HashMap<&'static str, &'static str>,
    pub(crate) agent_requires_trace_id: bool,
    pub(crate) has_trace_id: bool,
    /// Live spend by counter label, such as `"team t1"`; an unscripted counter reads zero.
    pub(crate) spends: HashMap<&'static str, f64>,
    pub(crate) organization_budgets: HashMap<&'static str, RecordedBudget>,
    pub(crate) tag_budgets: Vec<NamedBudget>,
    pub(crate) model_access_group_budgets: Vec<NamedBudget>,
    pub(crate) member_budget: MemberBudget,
    /// The JWT `finish_jwt` last finished an auto-registered key for.
    pub(crate) finished_mapping: Mutex<Option<MappedJwt<String>>>,
    pub(crate) calls: Mutex<Vec<String>>,
}

fn or_dash(value: Option<&str>) -> &str {
    value.unwrap_or("-")
}

fn membership_label(membership: Membership<'_, String>) -> String {
    match membership {
        Membership::Unread => "-".to_owned(),
        Membership::Read(None) => "none".to_owned(),
        Membership::Read(Some(row)) => row.clone(),
    }
}

fn spend_label(counter: SpendCounter<'_>) -> String {
    let scope: &str = counter.into();
    let entity = match counter {
        SpendCounter::Team { team_id: id, .. }
        | SpendCounter::Organization { org_id: id, .. }
        | SpendCounter::Tag { tag: id, .. }
        | SpendCounter::ModelAccessGroup { group: id, .. }
        | SpendCounter::User { user_id: id, .. }
        | SpendCounter::EndUser { end_user_id: id, .. }
        | SpendCounter::Key { token: id, .. } => id.to_owned(),
        SpendCounter::TeamWindow { team_id: id, window } | SpendCounter::KeyWindow { token: id, window } => {
            format!("{id} {}", window.duration)
        }
        SpendCounter::TeamMember { user_id, team_id, .. } => format!("{user_id} {team_id}"),
    };
    format!("{scope} {entity}")
}

fn keyed(record: &KeyRecord, label: &str) -> Keyed<String> {
    Keyed {
        record: record.clone(),
        row: label.to_owned(),
    }
}

impl ScriptedHost {
    fn record(&self, call: impl Into<String>) {
        self.calls.lock().unwrap().push(call.into());
    }

    /// Records `call` and fails it when the script says so.
    fn step(&self, name: &'static str, call: impl Into<String>) -> Result<(), Error> {
        self.record(call);
        self.failing.get(name).cloned().map_or(Ok(()), Err)
    }

    pub(crate) fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

impl AuthHost for ScriptedHost {
    type Rows = Labels;

    async fn end_user(&self, _: &str, resolved: Option<&String>) -> Result<String, Error> {
        self.step(
            "end_user",
            format!("end_user {}", or_dash(resolved.map(String::as_str))),
        )?;
        self.end_user_over_budget
            .clone()
            .map_or_else(|| Ok("end-user".to_owned()), |denial| Err(denial.into()))
    }
    async fn cached_key(&self, _: &str) -> Result<Option<Keyed<String>>, Error> {
        self.step("cached_key", "cached_key")?;
        Ok(self.cached.as_ref().map(|key| keyed(key, "cached")))
    }
    async fn ui_key(&self, _: &str) -> Result<Option<Keyed<String>>, Error> {
        self.step("ui_key", "ui_key")?;
        Ok(self.ui.as_ref().map(|key| keyed(key, "ui")))
    }
    async fn with_session_grants(
        &self,
        key: &String,
        user: &String,
        team: Option<&String>,
        membership: Option<&String>,
    ) -> Result<Keyed<String>, Error> {
        self.step(
            "with_session_grants",
            format!(
                "with_session_grants {key} {user} {} {}",
                or_dash(team.map(String::as_str)),
                or_dash(membership.map(String::as_str))
            ),
        )?;
        Ok(keyed(&self.refreshed, "refreshed"))
    }
    async fn evict_key(&self, hash: &str) -> Result<(), Error> {
        self.step("evict", format!("evict {hash}"))
    }
    async fn cached_team(&self, team_id: &str) -> Result<Option<CachedTeam<String>>, Error> {
        self.step("cached_team", format!("cached_team {team_id}"))?;
        Ok(Some(CachedTeam {
            refreshed_at: self.team_refreshed_at,
            row: format!("cached-team {team_id}"),
        }))
    }
    async fn with_team(&self, key: &String, team: &String) -> Result<String, Error> {
        self.step("with_team", format!("with_team {key} {team}"))?;
        Ok(format!("{key}+team"))
    }
    async fn db_key(&self, _: &str, end_user: &String) -> Result<Option<Keyed<String>>, Error> {
        self.step("db_key", format!("db_key {end_user}"))?;
        Ok(self.db.as_ref().map(|key| keyed(key, "db")))
    }
    async fn with_temp_budget(&self, key: &String) -> Result<String, Error> {
        self.step("temp_budget", format!("temp_budget {key}"))?;
        Ok(format!("{key}+prepared"))
    }
    async fn normalize_model(&self, key: &String) -> Result<(), Error> {
        self.step("normalize", format!("normalize {key}"))
    }
    async fn prefetch_auth_objects(&self, key: &String, end_user: &String) -> Result<(), Error> {
        self.step("prefetch", format!("prefetch {key} {end_user}"))
    }
    async fn key_owner(&self, user_id: &str) -> Result<Option<Owner<String>>, Error> {
        self.step("key_owner", format!("key_owner {user_id}"))?;
        Ok(self.key_owner.map(|deactivated| Owner {
            deactivated,
            row: format!("owner {user_id}"),
        }))
    }
    async fn key_skips_budget_checks(&self, key: &String) -> Result<bool, Error> {
        self.step("key_skips_budget_checks", format!("key_skips_budget_checks {key}"))?;
        Ok(self.key_skips_budget_checks)
    }
    async fn team_member_max_budget(&self, user_id: &str, team_id: &str) -> Result<Option<f64>, Error> {
        self.step(
            "team_member_max_budget",
            format!("team_member_max_budget {user_id} {team_id}"),
        )?;
        Ok(self.team_member_max_budget)
    }
    async fn throttle_key(&self, key: &String, share: f64) -> Result<(), Error> {
        self.step("throttle_key", format!("throttle_key {key} {share}"))
    }
    async fn key_model_budgets(&self, key: &String, owner: Option<&String>) -> Result<(), Error> {
        self.step(
            "key_model_budgets",
            format!("key_model_budgets {key} {}", or_dash(owner.map(String::as_str))),
        )?;
        self.key_over_budget
            .clone()
            .map_or(Ok(()), |denial| Err(denial.into()))
    }
    async fn key_team(&self, team_id: &str) -> Result<Option<String>, Error> {
        self.step("key_team", format!("key_team {team_id}"))?;
        Ok(self.key_team_found.then(|| format!("team {team_id}")))
    }
    async fn object_permission(&self, id: &str) -> Result<Option<String>, Error> {
        self.step("object_permission", format!("object_permission {id}"))?;
        Ok(Some(format!("permission {id}")))
    }
    async fn project(&self, project_id: &str) -> Result<Option<String>, Error> {
        self.step("project", format!("project {project_id}"))?;
        Ok(Some(format!("project {project_id}")))
    }
    async fn global_proxy_spend(&self) -> Result<Option<f64>, Error> {
        self.step("global_proxy_spend", "global_proxy_spend")?;
        Ok(self.global_proxy_spend)
    }
    async fn key_budget_alert(
        &self,
        key: &String,
        _: Option<&String>,
        alert: KeyBudgetAlert,
    ) -> Result<(), Error> {
        self.step(
            "key_budget_alert",
            format!("key_budget_alert {key} {}", <&str>::from(alert)),
        )
    }

    async fn fetch_json(&self, url: &str) -> Result<Option<Value>, Error> {
        self.record(format!("fetch {url}"));
        self.fetched.get(url).cloned().ok_or(Error::HostFailed)
    }
    async fn fetch_userinfo(&self, endpoint: &str, token: &str) -> Result<Option<Value>, Error> {
        self.record(format!("userinfo {endpoint} {token}"));
        Ok(self.userinfo.clone())
    }
    async fn cached_json(&self, key: &str) -> Result<Option<Value>, Error> {
        self.record(format!("get {key}"));
        Ok(self.json_cache.lock().unwrap().get(key).cloned())
    }
    async fn cache_json(&self, key: &str, value: &Value, ttl: Duration) -> Result<(), Error> {
        self.record(format!("set {key} {}", ttl.as_secs_f64()));
        self.json_cache
            .lock()
            .unwrap()
            .insert(key.to_owned(), value.clone());
        Ok(())
    }
    async fn forget_json(&self, key: &str) -> Result<(), Error> {
        self.record(format!("forget {key}"));
        self.json_cache.lock().unwrap().remove(key);
        Ok(())
    }

    async fn team(&self, team_id: &str, upsert: bool) -> Result<TeamRead<String>, Error> {
        self.record(format!(
            "team {team_id}{}",
            if upsert { " upsert" } else { "" }
        ));
        self.teams
            .get(team_id)
            .cloned()
            .unwrap_or(Ok(TeamRead::Absent))
    }
    async fn team_by_alias(&self, alias: &str) -> Result<AliasRead<String>, Error> {
        self.record(format!("alias {alias}"));
        self.aliases
            .get(alias)
            .cloned()
            .unwrap_or(Ok(AliasRead::Absent))
    }
    async fn granting_team(&self, team_id: &str) -> Result<TeamRead<String>, Error> {
        self.record(format!("granting_team {team_id}"));
        self.teams
            .get(team_id)
            .cloned()
            .unwrap_or(Ok(TeamRead::Absent))
    }
    async fn load_team_membership(
        &self,
        user_id: &str,
        team_id: &str,
    ) -> Result<Option<String>, Error> {
        self.record(format!("membership {user_id} {team_id}"));
        if self.membership_fails {
            Err(Error::HostFailed)
        } else {
            Ok(Some(format!("membership {user_id} {team_id}")))
        }
    }

    async fn user(&self, lookup: &UserLookup<'_>) -> Result<UserRead<String>, Error> {
        self.record(format!(
            "user {} {}",
            lookup.user_id,
            or_dash(lookup.user_email)
        ));
        Ok(self
            .users
            .get(lookup.user_id)
            .cloned()
            .map_or(UserRead::Absent, UserRead::Found))
    }
    async fn resolve_user_identity(
        &self,
        lookup: &UserLookup<'_>,
        team_id: Option<&str>,
    ) -> Result<IdentityRead<Labels>, Error> {
        let upsert = if lookup.upsert { " upsert" } else { "" };
        self.record(format!(
            "resolve_user {} team={}{upsert}",
            lookup.user_id,
            or_dash(team_id)
        ));
        let user = self
            .users
            .get(lookup.user_id)
            .cloned()
            .map_or(UserRead::Absent, UserRead::Found);
        let membership = match (&user, team_id) {
            (UserRead::Found(_), Some(team_id)) => {
                Some(format!("membership {} {team_id}", lookup.user_id))
            }
            _ => None,
        };
        Ok(IdentityRead { user, membership })
    }
    async fn set_user_role(&self, user: &String, role: LitellmRole) -> Result<String, Error> {
        self.record(format!("set_role {user} {}", <&str>::from(role)));
        Ok(format!("{user}+role"))
    }
    async fn patch_team_membership(
        &self,
        user: &String,
        add: &[String],
        remove: &[String],
    ) -> Result<String, Error> {
        self.record(format!(
            "patch {user} +{} -{}",
            add.join(","),
            remove.join(",")
        ));
        Ok(format!("{user}+teams"))
    }
    async fn add_team_member(&self, user_id: &str, team_id: &str) -> Result<(), Error> {
        self.record(format!("add_member {user_id} {team_id}"));
        Ok(())
    }
    async fn org(&self, org_id: &str) -> Result<Option<Org<String>>, Error> {
        self.record(format!("org {org_id}"));
        Ok(self.orgs.get(org_id).cloned())
    }
    async fn org_by_alias(&self, alias: &str) -> Result<Option<Org<String>>, Error> {
        self.record(format!("org_alias {alias}"));
        Ok(self.orgs.get(alias).cloned())
    }
    async fn jwt_end_user(&self, end_user_id: &str) -> Result<String, Error> {
        self.record(format!("end_user {end_user_id}"));
        Ok(format!("end-user {end_user_id}"))
    }
    async fn finish_jwt(&self, finished: FinishedJwt<'_, Labels>) -> Result<String, Error> {
        let token = match finished {
            FinishedJwt::Identity(_) => "finish identity".to_owned(),
            FinishedJwt::AutoRegisteredKey(jwt, key) => {
                *self.finished_mapping.lock().unwrap() = Some(jwt.clone());
                format!("finish auto_registered {key}")
            }
        };
        self.record(token.clone());
        self.jwt_user_over_budget
            .clone()
            .map_or(Ok(token), |denial| Err(denial.into()))
    }

    async fn key_mapping_hash(
        &self,
        claim_field: &str,
        claim_value: &str,
        issuer: Option<&str>,
    ) -> Result<Option<String>, Error> {
        let scope = format!("{claim_field}={claim_value} {}", or_dash(issuer));
        self.record(format!("mapping {scope}"));
        Ok(self.mappings.get(&scope).cloned())
    }
    async fn key_by_hash(&self, hash: &str) -> Result<Option<MappedKey<String>>, Error> {
        self.record(format!("key_by_hash {hash}"));
        Ok(self.keys.get(hash).cloned())
    }
    async fn reusable_key_hash(
        &self,
        user_id: &str,
        team_id: Option<&str>,
    ) -> Result<Option<String>, Error> {
        self.record(format!("reusable {user_id} {}", or_dash(team_id)));
        Ok(self.reusable_key.map(str::to_owned))
    }
    async fn create_key_mapping(&self, row: &KeyMappingRow<'_>) -> Result<KeyMappingWrite, Error> {
        self.record(format!(
            "create_mapping {}={} existing={}",
            row.claim_field,
            row.claim_value,
            or_dash(row.existing_hash)
        ));
        Ok(self
            .key_mapping_write
            .clone()
            .unwrap_or(KeyMappingWrite::Created(
                row.existing_hash.unwrap_or("minted").to_owned(),
            )))
    }

    async fn managed_agent(&self, _: &Value) -> Result<Option<ManagedAgent<String>>, Error> {
        self.record("managed_agent");
        Ok(self.managed.clone())
    }
    async fn record_agent_authentication(
        &self,
        agent: &String,
    ) -> Result<Option<AgentIdentityFailure>, Error> {
        self.record(format!("record_agent {agent}"));
        Ok(self.agent_failure.clone())
    }
    async fn registered_agent(&self, claim: &str) -> Result<Option<RegisteredAgent>, Error> {
        self.record(format!("agent {claim}"));
        Ok(self.agents.get(claim).cloned())
    }
    async fn delegated_agent_team(
        &self,
        delegated: &DelegatedTeam<'_>,
    ) -> Result<Option<String>, Error> {
        let allowed = delegated.allowed_team_ids.map_or("-".to_owned(), |ids| {
            ids.iter().cloned().collect::<Vec<_>>().join(",")
        });
        self.record(format!(
            "delegated {} team={} explicit={} allowed={allowed}",
            delegated.agent_id,
            or_dash(delegated.team_id),
            delegated.explicit_team
        ));
        Ok(self.delegated_team.map(str::to_owned))
    }
    async fn unpriced_models(&self) -> Result<Vec<String>, Error> {
        self.record("unpriced_models");
        Ok(self.unpriced.clone())
    }
    async fn agent_requires_trace_id(&self, agent_id: &str) -> Result<bool, Error> {
        self.record(format!("agent_trace {agent_id}"));
        Ok(self.agent_requires_trace_id)
    }
    async fn has_trace_id(&self, headers_only: bool) -> Result<bool, Error> {
        self.record(format!("has_trace_id {headers_only}"));
        Ok(self.has_trace_id)
    }
    async fn team_organization(&self, team_id: &str) -> Result<Option<String>, Error> {
        self.record(format!("team_organization {team_id}"));
        Ok(self.team_organizations.get(team_id).map(|org| (*org).to_owned()))
    }
    async fn merge_budget_tags(&self) -> Result<(), Error> {
        self.step("merge_budget_tags", "merge_budget_tags")
    }
    async fn check_team_model_access(&self, membership: Membership<'_, String>) -> Result<(), Error> {
        self.step(
            "check_team_model_access",
            format!("check_team_model_access {}", membership_label(membership)),
        )
    }
    async fn check_model_grants(&self) -> Result<(), Error> {
        self.step("check_model_grants", "check_model_grants")
    }
    async fn check_project(&self) -> Result<(), Error> {
        self.step("check_project", "check_project")
    }
    async fn stamp_access_groups(&self, membership: Membership<'_, String>) -> Result<Vec<String>, Error> {
        self.step(
            "stamp_access_groups",
            format!("stamp_access_groups {}", membership_label(membership)),
        )?;
        Ok(Vec::new())
    }
    async fn spend(&self, counter: SpendCounter<'_>) -> Result<f64, Error> {
        let label = spend_label(counter);
        self.step("spend", format!("spend {label}"))?;
        Ok(self.spends.get(label.as_str()).copied().unwrap_or(0.0))
    }
    async fn organization_budget(&self, org_id: &str) -> Result<Option<RecordedBudget>, Error> {
        self.step("organization_budget", format!("organization_budget {org_id}"))?;
        Ok(self.organization_budgets.get(org_id).copied())
    }
    async fn tag_budgets(&self) -> Result<Vec<NamedBudget>, Error> {
        self.step("tag_budgets", "tag_budgets")?;
        Ok(self.tag_budgets.clone())
    }
    async fn model_access_group_budgets(&self, groups: &[String]) -> Result<Vec<NamedBudget>, Error> {
        self.step(
            "model_access_group_budgets",
            format!("model_access_group_budgets {}", groups.join(",")),
        )?;
        Ok(self.model_access_group_budgets.clone())
    }
    async fn team_member_budget_state(&self, membership: Option<&String>) -> Result<MemberBudget, Error> {
        self.step(
            "team_member_budget_state",
            format!("team_member_budget_state {}", or_dash(membership.map(String::as_str))),
        )?;
        Ok(self.member_budget)
    }
    async fn budget_alert(&self, alert: BudgetAlert<'_>) -> Result<(), Error> {
        self.step("budget_alert", format!("alert {}", <&str>::from(alert)))
    }
    async fn check_guardrail_modification(&self) -> Result<(), Error> {
        self.step("check_guardrail_modification", "check_guardrail_modification")
    }
    async fn check_object_access(&self) -> Result<(), Error> {
        self.step("check_object_access", "check_object_access")
    }
    async fn custom_validate(&self, _: &Value) -> Result<bool, Error> {
        self.record("custom_validate");
        Ok(self.custom_validate_passes)
    }

    async fn team_model_access(&self, team: &String, model: &str) -> Result<ModelAccess, Error> {
        self.record(format!("model {team} {model}"));
        Ok(if self.model_access.contains(team.as_str()) {
            ModelAccess::Allowed
        } else {
            ModelAccess::Denied
        })
    }
}
