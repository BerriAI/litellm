//! Projects the shared spend delivery payload into the rows Python's accounting worker updates.

use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, NaiveDateTime};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;

/// The order in which Python's accounting writer takes entity row locks.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum Entity {
    Organization,
    OrganizationMember,
    Team,
    User,
    EndUser,
    TeamMember,
    Key,
    Project,
    Tag,
    ModelAccessGroup,
    Agent,
    AppGroup,
    AppGroupCustomerOrg,
    App,
    AppCustomerOrg,
    AppCustomerMember,
    CustomerOrg,
    CustomerTeam,
    CustomerMember,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct Entities {
    api_key: Option<String>,
    user: Option<String>,
    proxy_budget_name: Option<String>,
    team_id: Option<String>,
    organization_id: Option<String>,
    end_user: Option<String>,
    agent_id: Option<String>,
    project_id: Option<String>,
    app_id: Option<String>,
    app_group_id: Option<String>,
    #[serde(flatten)]
    customer: Customer,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct Customer {
    customer_org_id: Option<String>,
    customer_team_id: Option<String>,
    customer_member_id: Option<String>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
struct Metadata {
    #[serde(deserialize_with = "structured")]
    accounting_entities: Entities,
    accounting_model_access_groups: Vec<String>,
    accounting_windows: Vec<Window>,
    user_api_key_customer_usage: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Structured<T> {
    Encoded(String),
    Decoded(T),
    Empty(()),
}

fn structured<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: DeserializeOwned + Default,
{
    match Structured::deserialize(deserializer)? {
        Structured::Encoded(text) => serde_json::from_str(&text).map_err(serde::de::Error::custom),
        Structured::Decoded(value) => Ok(value),
        Structured::Empty(()) => Ok(T::default()),
    }
}

/// The subset of a persisted spend log needed for its accounting delivery.
#[derive(Default, Deserialize)]
#[serde(default)]
pub(super) struct Payload {
    spend: Option<f64>,
    #[serde(rename = "startTime")]
    start_time: Option<String>,
    #[serde(deserialize_with = "structured")]
    metadata: Metadata,
    #[serde(deserialize_with = "structured")]
    request_tags: Vec<String>,
    #[serde(skip)]
    customer_fallback: Customer,
}

#[derive(Deserialize)]
struct Window {
    entity_type: String,
    entity_id: String,
    window_duration: String,
    window_start: String,
    spend: Option<f64>,
}

#[derive(Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct WindowKey<'a> {
    pub(super) entity_type: &'a str,
    pub(super) entity_id: &'a str,
    pub(super) duration: &'a str,
    pub(super) starts_at: NaiveDateTime,
}

#[derive(Default)]
pub(super) struct WindowCharge {
    pub(super) spend: f64,
    pub(super) earliest_request: Option<NaiveDateTime>,
}

pub(super) fn parse_timestamp(text: &str) -> Option<NaiveDateTime> {
    DateTime::parse_from_rfc3339(text)
        .map(|date| date.naive_utc())
        .ok()
        .or_else(|| text.parse().ok())
}

pub(super) fn windows(
    payloads: &[Payload],
) -> Result<BTreeMap<WindowKey<'_>, WindowCharge>, litellm_db::Error> {
    let mut totals: BTreeMap<_, WindowCharge> = BTreeMap::new();
    for payload in payloads {
        let earliest_request = payload.start_time.as_deref().map(parse_timestamp);
        for window in &payload.metadata.accounting_windows {
            if [
                &window.entity_type,
                &window.entity_id,
                &window.window_duration,
            ]
            .into_iter()
            .any(|part| part.len() > 2000)
            {
                continue;
            }
            let starts_at =
                parse_timestamp(&window.window_start).ok_or_else(invalid_window_timestamp)?;
            let earliest_request = earliest_request
                .map(|start| start.ok_or_else(invalid_window_timestamp))
                .transpose()?;
            let total = totals
                .entry(WindowKey {
                    entity_type: &window.entity_type,
                    entity_id: &window.entity_id,
                    duration: &window.window_duration,
                    starts_at,
                })
                .or_default();
            total.spend += window.spend.or(payload.spend).unwrap_or_default();
            if let Some(start) = earliest_request {
                total.earliest_request = Some(
                    total
                        .earliest_request
                        .map_or(start, |prior| prior.min(start)),
                );
            }
        }
    }
    Ok(totals)
}

fn invalid_window_timestamp() -> litellm_db::Error {
    sqlx::Error::Encode("invalid accounting window timestamp".into()).into()
}

impl Payload {
    pub(super) fn decode(value: Value) -> Result<Self, serde_json::Error> {
        let mut payload: Self = serde_json::from_value(value)?;
        payload.customer_fallback = payload
            .metadata
            .user_api_key_customer_usage
            .as_deref()
            .and_then(|value| serde_json::from_str(value).ok())
            .unwrap_or_default();
        Ok(payload)
    }

    fn customer(&self) -> &Customer {
        let customer = &self.metadata.accounting_entities.customer;
        if customer.customer_org_id.is_some() {
            customer
        } else {
            &self.customer_fallback
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct EntityCharge<'a> {
    id: &'a str,
    member_id: Option<&'a str>,
    spend: f64,
}

type Identity<'a> = (&'a str, Option<&'a str>);

#[derive(Default)]
pub(super) struct EntityCharges<'a>(BTreeMap<Entity, BTreeMap<Identity<'a>, f64>>);

impl<'a> EntityCharges<'a> {
    pub(super) fn project(payloads: &'a [Payload]) -> Self {
        let mut charges = Self::default();
        let mut app_charges = Self::default();
        for payload in payloads {
            let metadata = &payload.metadata;
            let entities = &metadata.accounting_entities;
            let customer = payload.customer();
            let spend = payload.spend.unwrap_or_default();
            let user = entities.user.as_deref();
            let team = entities.team_id.as_deref();
            let org = entities.organization_id.as_deref();
            for (kind, id) in [
                (Entity::Organization, org),
                (Entity::Team, team),
                (Entity::User, user),
                (Entity::User, entities.proxy_budget_name.as_deref()),
                (Entity::EndUser, entities.end_user.as_deref()),
                (Entity::Key, entities.api_key.as_deref()),
                (Entity::Project, entities.project_id.as_deref()),
                (Entity::Agent, entities.agent_id.as_deref()),
                (Entity::CustomerOrg, customer.customer_org_id.as_deref()),
                (Entity::CustomerTeam, customer.customer_team_id.as_deref()),
                (
                    Entity::CustomerMember,
                    customer.customer_member_id.as_deref(),
                ),
            ] {
                charges.add(kind, id, None, spend);
            }
            if user.is_some() {
                charges.add(Entity::TeamMember, team, user, spend);
                charges.add(Entity::OrganizationMember, org, user, spend);
            }
            for group in &metadata.accounting_model_access_groups {
                charges.add(Entity::ModelAccessGroup, Some(group), None, spend);
            }
            for tag in payload.request_tags.iter().filter(|tag| !tag.is_empty()) {
                charges.add(Entity::Tag, Some(tag), None, spend);
            }
            let app = entities.app_id.as_deref().filter(|id| !id.is_empty());
            let group = entities.app_group_id.as_deref().filter(|id| !id.is_empty());
            app_charges.add(Entity::App, app, None, spend);
            app_charges.add(Entity::AppGroup, group, None, spend);
            if let Some(customer_org) = customer.customer_org_id.as_deref() {
                app_charges.add(Entity::AppCustomerOrg, app, Some(customer_org), spend);
                app_charges.add(
                    Entity::AppGroupCustomerOrg,
                    group,
                    Some(customer_org),
                    spend,
                );
            }
            if let Some(customer_member) = customer.customer_member_id.as_deref() {
                app_charges.add(Entity::AppCustomerMember, app, Some(customer_member), spend);
            }
        }
        if app_charges.0.contains_key(&Entity::App) {
            charges.0.append(&mut app_charges.0);
        }
        charges
    }

    fn add(&mut self, entity: Entity, id: Option<&'a str>, member_id: Option<&'a str>, spend: f64) {
        if let Some(id) = id.filter(|id| id.len() <= 2000)
            && member_id.is_none_or(|id| id.len() <= 2000)
        {
            *self
                .0
                .entry(entity)
                .or_default()
                .entry((id, member_id))
                .or_default() += spend;
        }
    }

    pub(super) fn teams(&self) -> Vec<String> {
        [Entity::Team, Entity::TeamMember]
            .into_iter()
            .filter_map(|kind| self.0.get(&kind))
            .flat_map(|rows| rows.keys().map(|&(id, _)| id))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(str::to_owned)
            .collect()
    }

    pub(super) fn groups(&self) -> impl Iterator<Item = (Entity, Vec<EntityCharge<'a>>)> + '_ {
        self.0.iter().map(|(&entity, rows)| {
            let charges = rows
                .iter()
                .map(|(&(id, member_id), &spend)| EntityCharge {
                    id,
                    member_id,
                    spend,
                })
                .collect();
            (entity, charges)
        })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::{Entity, EntityCharges, Payload};

    #[rstest]
    #[case::project(Entity::Project, "project", None, 4.0)]
    #[case::owner_and_proxy_share_row(Entity::User, "owner", None, 8.0)]
    #[case::team_member(Entity::TeamMember, "team", Some("owner"), 4.0)]
    #[case::organization_member(Entity::OrganizationMember, "org", Some("owner"), 4.0)]
    #[case::customer_fallback(Entity::CustomerTeam, "customer-team", None, 4.0)]
    #[case::app_member(Entity::AppCustomerMember, "app", Some("customer-member"), 4.0)]
    fn accounting_entities_aggregate_by_destination(
        #[case] entity: Entity,
        #[case] id: &str,
        #[case] member_id: Option<&str>,
        #[case] spend: f64,
    ) {
        let delivery = || {
            Payload::decode(json!({
            "spend": 2.0,
            "user": "not-the-accounting-user",
            "metadata": {
                "accounting_entities": {
                    "user": "owner", "proxy_budget_name": "owner", "team_id": "team",
                    "project_id": "project", "organization_id": "org", "app_id": "app",
                },
                "user_api_key_customer_usage": "{\"customer_org_id\":\"customer-org\",\"customer_team_id\":\"customer-team\",\"customer_member_id\":\"customer-member\"}",
            },
        })).unwrap()
        };
        let payloads = [delivery(), delivery()];

        let charges = EntityCharges::project(&payloads);

        assert_eq!(charges.0[&entity][&(id, member_id)], spend);
        assert_eq!(charges.teams(), ["team"]);
    }
}
