use time::{OffsetDateTime, UtcOffset};

use crate::{
    Batch, Cost, CounterKey, DailyEntity, DailyKey, DailyRollups, DailyTally, EntityKey, Totals,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Success,
    Failure,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Attribution {
    pub key: Option<String>,
    pub user: Option<String>,
    pub team: Option<String>,
    pub organization: Option<String>,
    pub end_user: Option<String>,
    pub project: Option<String>,
    pub agent: Option<String>,
    pub tags: Vec<String>,
    pub model_access_groups: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cache_read_input_tokens: u64,
    pub cache_creation_input_tokens: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpendEvent {
    pub cost: f64,
    pub attribution: Attribution,
    pub started_at: OffsetDateTime,
    pub model: Option<String>,
    pub model_group: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub endpoint: Option<String>,
    pub mcp_namespaced_tool_name: Option<String>,
    pub usage: Usage,
    pub outcome: Outcome,
    pub response_time_ms: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Charges {
    pub counters: Vec<(CounterKey, Cost)>,
    pub totals: Totals,
    pub daily: DailyRollups,
}

pub fn charges(event: &SpendEvent) -> Charges {
    let cost = Cost(event.cost);
    let entities = charged_entities(&event.attribution);
    Charges {
        counters: entities
            .iter()
            .filter(|entity| has_budget_counter(entity))
            .map(|entity| (CounterKey::lifetime(entity.clone()), cost))
            .collect(),
        totals: Batch::from_entries(entities.into_iter().map(|entity| (entity, cost))),
        daily: Batch::from_entries(
            daily_entities(&event.attribution)
                .into_iter()
                .map(|entity| (daily_key(event, entity), daily_tally(event))),
        ),
    }
}

fn charged_entities(who: &Attribution) -> Vec<EntityKey> {
    let team_member = who
        .team
        .as_ref()
        .zip(who.user.as_ref())
        .map(|(team, user)| EntityKey::TeamMember {
            team_id: team.clone(),
            user_id: user.clone(),
        });
    let organization_member =
        who.organization
            .as_ref()
            .zip(who.user.as_ref())
            .map(|(organization, user)| EntityKey::OrganizationMember {
                organization_id: organization.clone(),
                user_id: user.clone(),
            });
    [
        who.key.clone().map(EntityKey::Key),
        who.user.clone().map(EntityKey::User),
        who.end_user.clone().map(EntityKey::EndUser),
        who.team.clone().map(EntityKey::Team),
        team_member,
        who.organization.clone().map(EntityKey::Organization),
        organization_member,
        who.project.clone().map(EntityKey::Project),
        who.agent.clone().map(EntityKey::Agent),
    ]
    .into_iter()
    .flatten()
    .chain(who.tags.iter().cloned().map(EntityKey::Tag))
    .chain(
        who.model_access_groups
            .iter()
            .cloned()
            .map(EntityKey::ModelAccessGroup),
    )
    .collect()
}

fn has_budget_counter(entity: &EntityKey) -> bool {
    match entity {
        EntityKey::Key(_)
        | EntityKey::User(_)
        | EntityKey::EndUser(_)
        | EntityKey::Team(_)
        | EntityKey::TeamMember { .. }
        | EntityKey::Organization(_)
        | EntityKey::Project(_)
        | EntityKey::Tag(_)
        | EntityKey::ModelAccessGroup(_) => true,
        EntityKey::OrganizationMember { .. } | EntityKey::Agent(_) => false,
    }
}

fn daily_entities(who: &Attribution) -> Vec<DailyEntity> {
    [
        who.user.clone().map(DailyEntity::User),
        who.team.clone().map(DailyEntity::Team),
        who.organization.clone().map(DailyEntity::Organization),
        who.end_user.clone().map(DailyEntity::EndUser),
        who.agent.clone().map(DailyEntity::Agent),
    ]
    .into_iter()
    .flatten()
    .chain(who.tags.iter().cloned().map(DailyEntity::Tag))
    .collect()
}

fn daily_key(event: &SpendEvent, entity: DailyEntity) -> DailyKey {
    DailyKey {
        date: event.started_at.to_offset(UtcOffset::UTC).date(),
        entity,
        api_key: event.attribution.key.clone().unwrap_or_default(),
        model: event.model.clone(),
        custom_llm_provider: event.custom_llm_provider.clone(),
        mcp_namespaced_tool_name: event.mcp_namespaced_tool_name.clone(),
        endpoint: event.endpoint.clone(),
    }
}

fn daily_tally(event: &SpendEvent) -> DailyTally {
    let succeeded = event.outcome == Outcome::Success;
    DailyTally {
        model_group: event.model_group.clone(),
        spend: event.cost,
        prompt_tokens: event.usage.prompt_tokens,
        completion_tokens: event.usage.completion_tokens,
        cache_read_input_tokens: event.usage.cache_read_input_tokens,
        cache_creation_input_tokens: event.usage.cache_creation_input_tokens,
        api_requests: 1,
        successful_requests: u64::from(succeeded),
        failed_requests: u64::from(!succeeded),
        total_response_time_ms: event.response_time_ms.unwrap_or_default(),
        timed_requests: u64::from(event.response_time_ms.is_some()),
        ..DailyTally::default()
    }
}
