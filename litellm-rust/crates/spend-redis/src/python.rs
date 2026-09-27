use std::collections::BTreeMap;

use litellm_spend::{Cost, CounterKey, EntityKey, Totals};
use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, percent_decode_str, utf8_percent_encode};
use serde::{Deserialize, Serialize};

use crate::{BatchCodec, CounterNaming, PythonFormatError};

const SEPARATOR: &str = "::";
const QUOTED: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'_')
    .remove(b'.')
    .remove(b'-')
    .remove(b'~');

#[derive(Clone, Copy, Debug, Default)]
pub struct PythonCounterNaming;

impl CounterNaming for PythonCounterNaming {
    fn counter(&self, key: &CounterKey) -> Option<String> {
        let lifetime = match &key.entity {
            EntityKey::Key(token) => format!("spend:key:{token}"),
            EntityKey::User(id) => format!("spend:user:{id}"),
            EntityKey::EndUser(id) => format!("spend:end_user:{id}"),
            EntityKey::Team(id) => format!("spend:team:{id}"),
            EntityKey::TeamMember { team_id, user_id } => {
                format!("spend:team_member:{user_id}:{team_id}")
            }
            EntityKey::Organization(id) => format!("spend:org:{id}"),
            EntityKey::Project(id) => format!("spend:project:{id}"),
            EntityKey::Tag(name) => format!("spend:tag:{name}"),
            EntityKey::ModelAccessGroup(name) => format!("spend:model_access_group:{name}"),
            EntityKey::OrganizationMember { .. } | EntityKey::Agent(_) => return None,
        };
        Some(match &key.window {
            Some(duration) => format!("{lifetime}:window:{duration}"),
            None => lifetime,
        })
    }
}

type Spend = BTreeMap<String, f64>;

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct Transactions {
    user_list_transactions: Option<Spend>,
    end_user_list_transactions: Option<Spend>,
    key_list_transactions: Option<Spend>,
    team_list_transactions: Option<Spend>,
    team_member_list_transactions: Option<Spend>,
    org_list_transactions: Option<Spend>,
    org_member_list_transactions: Option<Spend>,
    project_list_transactions: Option<Spend>,
    tag_list_transactions: Option<Spend>,
    agent_list_transactions: Option<Spend>,
    model_access_group_list_transactions: Option<Spend>,
}

impl Transactions {
    fn slot(&mut self, entity: &EntityKey) -> &mut Spend {
        match entity {
            EntityKey::User(_) => &mut self.user_list_transactions,
            EntityKey::EndUser(_) => &mut self.end_user_list_transactions,
            EntityKey::Key(_) => &mut self.key_list_transactions,
            EntityKey::Team(_) => &mut self.team_list_transactions,
            EntityKey::TeamMember { .. } => &mut self.team_member_list_transactions,
            EntityKey::Organization(_) => &mut self.org_list_transactions,
            EntityKey::OrganizationMember { .. } => &mut self.org_member_list_transactions,
            EntityKey::Project(_) => &mut self.project_list_transactions,
            EntityKey::Tag(_) => &mut self.tag_list_transactions,
            EntityKey::Agent(_) => &mut self.agent_list_transactions,
            EntityKey::ModelAccessGroup(_) => &mut self.model_access_group_list_transactions,
        }
        .get_or_insert_default()
    }

    fn entries(self) -> Result<Vec<(EntityKey, Cost)>, PythonFormatError> {
        let plain = [
            (
                self.user_list_transactions,
                EntityKey::User as fn(String) -> EntityKey,
            ),
            (self.end_user_list_transactions, EntityKey::EndUser),
            (self.key_list_transactions, EntityKey::Key),
            (self.team_list_transactions, EntityKey::Team),
            (self.org_list_transactions, EntityKey::Organization),
            (self.project_list_transactions, EntityKey::Project),
            (self.tag_list_transactions, EntityKey::Tag),
            (self.agent_list_transactions, EntityKey::Agent),
            (
                self.model_access_group_list_transactions,
                EntityKey::ModelAccessGroup,
            ),
        ]
        .into_iter()
        .flat_map(|(spend, entity)| {
            spend
                .unwrap_or_default()
                .into_iter()
                .map(move |(id, cost)| Ok((entity(id), Cost(cost))))
        });
        let team_members = self
            .team_member_list_transactions
            .unwrap_or_default()
            .into_iter()
            .map(|(key, cost)| Ok((team_member(&key)?, Cost(cost))));
        let organization_members = self
            .org_member_list_transactions
            .unwrap_or_default()
            .into_iter()
            .map(|(key, cost)| Ok((organization_member(&key)?, Cost(cost))));
        plain
            .chain(team_members)
            .chain(organization_members)
            .collect()
    }
}

fn id_of(entity: &EntityKey) -> Result<String, PythonFormatError> {
    match entity {
        EntityKey::User(id)
        | EntityKey::EndUser(id)
        | EntityKey::Key(id)
        | EntityKey::Team(id)
        | EntityKey::Organization(id)
        | EntityKey::Project(id)
        | EntityKey::Tag(id)
        | EntityKey::Agent(id)
        | EntityKey::ModelAccessGroup(id) => Ok(id.clone()),
        EntityKey::TeamMember { team_id, user_id } => {
            let (team_id, user_id) = (unambiguous(team_id)?, unambiguous(user_id)?);
            Ok(format!("team_id::{team_id}::user_id::{user_id}"))
        }
        EntityKey::OrganizationMember {
            organization_id,
            user_id,
        } => Ok(format!(
            "organization_id::{}::user_id::{}",
            utf8_percent_encode(organization_id, QUOTED),
            utf8_percent_encode(user_id, QUOTED)
        )),
    }
}

fn unambiguous(id: &str) -> Result<&str, PythonFormatError> {
    if id.contains(SEPARATOR) {
        return Err(PythonFormatError::SeparatorInId(id.to_owned()));
    }
    Ok(id)
}

fn member_parts<'a>(
    key: &'a str,
    group_label: &str,
) -> Result<(&'a str, &'a str), PythonFormatError> {
    let parts: Vec<_> = key.split(SEPARATOR).collect();
    match parts.as_slice() {
        [label, group, "user_id", user] if *label == group_label => Ok((group, user)),
        _ => Err(PythonFormatError::MemberKey(key.to_owned())),
    }
}

fn team_member(key: &str) -> Result<EntityKey, PythonFormatError> {
    let (team_id, user_id) = member_parts(key, "team_id")?;
    Ok(EntityKey::TeamMember {
        team_id: team_id.to_owned(),
        user_id: user_id.to_owned(),
    })
}

fn unquoted(key: &str, part: &str) -> Result<String, PythonFormatError> {
    percent_decode_str(part)
        .decode_utf8()
        .map(|decoded| decoded.into_owned())
        .map_err(|_| PythonFormatError::MemberKey(key.to_owned()))
}

fn organization_member(key: &str) -> Result<EntityKey, PythonFormatError> {
    let (organization_id, user_id) = member_parts(key, "organization_id")?;
    Ok(EntityKey::OrganizationMember {
        organization_id: unquoted(key, organization_id)?,
        user_id: unquoted(key, user_id)?,
    })
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PythonTotalsCodec;

impl BatchCodec<EntityKey, Cost> for PythonTotalsCodec {
    type Error = PythonFormatError;

    fn list(&self) -> &str {
        "litellm_spend_update_buffer"
    }

    fn encode(&self, batch: &Totals) -> Result<String, Self::Error> {
        let mut transactions = Transactions::default();
        for (entity, cost) in batch.iter() {
            transactions.slot(entity).insert(id_of(entity)?, cost.0);
        }
        Ok(serde_json::to_string(&transactions)?)
    }

    fn decode(&self, blob: &str) -> Result<Totals, Self::Error> {
        let transactions: Transactions = serde_json::from_str(blob)?;
        Ok(Totals::from_entries(transactions.entries()?))
    }
}
