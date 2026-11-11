//! The registry lookups the host serves as `registry()` queries, answered over the deployments and
//! the host indexes ([`crate::host_index::HostIndexes`]) the way the Python router answers them.
//!
//! A lookup answers with [`Row`]s: positions in the deployment list (each with how the row renames
//! the deployment), wildcard routes, or a position the Python list could not be read at. The host
//! turns a position into the entry object it stores, so identity and in-place reads stay its own;
//! a [`Row::Raw`] is handed back for the host to index its list by, which raises exactly where the
//! Python router's own read raised (an index a caller wrote junk into).

use std::sync::Arc;

use serde_json::{Map, Value, json};

use crate::{
    host_index::{HostIndexes, python_str, truthy},
    registry::{Deployment, Registry},
};

/// One row of a lookup's answer.
#[derive(Clone, Debug, PartialEq)]
pub enum Row {
    /// The deployment at `position`, under `name` when the row renames it (an alias or a routing
    /// group), as a routing group's member `member` of `group` when `group` is set (its row drops
    /// the member's `access_groups` and takes the group's priority as its `order`).
    At {
        position: usize,
        name: Option<String>,
        group: Option<(String, String)>,
    },
    /// A wildcard route: a copy of a pattern router's deployment, under `name`.
    Pattern { entry: Value, name: String },
    /// A position the list cannot be read at: the host indexes its list by it, and raises.
    Raw(Value),
    /// A position container that is not a list: the host iterates it as its list's positions.
    RawPositions(Value),
}

impl Row {
    /// The row as the host reads it.
    pub fn wire(&self) -> Value {
        match self {
            Self::At {
                position,
                name,
                group,
            } => {
                let mut row = Map::new();
                row.insert("at".to_owned(), json!(position));
                if let Some(name) = name {
                    row.insert("name".to_owned(), json!(name));
                }
                if let Some((group, member)) = group {
                    row.insert("group".to_owned(), json!(group));
                    row.insert("member".to_owned(), json!(member));
                }
                Value::Object(row)
            }
            Self::Pattern { entry, name } => json!({"pattern": entry, "name": name}),
            Self::Raw(value) => json!({"raw": value}),
            Self::RawPositions(value) => json!({"raw_positions": value}),
        }
    }
}

/// An alias `model_list` lists: its name and the group it targets (`hidden` aliases already left
/// out by the host when the whole list is asked for).
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
pub struct AliasItem {
    pub alias: String,
    pub target: String,
}

/// A routing group callable by its name (the host leaves out one a deployment or an alias
/// shadows): its name and member model groups.
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
pub struct CallableGroup {
    pub name: String,
    pub members: Vec<String>,
}

/// What `get_model_list` composes beyond the deployments: the aliases and callable routing groups
/// it lists, and the wildcard requests a name no deployment serves is routed by.
#[derive(Clone, Debug, Default, PartialEq, serde::Deserialize)]
#[serde(default)]
pub struct ListScope {
    pub aliases: Vec<AliasItem>,
    pub groups: Vec<CallableGroup>,
    /// `get_deployments_by_pattern`'s requests: the name, then the name under its provider.
    pub wildcard: Vec<String>,
    /// Only one part of the list: `"aliases"` (`_core_get_model_list_from_model_alias`) or
    /// `"groups"` (`_core_get_model_list_from_routing_groups`).
    pub only: Option<String>,
}

/// A row read for a lookup's filters: the deployment's fields as the row shows them.
struct View<'a> {
    name: &'a str,
    info: &'a Map<String, Value>,
    params: &'a Map<String, Value>,
    strips_access_groups: bool,
}

/// The lookups over one registry version.
pub struct Lookup<'a> {
    deployments: &'a [Arc<Deployment>],
    host: &'a HostIndexes,
}

impl<'a> Lookup<'a> {
    pub fn new(registry: &'a Registry) -> Self {
        Self {
            deployments: registry.deployments_from(0),
            host: registry.host_indexes(),
        }
    }

    /// `list[value]` as Python reads it: an int (or bool) in range, negative from the end.
    fn position(&self, value: &Value) -> Option<usize> {
        let len = i64::try_from(self.deployments.len()).ok()?;
        let index = match value {
            Value::Bool(flag) => i64::from(*flag),
            Value::Number(number) => number.as_i64()?,
            _ => return None,
        };
        let index = if index < 0 { len + index } else { index };
        (0..len)
            .contains(&index)
            .then(|| usize::try_from(index).ok())
            .flatten()
    }

    /// Every position `held` (what an index holds) names, read before any is filtered, as the
    /// Python router reads them into a tuple first: `Err` with the row that raises.
    fn positions(&self, held: &Value) -> Result<Vec<usize>, Row> {
        let items = match held {
            Value::Array(items) => items,
            // Iterating an empty string or dict reads no position; anything else raises.
            Value::String(text) if text.is_empty() => return Ok(Vec::new()),
            Value::Object(map) if map.is_empty() => return Ok(Vec::new()),
            _ => return Err(Row::RawPositions(held.clone())),
        };
        items
            .iter()
            .map(|item| self.position(item).ok_or_else(|| Row::Raw(item.clone())))
            .collect()
    }

    fn view(&self, position: usize) -> View<'a> {
        let deployment = &self.deployments[position];
        View {
            name: &deployment.model_name,
            info: &deployment.model_info.raw,
            params: &deployment.raw_params,
            strips_access_groups: false,
        }
    }

    fn row_view<'r>(&self, row: &'r Row) -> Option<View<'r>>
    where
        'a: 'r,
    {
        match row {
            Row::At {
                position,
                name,
                group,
            } => {
                let base = self.view(*position);
                Some(View {
                    name: name.as_deref().unwrap_or(base.name),
                    strips_access_groups: group.is_some(),
                    ..base
                })
            }
            Row::Pattern { entry, name } => Some(View {
                name,
                info: entry
                    .get("model_info")
                    .and_then(Value::as_object)
                    .unwrap_or(empty()),
                params: entry
                    .get("litellm_params")
                    .and_then(Value::as_object)
                    .unwrap_or(empty()),
                strips_access_groups: false,
            }),
            Row::Raw(_) | Row::RawPositions(_) => None,
        }
    }

    // -- deployments by name -------------------------------------------------------------------

    /// `_core_get_all_deployments`: the team's deployments under public name `model_name`, else
    /// the group's, else (for a team) every deployment the name includes; each under `rename`.
    pub fn all(&self, model_name: &str, team_id: Option<&str>, rename: Option<&str>) -> Vec<Row> {
        let rows = |positions: Vec<usize>| -> Vec<Row> {
            positions
                .into_iter()
                .filter(|position| should_include(model_name, &self.view(*position), team_id))
                .map(|position| Row::At {
                    position,
                    name: rename.map(str::to_owned),
                    group: None,
                })
                .collect()
        };
        if let Some(team) = team_id
            && let Some(held) = self.host.team_named(team, model_name)
        {
            match self.positions(held) {
                Ok(positions) => {
                    let found = rows(positions);
                    if !found.is_empty() {
                        return found;
                    }
                }
                Err(raw) => return vec![raw],
            }
        }
        if let Some(held) = self.host.named(model_name) {
            return match self.positions(held) {
                Ok(positions) => rows(positions),
                Err(raw) => vec![raw],
            };
        }
        if team_id.is_some() {
            return rows((0..self.deployments.len()).collect());
        }
        Vec::new()
    }

    /// `_team_deployments_across_teams`: every team's deployments under public name `model`, and
    /// how many distinct teams own them.
    pub fn across_teams(&self, model: &str) -> (Vec<Row>, usize) {
        let mut rows = Vec::new();
        for (_, public, held) in &self.host.team_model {
            if public != model {
                continue;
            }
            match self.positions(held) {
                Ok(positions) => rows.extend(positions.into_iter().map(|position| Row::At {
                    position,
                    name: None,
                    group: None,
                })),
                Err(raw) => return (vec![raw], 0),
            }
        }
        let mut teams: Vec<&Value> = Vec::new();
        for row in &rows {
            if let Some(view) = self.row_view(row)
                && let Some(team) = view.info.get("team_id").filter(|team| !team.is_null())
                && !teams.contains(&team)
            {
                teams.push(team);
            }
        }
        let count = teams.len();
        (rows, count)
    }

    /// `_core_config_deployments`: the deployments a config declared (no `db_model: true`).
    pub fn config(&self) -> Vec<Row> {
        (0..self.deployments.len())
            .filter(|position| {
                self.view(*position).info.get("db_model") != Some(&Value::Bool(true))
            })
            .map(|position| Row::At {
                position,
                name: None,
                group: None,
            })
            .collect()
    }

    /// `_core_get_model_ids`: the ids of the group's deployments, or (no group) of the id map's
    /// entries, without team deployments when `exclude_team_models`. `Err` with the row that raises.
    pub fn model_ids(
        &self,
        model_name: Option<&str>,
        exclude_team_models: bool,
    ) -> Result<Vec<String>, Row> {
        let kept = |view: &View<'_>| {
            view.info.contains_key("id")
                && !(exclude_team_models && view.info.get("team_id").is_some_and(truthy))
        };
        if let Some(name) = model_name {
            let Some(held) = self.host.named(name) else {
                return Ok(Vec::new());
            };
            return Ok(self
                .positions(held)?
                .into_iter()
                .map(|position| self.view(position))
                .filter(kept)
                .map(|view| python_str(&view.info["id"]))
                .collect());
        }
        let mut ids = Vec::new();
        for (id, held) in &self.host.id_map {
            let position = self.position(held).ok_or_else(|| Row::Raw(held.clone()))?;
            if kept(&self.view(position)) {
                ids.push(id.clone());
            }
        }
        Ok(ids)
    }

    // -- the model list --------------------------------------------------------------------------

    /// `_core_get_model_list`: the name's deployments, its alias rows and routing group rows, its
    /// wildcard routes when nothing else serves it, or (no name) every alias and group row and
    /// then every deployment.
    pub fn model_list(
        &self,
        model_name: Option<&str>,
        team_id: Option<&str>,
        scope: &ListScope,
    ) -> Vec<Row> {
        let only = scope.only.as_deref();
        let mut rows = Vec::new();
        if let Some(name) = model_name
            && only.is_none()
        {
            rows.extend(self.all(name, team_id, None));
        }
        let aliases: &[AliasItem] = if only == Some("groups") {
            &[]
        } else {
            &scope.aliases
        };
        for alias in aliases {
            match scope.groups.iter().find(|group| group.name == alias.target) {
                Some(group) => {
                    rows.extend(self.group_rows(group).into_iter().map(|row| match row {
                        Row::At {
                            position, group, ..
                        } => Row::At {
                            position,
                            name: Some(alias.alias.clone()),
                            group,
                        },
                        other => other,
                    }))
                }
                None => rows.extend(self.all(&alias.target, None, Some(&alias.alias))),
            }
        }
        match (model_name, only) {
            (_, Some("aliases")) => {}
            (Some(name), _) => {
                if let Some(group) = scope.groups.iter().find(|group| group.name == name) {
                    rows.extend(self.group_rows(group));
                }
            }
            (None, _) => {
                for group in &scope.groups {
                    rows.extend(self.group_rows(group));
                }
            }
        }
        if only.is_some() {
            return rows;
        }
        if let Some(name) = model_name
            && rows.is_empty()
        {
            rows.extend(self.wildcard(name, team_id, &scope.wildcard));
        }
        if model_name.is_none() {
            rows.extend((0..self.deployments.len()).map(|position| Row::At {
                position,
                name: None,
                group: None,
            }));
        }
        rows
    }

    /// The whole list's rows, or the first row the host's read of the list raises at.
    fn readable(&self, rows: Vec<Row>) -> Result<Vec<Row>, Row> {
        match rows
            .iter()
            .find(|row| matches!(row, Row::Raw(_) | Row::RawPositions(_)))
        {
            Some(raw) => Err(raw.clone()),
            None => Ok(rows),
        }
    }

    /// A callable routing group's rows: each member's deployments under the group's name.
    fn group_rows(&self, group: &CallableGroup) -> Vec<Row> {
        let mut rows = Vec::new();
        for member in &group.members {
            for row in self.all(member, None, Some(&group.name)) {
                rows.push(match row {
                    Row::At { position, name, .. } => Row::At {
                        position,
                        name,
                        group: Some((group.name.clone(), member.clone())),
                    },
                    other => other,
                });
            }
        }
        rows
    }

    /// `_get_wildcard_deployments`: the global router's routes for the requests, then the team's.
    fn wildcard(&self, name: &str, team_id: Option<&str>, requests: &[String]) -> Vec<Row> {
        let route = |router: &crate::host_index::PatternRouter| -> Vec<Value> {
            requests
                .iter()
                .find_map(|request| {
                    router
                        .route(request, None)
                        .filter(|routed| !routed.is_empty())
                })
                .unwrap_or_default()
        };
        let mut routed = route(&self.host.patterns);
        if let Some(team) = team_id
            && let Some((_, router)) = self.host.team_patterns.iter().find(|(t, _)| t == team)
        {
            routed.extend(route(router));
        }
        routed
            .into_iter()
            .map(|entry| Row::Pattern {
                entry,
                name: name.to_owned(),
            })
            .collect()
    }

    pub fn has_team_wildcards(&self, team_id: &str) -> bool {
        self.host.team_patterns.iter().any(|(t, _)| t == team_id)
    }

    /// The deployments `team_id`'s own wildcard pattern routes `name` to
    /// (`_core_team_pattern_routers`, keyed by `team_public_model_name`).
    pub fn team_wildcard(&self, name: &str, team_id: &str) -> Vec<Arc<Deployment>> {
        let Some((_, router)) = self.host.team_patterns.iter().find(|(t, _)| t == team_id) else {
            return Vec::new();
        };
        router
            .route(name, None)
            .unwrap_or_default()
            .iter()
            .filter_map(|entry| entry.get("model_info")?.get("id").map(python_str))
            .filter_map(|id| self.deployments.iter().find(|d| d.id == id).cloned())
            .collect()
    }

    /// `_core_get_model_names`: per row of the whole list, a team deployment's public name for
    /// its own team, else the row's name.
    pub fn model_names(
        &self,
        team_id: Option<&str>,
        scope: &ListScope,
    ) -> Result<Vec<String>, Row> {
        let rows = self.readable(self.model_list(None, None, scope))?;
        let mut names = Vec::new();
        for row in &rows {
            let Some(view) = self.row_view(row) else {
                continue;
            };
            match view.info.get("team_id").filter(|team| truthy(team)) {
                Some(team) => {
                    if team_id.is_some_and(|id| team == &Value::String(id.to_owned()))
                        && let Some(public) = view
                            .info
                            .get("team_public_model_name")
                            .filter(|public| truthy(public))
                    {
                        names.push(python_str(public));
                    }
                }
                None => names.push(view.name.to_owned()),
            }
        }
        Ok(names)
    }

    /// `_core_get_model_access_groups`: access group to the names of the rows that declare it,
    /// in the order they appear.
    pub fn access_groups(
        &self,
        model_name: Option<&str>,
        access_group: Option<&str>,
        team_id: Option<&str>,
        scope: &ListScope,
    ) -> Result<Vec<(String, Vec<String>)>, Row> {
        let mut groups: Vec<(String, Vec<String>)> = Vec::new();
        for row in self.readable(self.model_list(model_name, team_id, scope))? {
            let Some(view) = self.row_view(&row) else {
                continue;
            };
            if view.info.is_empty() || view.strips_access_groups {
                continue;
            }
            // `access_groups or []`, iterated: a list's items, a string's characters.
            let declared: Vec<String> = match view.info.get("access_groups") {
                Some(Value::Array(items)) => items.iter().map(python_str).collect(),
                Some(Value::String(text)) => text.chars().map(String::from).collect(),
                _ => continue,
            };
            for group in declared {
                if access_group.is_some_and(|wanted| wanted != group) {
                    continue;
                }
                match groups.iter_mut().find(|(held, _)| *held == group) {
                    Some((_, names)) => names.push(view.name.to_owned()),
                    None => groups.push((group, vec![view.name.to_owned()])),
                }
            }
        }
        Ok(groups)
    }

    /// `_core_resolve_model_name_from_model_id`'s scan of the whole list: whether it lists any
    /// row, and the name of the first row whose `litellm_params.model` is `model_id`, or ends in
    /// `/model_id` or `:model_id`.
    pub fn name_serving(
        &self,
        model_id: &str,
        scope: &ListScope,
    ) -> Result<(bool, Option<String>), Row> {
        let rows = self.readable(self.model_list(None, None, scope))?;
        let (slash, colon) = (format!("/{model_id}"), format!(":{model_id}"));
        let name = rows.iter().find_map(|row| {
            let view = self.row_view(row)?;
            let model = view.params.get("model").and_then(Value::as_str)?;
            (model == model_id || model.ends_with(&slash) || model.ends_with(&colon))
                .then_some(view.name)
                .filter(|name| !name.is_empty())
                .map(str::to_owned)
        });
        Ok((!rows.is_empty(), name))
    }

    // -- teams and credentials -------------------------------------------------------------------

    /// `_core_team_model_has_alternatives`: whether the team's public name of deployment `id` has
    /// another unpaused deployment of that team.
    pub fn team_model_has_alternatives(&self, id: &str) -> Result<bool, Row> {
        let Some(held) = self.host.id_entry(id) else {
            return Ok(false);
        };
        let position = self.position(held).ok_or_else(|| Row::Raw(held.clone()))?;
        let info = self.view(position).info;
        let (Some(Value::String(team)), Some(Value::String(public))) =
            (info.get("team_id"), info.get("team_public_model_name"))
        else {
            return Ok(false);
        };
        let Some(siblings) = self.host.team_named(team, public) else {
            return Ok(false);
        };
        let unpaused = self
            .positions(siblings)?
            .into_iter()
            .filter(|position| self.view(*position).info.get("blocked") != Some(&Value::Bool(true)))
            .count();
        Ok(unpaused > 1)
    }

    /// `_get_model_group_deployment_usable_by_team`: the first deployment of the group the team
    /// may use.
    pub fn group_deployment_usable_by_team(
        &self,
        model_group: &str,
        team_id: Option<&str>,
    ) -> Result<Option<usize>, Row> {
        // `.get(name) or ()`: nothing held, or a falsy value, reads no position.
        let Some(held) = self.host.named(model_group).filter(|held| truthy(held)) else {
            return Ok(None);
        };
        let Value::Array(items) = held else {
            return Err(Row::RawPositions(held.clone()));
        };
        for item in items {
            let position = self.position(item).ok_or_else(|| Row::Raw(item.clone()))?;
            if usable_by_team(&self.view(position), team_id) {
                return Ok(Some(position));
            }
        }
        Ok(None)
    }

    /// `_get_team_public_name_deployment`: the first deployment of the team's public name.
    pub fn team_public_name_deployment(
        &self,
        model_id: &str,
        team_id: &str,
    ) -> Result<Option<usize>, Row> {
        match self.host.team_named(team_id, model_id) {
            Some(Value::Array(items)) => match items.first() {
                Some(item) => self
                    .position(item)
                    .map(Some)
                    .ok_or_else(|| Row::Raw(item.clone())),
                None => Ok(None),
            },
            Some(held) if truthy(held) => Err(Row::RawPositions(held.clone())),
            Some(_) | None => Ok(None),
        }
    }

    // -- routable deployments --------------------------------------------------------------------

    /// `_routable_deployments`: the routable group's deployments for the team (those it may use
    /// when any), without paused ones.
    pub fn routable(&self, model_group: &str, team_id: Option<&str>) -> Vec<Row> {
        let named = self.all(model_group, team_id, None);
        if named
            .iter()
            .any(|row| matches!(row, Row::Raw(_) | Row::RawPositions(_)))
        {
            return named;
        }
        let usable: Vec<&Row> = named
            .iter()
            .filter(|row| {
                self.row_view(row)
                    .is_some_and(|view| usable_by_team(&view, team_id))
            })
            .collect();
        let pool: Vec<&Row> = if usable.is_empty() {
            named.iter().collect()
        } else {
            usable
        };
        pool.into_iter()
            .filter(|row| {
                self.row_view(row)
                    .is_some_and(|view| view.info.get("blocked") != Some(&Value::Bool(true)))
            })
            .cloned()
            .collect()
    }

    /// `_core_get_candidate_model_ids_for_route`'s ids of `rows`, in first-seen order.
    pub fn ids_of(&self, rows: &[Row]) -> Result<Vec<String>, Row> {
        let mut ids: Vec<String> = Vec::new();
        for row in rows {
            let view = match row {
                Row::Raw(_) | Row::RawPositions(_) => return Err(row.clone()),
                _ => self.row_view(row).expect("a readable row"),
            };
            if let Some(id) = view.info.get("id").filter(|id| !id.is_null()) {
                let id = python_str(id);
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
        Ok(ids)
    }

    /// `_core_get_candidate_model_ids_for_route` once the alias resolved `resolved`: the
    /// members' ids when it names a callable routing group (`members`), else, for a name that is
    /// no model group, the team's deployments under it or its wildcard routes, else the group's.
    pub fn candidate_ids(
        &self,
        resolved: &str,
        team_id: Option<&str>,
        members: Option<&[String]>,
        requests: &[String],
    ) -> Result<Vec<String>, Row> {
        if let Some(members) = members {
            let mut ids: Vec<String> = Vec::new();
            for member in members {
                for id in self.ids_of(&self.all(member, team_id, None))? {
                    if !ids.contains(&id) {
                        ids.push(id);
                    }
                }
            }
            return Ok(ids);
        }
        let named = Value::String(resolved.to_owned());
        if !self.host.model_names.contains(&named) {
            if team_id.is_some() {
                let team_rows = self.all(resolved, team_id, None);
                if !team_rows.is_empty() {
                    return self.ids_of(&team_rows);
                }
            }
            let wildcard = self.wildcard_route(resolved, team_id, requests);
            if !wildcard.is_empty() {
                return self.ids_of(&wildcard);
            }
        }
        self.ids_of(&self.all(resolved, team_id, None))
    }

    /// `_core_wildcard_route_deployments`: the team router's routes for `model`, else the global
    /// router's.
    pub fn wildcard_route(
        &self,
        model: &str,
        team_id: Option<&str>,
        requests: &[String],
    ) -> Vec<Row> {
        let route = |router: &crate::host_index::PatternRouter| -> Vec<Value> {
            requests
                .iter()
                .find_map(|request| {
                    router
                        .route(request, None)
                        .filter(|routed| !routed.is_empty())
                })
                .unwrap_or_default()
        };
        let team_owned = team_id
            .and_then(|team| self.host.team_patterns.iter().find(|(t, _)| t == team))
            .map(|(_, router)| route(router))
            .unwrap_or_default();
        let routed = if team_owned.is_empty() {
            route(&self.host.patterns)
        } else {
            team_owned
        };
        routed
            .into_iter()
            .map(|entry| Row::Pattern {
                entry,
                name: model.to_owned(),
            })
            .collect()
    }
}

/// `_core_should_include_deployment`: the team's deployment under its public name, or a
/// deployment of the group the team may use.
fn should_include(model_name: &str, view: &View<'_>, team_id: Option<&str>) -> bool {
    let team_value = team_id.map(|team| Value::String(team.to_owned()));
    let model_team = view.info.get("team_id");
    if let Some(team) = &team_value
        && model_team == Some(team)
        && view.info.get("team_public_model_name") == Some(&Value::String(model_name.to_owned()))
    {
        return true;
    }
    if view.name == model_name {
        return team_value.is_none()
            || model_team.is_none_or(Value::is_null)
            || model_team == team_value.as_ref();
    }
    false
}

/// `deployment_usable_by_team`: a deployment no team owns, or one the team owns.
fn usable_by_team(view: &View<'_>, team_id: Option<&str>) -> bool {
    match view.info.get("team_id").filter(|team| !team.is_null()) {
        None => true,
        Some(owner) => team_id.is_some_and(|team| owner == &Value::String(team.to_owned())),
    }
}

fn empty() -> &'static Map<String, Value> {
    static EMPTY: std::sync::OnceLock<Map<String, Value>> = std::sync::OnceLock::new();
    EMPTY.get_or_init(Map::new)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::registry::DeploymentSpec;

    fn registry(entries: Value) -> Registry {
        let mut registry = Registry::default();
        for entry in entries.as_array().unwrap() {
            let spec: DeploymentSpec = serde_json::from_value(entry.clone()).unwrap();
            registry.push_built(spec, false).unwrap();
        }
        registry
    }

    #[test]
    fn a_teams_wildcard_routes_only_the_names_its_public_pattern_matches() {
        let registry = registry(json!([
            {"model_name": "t1-w", "litellm_params": {"model": "openai/*"},
             "model_info": {"id": "tw1", "team_id": "T1", "team_public_model_name": "tw-*"}},
            {"model_name": "g", "litellm_params": {"model": "openai/g"}, "model_info": {"id": "gp"}},
        ]));
        let lookup = Lookup::new(&registry);
        let ids = |name: &str, team: &str| -> Vec<String> {
            let routed = lookup.team_wildcard(name, team);
            routed.iter().map(|d| d.id.clone()).collect()
        };
        assert_eq!(ids("tw-gpt-4o", "T1"), ["tw1"]);
        assert!(ids("gpt-4o", "T1").is_empty());
        assert!(ids("tw-gpt-4o", "T2").is_empty());
    }

    fn entry(name: &str, id: &str, info: Value) -> Value {
        let mut info = info.as_object().cloned().unwrap_or_default();
        info.insert("id".to_owned(), json!(id));
        json!({"model_name": name, "litellm_params": {"model": format!("openai/{id}")}, "model_info": info})
    }

    fn at(position: usize) -> Row {
        Row::At {
            position,
            name: None,
            group: None,
        }
    }

    #[test]
    fn all_prefers_the_teams_public_name_then_the_group() {
        let registry = registry(json!([
            entry("gpt", "a", json!({})),
            entry(
                "team-gpt",
                "t",
                json!({"team_id": "T", "team_public_model_name": "gpt"})
            ),
            entry("gpt", "b", json!({"team_id": "U"})),
        ]));
        let lookup = Lookup::new(&registry);
        assert_eq!(lookup.all("gpt", Some("T"), None), vec![at(1)]);
        assert_eq!(lookup.all("gpt", Some("V"), None), vec![at(0)]);
        assert_eq!(lookup.all("gpt", None, None), vec![at(0), at(2)]);
        assert_eq!(lookup.all("missing", None, None), vec![]);
    }

    #[test]
    fn a_junk_position_is_handed_back_to_raise() {
        let mut registry = registry(json!([entry("gpt", "a", json!({}))]));
        let mut model_name = registry.host_indexes().model_name.clone();
        model_name.insert("gpt".to_owned(), json!([0, "x"]));
        let changes = crate::registry::ChangeSet {
            host_index: Some(crate::host_index::IndexWrite {
                model_name: Some(model_name),
                ..Default::default()
            }),
            ..Default::default()
        };
        registry = registry.apply(&changes, None).unwrap();
        let lookup = Lookup::new(&registry);
        assert_eq!(lookup.all("gpt", None, None), vec![Row::Raw(json!("x"))]);
    }

    #[test]
    fn the_model_list_lists_aliases_and_groups_then_every_deployment() {
        let registry = registry(json!([
            entry("gpt", "a", json!({"access_groups": ["g1"]})),
            entry("claude", "b", json!({})),
        ]));
        let lookup = Lookup::new(&registry);
        let scope = ListScope {
            aliases: vec![AliasItem {
                alias: "fast".to_owned(),
                target: "gpt".to_owned(),
            }],
            groups: vec![CallableGroup {
                name: "pool".to_owned(),
                members: vec!["gpt".to_owned(), "claude".to_owned()],
            }],
            wildcard: Vec::new(),
            only: None,
        };
        let rows = lookup.model_list(None, None, &scope);
        assert_eq!(rows.len(), 5);
        assert_eq!(
            rows[0],
            Row::At {
                position: 0,
                name: Some("fast".to_owned()),
                group: None
            }
        );
        assert_eq!(
            lookup.model_names(None, &scope).unwrap(),
            ["fast", "pool", "pool", "gpt", "claude"]
        );
        assert_eq!(
            lookup.access_groups(None, None, None, &scope).unwrap(),
            vec![("g1".to_owned(), vec!["fast".to_owned(), "gpt".to_owned()])]
        );
    }
}
