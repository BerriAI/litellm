//! The names a requested model may be granted under once aliases are applied, in the order the
//! proxy's `_can_object_call_model` applies them: the team's, the key's, `litellm.model_alias_map`,
//! then the router's own.

use std::collections::BTreeMap;

/// The alias tables a request's model goes through. The host reads each from where it lives.
#[derive(Clone, Copy, Debug)]
pub struct Aliases<'a> {
    /// The team's `model_aliases`.
    pub team: &'a BTreeMap<String, String>,
    /// The key's `aliases`.
    pub key: &'a BTreeMap<String, String>,
    /// `litellm.model_alias_map`.
    pub global: &'a BTreeMap<String, String>,
    /// The model group the router's `model_group_alias` rewrites the requested name to.
    pub router: Option<&'a str>,
}

/// The names to judge a request's model by. A host adds the access groups of their deployments
/// to make the [`ModelAsk`](crate::authz::ModelAsk).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidates<'a> {
    pub names: Vec<&'a str>,
    /// One of `names` is a name the team aliases.
    pub team_alias: bool,
}

fn through<'a>(table: &'a BTreeMap<String, String>, name: &'a str) -> &'a str {
    table.get(name).map_or(name, String::as_str)
}

/// A key's alias replaces the name, so the model it dispatches to is the only candidate. With
/// none, the candidates are the name as asked and the model a global or router alias names for it.
pub fn candidates<'a>(name: &'a str, aliases: &Aliases<'a>) -> Candidates<'a> {
    let after_team = through(aliases.team, name);
    let after_key = through(aliases.key, after_team);
    let after_global = through(aliases.global, after_key);
    let dispatched = through(aliases.key, after_global);
    let names = if after_key != after_team || dispatched != after_global {
        vec![dispatched]
    } else {
        let target = aliases.global.get(name).map(String::as_str);
        let target = target
            .or(aliases.router)
            .filter(|target| !target.is_empty());
        std::iter::once(name).chain(target).collect()
    };
    Candidates {
        team_alias: names.iter().any(|name| aliases.team.contains_key(*name)),
        names,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn table(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
        entries
            .iter()
            .map(|(name, target)| ((*name).to_owned(), (*target).to_owned()))
            .collect()
    }

    #[rstest]
    #[case::no_alias(&[], &[], &[], None, &["gpt"], false)]
    #[case::the_routers(&[], &[], &[], Some("gpt-group"), &["gpt", "gpt-group"], false)]
    #[case::a_global_one_before_the_routers(
        &[], &[], &[("gpt", "gpt-global")], Some("gpt-group"), &["gpt", "gpt-global"], false
    )]
    #[case::a_global_one_that_names_nothing(
        &[], &[], &[("gpt", "")], Some("gpt-group"), &["gpt"], false
    )]
    #[case::the_teams_is_a_grant_not_a_candidate(
        &[("gpt", "gpt-team")], &[], &[], None, &["gpt"], true
    )]
    #[case::the_keys_replaces_the_name(&[], &[("gpt", "gpt-key")], &[], Some("gpt-group"), &["gpt-key"], false)]
    #[case::the_keys_after_the_teams(
        &[("gpt", "gpt-team")], &[("gpt-team", "gpt-key")], &[], None, &["gpt-key"], false
    )]
    #[case::the_keys_after_a_global_one(
        &[], &[("gpt-global", "gpt-key")], &[("gpt", "gpt-global")], None, &["gpt-key"], false
    )]
    #[case::the_keys_then_a_global_one(
        &[], &[("gpt", "gpt-key")], &[("gpt-key", "gpt-global")], None, &["gpt-global"], false
    )]
    #[case::the_keys_onto_a_name_the_team_aliases(
        &[("gpt-key", "gpt-team")], &[("gpt", "gpt-key")], &[], None, &["gpt-key"], true
    )]
    #[case::the_keys_onto_itself(&[], &[("gpt", "gpt")], &[], Some("gpt-group"), &["gpt", "gpt-group"], false)]
    fn a_model_is_judged_by_the_names_its_aliases_give_it(
        #[case] team: &[(&str, &str)],
        #[case] key: &[(&str, &str)],
        #[case] global: &[(&str, &str)],
        #[case] router: Option<&str>,
        #[case] names: &[&str],
        #[case] team_alias: bool,
    ) {
        let (team, key, global) = (table(team), table(key), table(global));
        let aliases = Aliases {
            team: &team,
            key: &key,
            global: &global,
            router,
        };

        assert_eq!(
            candidates("gpt", &aliases),
            Candidates {
                names: names.to_vec(),
                team_alias
            }
        );
    }
}
