use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Facts {
    pub name: String,
    pub router_present: bool,
    pub alias_count: usize,
    pub global_alias_exists: Option<bool>,
    pub alias_entries: Vec<Option<bool>>,
    pub registry_loaded: bool,
    pub registry_alias_exists: Option<bool>,
    pub deployment: Option<bool>,
    pub names: Option<Vec<String>>,
}

#[derive(Debug, PartialEq, Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    Complete { present: bool },
    ReadGlobal,
    ReadAlias { index: usize },
    ReadRegistry,
    ReadAliases,
    ReadDeployment,
    ReadNames,
}

pub fn decide(facts: &Facts) -> Decision {
    let Some(global) = &facts.global_alias_exists else {
        return Decision::ReadGlobal;
    };
    if *global {
        return Decision::Complete { present: true };
    }
    for index in 0..facts.alias_count {
        let Some(aliases) = facts.alias_entries.get(index) else {
            return Decision::ReadAlias { index };
        };
        if *aliases == Some(true) {
            return Decision::Complete { present: true };
        }
    }
    if !facts.router_present {
        return Decision::Complete { present: false };
    }
    if !facts.registry_loaded {
        return Decision::ReadRegistry;
    }
    let Some(aliases) = &facts.registry_alias_exists else {
        return Decision::ReadAliases;
    };
    if *aliases {
        return Decision::Complete { present: true };
    }
    match facts.deployment {
        None => return Decision::ReadDeployment,
        Some(true) => return Decision::Complete { present: true },
        Some(false) => (),
    }
    let Some(names) = &facts.names else {
        return Decision::ReadNames;
    };
    Decision::Complete {
        present: !names.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use super::{Decision, Facts, decide};
    use rstest::rstest;

    #[rstest]
    #[case::global(Some(true), Decision::Complete { present: true })]
    #[case::missing(None, Decision::ReadGlobal)]
    #[case::next_alias(Some(false), Decision::ReadAlias { index: 0 })]
    fn alias_short_circuit(#[case] global_alias_exists: Option<bool>, #[case] expected: Decision) {
        let facts = Facts {
            name: "model".into(),
            router_present: true,
            alias_count: 3,
            global_alias_exists,
            ..Facts::default()
        };
        assert_eq!(decide(&facts), expected);
    }

    #[rstest]
    #[case::no_router(false, Decision::Complete { present: false })]
    #[case::registry(true, Decision::ReadRegistry)]
    fn registry_after_all_aliases(#[case] router_present: bool, #[case] expected: Decision) {
        let facts = Facts {
            name: "model".into(),
            router_present,
            global_alias_exists: Some(false),
            ..Facts::default()
        };
        assert_eq!(decide(&facts), expected);
    }
}
