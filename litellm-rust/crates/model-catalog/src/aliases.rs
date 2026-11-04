use std::hash::Hash;

use indexmap::IndexMap;

/// Why an alias was left unclaimed
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unclaimed<O> {
    /// The alias is already a top-level key of the map
    ShadowsModel,
    TakenBy(O),
}

/// Aliases in claim order, each served as the row of the first model that claimed it
#[derive(Clone, Debug)]
pub struct AliasClaims<A, O> {
    owners: IndexMap<A, O>,
}

impl<A: Clone + Hash + Eq, O: Clone + Hash + Eq> AliasClaims<A, O> {
    pub fn new() -> Self {
        Self {
            owners: IndexMap::new(),
        }
    }

    pub fn claim(&mut self, alias: &A, owner: &O, is_model: bool) -> Result<(), Unclaimed<O>> {
        if is_model {
            return Err(Unclaimed::ShadowsModel);
        }
        if let Some(taken_by) = self.owners.get(alias) {
            return Err(Unclaimed::TakenBy(taken_by.clone()));
        }
        self.owners.insert(alias.clone(), owner.clone());
        Ok(())
    }

    /// Each owner with its aliases, owners in the order of their first claim
    pub fn groups(&self) -> IndexMap<&O, Vec<&A>> {
        self.owners
            .iter()
            .fold(IndexMap::new(), |mut groups, (alias, owner)| {
                groups.entry(owner).or_insert_with(Vec::new).push(alias);
                groups
            })
    }

    pub fn into_owners(self) -> IndexMap<A, O> {
        self.owners
    }
}

impl<A: Clone + Hash + Eq, O: Clone + Hash + Eq> Default for AliasClaims<A, O> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    type Claims = AliasClaims<&'static str, &'static str>;

    #[rstest]
    #[case::free("a", "m", false, Ok(()))]
    #[case::shadows_a_model("a", "m", true, Err(Unclaimed::ShadowsModel))]
    #[case::taken("taken", "m", false, Err(Unclaimed::TakenBy("first")))]
    #[case::shadow_wins_over_taken("taken", "m", true, Err(Unclaimed::ShadowsModel))]
    fn a_claim_is_refused_by_a_model_key_first_then_by_an_earlier_claim(
        #[case] alias: &'static str,
        #[case] owner: &'static str,
        #[case] is_model: bool,
        #[case] expected: Result<(), Unclaimed<&'static str>>,
    ) {
        let mut claims = Claims::new();
        claims.claim(&"taken", &"first", false).unwrap();
        assert_eq!(claims.claim(&alias, &owner, is_model), expected);
    }

    #[rstest]
    fn groups_keep_owners_in_first_claim_order_and_aliases_in_claim_order() {
        let mut claims = Claims::new();
        for (alias, owner) in [("b1", "b"), ("a1", "a"), ("b2", "b"), ("a1", "c")] {
            let _ = claims.claim(&alias, &owner, false);
        }
        let groups: Vec<(&str, Vec<&str>)> = claims
            .groups()
            .into_iter()
            .map(|(owner, aliases)| (*owner, aliases.into_iter().copied().collect()))
            .collect();
        assert_eq!(groups, vec![("b", vec!["b1", "b2"]), ("a", vec!["a1"])]);
    }
}
