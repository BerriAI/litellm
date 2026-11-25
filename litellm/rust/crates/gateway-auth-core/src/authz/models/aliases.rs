//! The names a requested model may be granted under once aliases are applied, in the order the
//! proxy's `_can_object_call_model` applies them: the team's, the key's, `litellm.model_alias_map`,
//! then the router's own. Only the router's reaches a caller today, since a host declines a key
//! with a team or aliases of its own, and no host reads `litellm.model_alias_map`.

/// The name as asked, then the model group the router's `model_group_alias` names for it.
pub fn candidates<'a>(name: &'a str, router: Option<&'a str>) -> Vec<&'a str> {
    std::iter::once(name)
        .chain(router.filter(|target| !target.is_empty()))
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::no_alias(None, &["gpt"])]
    #[case::the_routers(Some("gpt-group"), &["gpt", "gpt-group"])]
    #[case::one_that_names_nothing(Some(""), &["gpt"])]
    fn a_model_is_judged_by_the_names_its_aliases_give_it(
        #[case] router: Option<&str>,
        #[case] names: &[&str],
    ) {
        assert_eq!(candidates("gpt", router), names);
    }
}
