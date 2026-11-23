//! Whether the model lists a host read for a caller reach the model it asks for, as the proxy's
//! `can_key_call_model` reads a list. The host supplies each list it read as a [`Layer`] and the
//! [`ModelAsk`] its router resolved the name to, and decides nothing itself.

/// One model list the host read for a caller. A request has to be reached by every layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer<'a> {
    /// The key's `models`.
    Key(&'a [String]),
}

/// The model a caller asks for: the `name` as the request spells it, the model `group` the
/// router resolves it to, and the access groups of that group's deployments, each of which a
/// list may name in the model's place.
#[derive(Clone, Copy, Debug)]
pub struct ModelAsk<'a> {
    pub name: &'a str,
    pub group: &'a str,
    pub access_groups: &'a [String],
}

/// The layer whose list does not reach the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Denied {
    #[error("the key's model list does not reach the model")]
    Key,
}

/// The entries that reach every model.
const EVERY_MODEL: [&str; 2] = ["*", "all-proxy-models"];

/// An empty list reaches every model. Any other reaches the model it names, by the name asked
/// for, its group or one of its access groups, and every model once it holds an entry that does.
fn reaches(models: &[String], ask: &ModelAsk<'_>) -> bool {
    models.is_empty()
        || models.iter().any(|model| {
            EVERY_MODEL.contains(&model.as_str())
                || model == ask.name
                || model == ask.group
                || ask.access_groups.contains(model)
        })
}

/// Whether every one of `layers` reaches `ask`, naming the first that does not. No layer at all,
/// as the master key has, reaches every model.
pub fn authorize<'a>(
    layers: impl IntoIterator<Item = Layer<'a>>,
    ask: &ModelAsk<'_>,
) -> Result<(), Denied> {
    layers.into_iter().try_for_each(|layer| match layer {
        Layer::Key(models) if reaches(models, ask) => Ok(()),
        Layer::Key(_) => Err(Denied::Key),
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn owned(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_owned()).collect()
    }

    #[rstest]
    #[case::the_name_asked_for(&["claude"], "claude", "claude", &[], true)]
    #[case::another_name(&["claude"], "gpt", "gpt", &[], false)]
    #[case::the_group_an_alias_names(&["claude"], "sonnet", "claude", &[], true)]
    #[case::only_the_alias(&["sonnet"], "claude", "claude", &[], false)]
    #[case::an_access_group_of_the_group(&["haiku"], "claude", "claude", &["haiku", "fast"], true)]
    #[case::an_access_group_of_another_group(&["haiku"], "gpt", "gpt", &["fast"], false)]
    #[case::every_model(&["*"], "gpt", "gpt", &[], true)]
    #[case::every_proxy_model(&["all-proxy-models"], "gpt", "gpt", &[], true)]
    #[case::a_name_that_only_starts_the_same(&["claude"], "claude-down", "claude-down", &[], false)]
    #[case::an_empty_list(&[], "gpt", "gpt", &[], true)]
    fn a_key_with_a_model_list_may_call_only_what_the_proxy_lets_it(
        #[case] models: &[&str],
        #[case] name: &str,
        #[case] group: &str,
        #[case] access_groups: &[&str],
        #[case] allowed: bool,
    ) {
        let (models, access_groups) = (owned(models), owned(access_groups));
        let ask = ModelAsk {
            name,
            group,
            access_groups: &access_groups,
        };

        let authorized = authorize([Layer::Key(&models)], &ask);

        assert_eq!(authorized, if allowed { Ok(()) } else { Err(Denied::Key) });
    }

    #[rstest]
    fn a_caller_with_no_layer_reaches_every_model() {
        let ask = ModelAsk {
            name: "gpt",
            group: "gpt",
            access_groups: &[],
        };

        assert_eq!(authorize([], &ask), Ok(()));
    }
}
