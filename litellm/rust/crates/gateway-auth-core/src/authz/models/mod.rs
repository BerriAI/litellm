//! Whether the model lists a host read for a caller reach the model it asks for. The host
//! supplies each list it read as a [`Layer`] and the [`ModelAsk`] it resolved the name to, and
//! decides nothing itself.

mod aliases;
mod list;

pub use aliases::candidates;
pub use list::ModelList;

use super::ModelAsk;
use crate::{AuthzDenial, DeniedList};

/// One model list the host read for a caller, named by the row that holds it. A request has to
/// be reached by every layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer<'a> {
    Key(ModelList<'a>),
}

impl<'a> Layer<'a> {
    pub fn list(&self) -> ModelList<'a> {
        match *self {
            Self::Key(list) => list,
        }
    }

    fn denial(&self, ask: &ModelAsk<'_>) -> AuthzDenial {
        let list = match self {
            Self::Key(_) => DeniedList::Key,
        };
        AuthzDenial::ModelAccess {
            list,
            model: ask.name.to_owned(),
        }
    }
}

/// Whether every one of `layers` reaches `ask`, with the denial of the first that does not. No
/// layer at all, as the master key has, reaches every model.
pub fn authorize<'a>(
    layers: impl IntoIterator<Item = Layer<'a>>,
    ask: &ModelAsk<'_>,
) -> Result<(), AuthzDenial> {
    layers.into_iter().try_for_each(|layer| {
        if layer.list().reaches(ask) {
            Ok(())
        } else {
            Err(layer.denial(ask))
        }
    })
}

#[cfg(test)]
pub(crate) fn owned(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::the_name_asked_for(&["claude"], &["claude"], &[], true)]
    #[case::another_name(&["claude"], &["gpt"], &[], false)]
    #[case::a_name_that_only_starts_the_same(&["claude"], &["claude-down"], &[], false)]
    #[case::the_group_an_alias_names(&["claude"], &["sonnet", "claude"], &[], true)]
    #[case::only_the_alias(&["sonnet"], &["claude"], &[], false)]
    #[case::an_access_group_of_the_model(&["fast"], &["claude"], &["haiku", "fast"], true)]
    #[case::an_access_group_of_another_model(&["haiku"], &["gpt"], &["fast"], false)]
    #[case::every_model(&["*"], &["gpt"], &[], true)]
    #[case::an_empty_list(&[], &["gpt"], &[], true)]
    #[case::an_unread_list(&["gpt", "all-team-models"], &["gpt"], &[], false)]
    fn a_key_may_call_only_what_the_proxy_lets_it(
        #[case] models: &[&str],
        #[case] candidates: &[&str],
        #[case] access_groups: &[&str],
        #[case] allowed: bool,
    ) {
        let (models, access_groups) = (owned(models), owned(access_groups));
        let ask = ModelAsk {
            name: candidates[0],
            candidates,
            access_groups: &access_groups,
        };

        let authorized = authorize([Layer::Key(ModelList::read(&models))], &ask);

        assert_eq!(
            authorized,
            if allowed {
                Ok(())
            } else {
                Err(AuthzDenial::ModelAccess {
                    list: DeniedList::Key,
                    model: candidates[0].to_owned(),
                })
            }
        );
    }

    #[rstest]
    fn a_refused_model_reads_as_every_model_denial_does() {
        let ask = ModelAsk {
            name: "gpt",
            candidates: &["gpt"],
            access_groups: &[],
        };

        let refused =
            authorize([Layer::Key(ModelList::read(&owned(&["claude"])))], &ask).unwrap_err();

        assert_eq!(
            refused.to_string(),
            crate::ModelAccessDenied { model: "gpt" }.to_string()
        );
    }

    #[rstest]
    fn a_caller_with_no_layer_reaches_every_model() {
        let ask = ModelAsk {
            name: "gpt",
            candidates: &["gpt"],
            access_groups: &[],
        };

        assert_eq!(authorize([], &ask), Ok(()));
    }
}
