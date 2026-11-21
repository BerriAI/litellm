//! Which models a caller may call: one rule over a list of model names, applied to every layer
//! the host builds for the caller.

const EVERY_MODEL: &str = "*";

/// One level of a caller that lists the models it may call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer<'a> {
    Key { models: &'a [String] },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
pub enum LayerKind {
    #[strum(serialize = "key")]
    Key,
}

/// A model a request asks for: the name it sent, the model group that name resolves to and the
/// access groups that group belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelAsk<'a> {
    pub name: &'a str,
    pub group: &'a str,
    pub access_groups: &'a [String],
}

/// The first layer whose list does not reach a model the request asks for.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{} not allowed to access model. Tried to access {model}", <&str>::from(layer))]
pub struct Denied {
    pub layer: LayerKind,
    pub model: String,
}

impl Layer<'_> {
    fn kind(&self) -> LayerKind {
        match self {
            Self::Key { .. } => LayerKind::Key,
        }
    }

    fn models(&self) -> &[String] {
        match self {
            Self::Key { models } => models,
        }
    }
}

/// Every layer must reach every ask: the model and each fallback.
pub fn authorize<'a>(
    mut layers: impl Iterator<Item = Layer<'a>>,
    asks: &[ModelAsk<'_>],
) -> Result<(), Denied> {
    let denied = layers.find_map(|layer| {
        let missed = asks.iter().find(|ask| !reaches(layer.models(), ask))?;
        Some(Denied {
            layer: layer.kind(),
            model: missed.name.to_owned(),
        })
    });
    denied.map_or(Ok(()), Err)
}

/// A list reaches a model when it is empty, holds `*`, or names the model, its group or one of
/// the group's access groups.
fn reaches(models: &[String], ask: &ModelAsk<'_>) -> bool {
    models.is_empty()
        || models.iter().any(|listed| {
            listed == EVERY_MODEL
                || listed == ask.name
                || listed == ask.group
                || ask.access_groups.contains(listed)
        })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn list(models: &[&str]) -> Vec<String> {
        models.iter().map(|model| (*model).to_owned()).collect()
    }

    fn key(models: &[String]) -> Layer<'_> {
        Layer::Key { models }
    }

    fn sonnet(access_groups: &[String]) -> ModelAsk<'_> {
        ModelAsk {
            name: "sonnet",
            group: "claude",
            access_groups,
        }
    }

    fn denied(model: &str) -> Result<(), Denied> {
        Err(Denied {
            layer: LayerKind::Key,
            model: model.to_owned(),
        })
    }

    #[rstest]
    #[case::an_empty_list(&[], Ok(()))]
    #[case::every_model(&["*"], Ok(()))]
    #[case::the_model(&["gpt-4o", "sonnet"], Ok(()))]
    #[case::its_group(&["gpt-4o", "claude"], Ok(()))]
    #[case::one_of_its_access_groups(&["gpt-4o", "beta"], Ok(()))]
    #[case::none_of_them(&["gpt-4o"], denied("sonnet"))]
    fn a_list_reaches_a_model_it_names(
        #[case] models: &[&str],
        #[case] expected: Result<(), Denied>,
    ) {
        let models = list(models);
        let access_groups = list(&["alpha", "beta"]);

        assert_eq!(
            authorize([key(&models)].into_iter(), &[sonnet(&access_groups)]),
            expected
        );
    }

    #[rstest]
    fn a_fallback_the_list_misses_is_denied() {
        let models = list(&["sonnet"]);
        let fallback = ModelAsk {
            name: "haiku",
            group: "claude-small",
            access_groups: &[],
        };

        assert_eq!(
            authorize([key(&models)].into_iter(), &[sonnet(&[]), fallback]),
            denied("haiku")
        );
    }

    #[rstest]
    fn a_later_layer_that_misses_is_denied() {
        let narrow = list(&["gpt-4o"]);

        assert_eq!(
            authorize([key(&[]), key(&narrow)].into_iter(), &[sonnet(&[])]),
            denied("sonnet")
        );
    }

    #[rstest]
    fn no_layer_denies_nothing() {
        assert_eq!(authorize(std::iter::empty(), &[sonnet(&[])]), Ok(()));
    }
}
