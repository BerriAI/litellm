use serde::{Deserialize, Serialize};

use crate::{
    field::{DERIVED_ONLY, FieldSet},
    hook::Hook,
};

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    strum::VariantArray,
)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Required,
    DefaultOn,
    OptIn,
}

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    strum::VariantArray,
)]
#[serde(rename_all = "snake_case")]
pub enum OptOutBy {
    Admin,
    Caller,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Selection {
    pub name: String,
    pub aliases: Vec<String>,
    pub activation: Activation,
    pub opt_out_by: Vec<OptOutBy>,
    pub honors_no_log: bool,
}

impl Selection {
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            aliases: Vec::new(),
            activation: Activation::DefaultOn,
            opt_out_by: vec![OptOutBy::Admin],
            honors_no_log: false,
        }
    }

    fn caller_can_remove(&self) -> bool {
        self.honors_no_log || self.opt_out_by.contains(&OptOutBy::Caller)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Subscription {
    pub hook: Hook,
    pub reads: FieldSet,
    pub writes: FieldSet,
    pub may_abort: bool,
    pub parallel: bool,
    pub awaited: bool,
    pub activation: Option<Activation>,
}

impl Subscription {
    pub const fn on(hook: Hook) -> Self {
        Self {
            hook,
            reads: FieldSet::EMPTY,
            writes: FieldSet::EMPTY,
            may_abort: false,
            parallel: false,
            awaited: false,
            activation: None,
        }
    }

    pub const fn blocking(&self) -> bool {
        !self.writes.is_empty() || self.may_abort
    }

    fn problem(&self) -> Option<SubscriptionProblem> {
        if !self.hook.subscribable() {
            return Some(SubscriptionProblem::NotSubscribable);
        }
        let derived = self.writes.intersection(DERIVED_ONLY);
        if !derived.is_empty() {
            return Some(SubscriptionProblem::DerivedWrite(derived));
        }
        let forbidden = self.writes.difference(self.hook.writable());
        if !forbidden.is_empty() {
            return Some(SubscriptionProblem::Writes(forbidden));
        }
        if self.may_abort && self.hook.outcomes().is_empty() {
            return Some(SubscriptionProblem::NoOutcome);
        }
        if self.awaited && self.blocking() {
            return Some(SubscriptionProblem::AwaitedBlocks);
        }
        if self.parallel && (!self.writes.is_empty() || !self.may_abort) {
            return Some(SubscriptionProblem::ParallelNotAbortOnly);
        }
        None
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub selection: Selection,
    pub subscriptions: Vec<Subscription>,
    pub has_request_state: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SubscriptionProblem {
    #[error("this hook point has no V2 subscription support yet")]
    NotSubscribable,
    #[error("writes {0:?}, which are derived from the hook's arguments and never writable")]
    DerivedWrite(FieldSet),
    #[error("writes {0:?}, which this hook point cannot apply")]
    Writes(FieldSet),
    #[error("may_abort is set, but this hook point allows no outcome")]
    NoOutcome,
    #[error("an awaited subscription must neither write nor abort")]
    AwaitedBlocks,
    #[error("parallel is only for subscriptions that may abort and write nothing")]
    ParallelNotAbortOnly,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ManifestError {
    #[error("{name} on {hook:?}: {problem}")]
    Subscription {
        name: String,
        hook: Hook,
        problem: SubscriptionProblem,
    },
    #[error("{name} declares more than one handler for {hook:?}")]
    Duplicate { name: String, hook: Hook },
    #[error("{0} writes or may abort, so neither no-log nor a caller opt-out may remove it")]
    Removable(String),
    #[error(
        "{name} selects by {selector:?}, which is not lowercase, so no opt-out list can name it"
    )]
    NotLowercase { name: String, selector: String },
}

impl Manifest {
    pub fn subscription(&self, hook: Hook) -> Option<&Subscription> {
        self.subscriptions
            .iter()
            .find(|subscription| subscription.hook == hook)
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        let name = &self.selection.name;
        if let Some(selector) = std::iter::once(name)
            .chain(&self.selection.aliases)
            .find(|selector| selector.to_lowercase() != **selector)
        {
            return Err(ManifestError::NotLowercase {
                name: name.clone(),
                selector: selector.clone(),
            });
        }
        for (index, subscription) in self.subscriptions.iter().enumerate() {
            if let Some(problem) = subscription.problem() {
                return Err(ManifestError::Subscription {
                    name: name.clone(),
                    hook: subscription.hook,
                    problem,
                });
            }
            if self.subscriptions[..index]
                .iter()
                .any(|prior| prior.hook == subscription.hook)
            {
                return Err(ManifestError::Duplicate {
                    name: name.clone(),
                    hook: subscription.hook,
                });
            }
        }
        if self.subscriptions.iter().any(Subscription::blocking)
            && self.selection.caller_can_remove()
        {
            return Err(ManifestError::Removable(name.clone()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::field::Field;

    fn manifest(subscription: Subscription, selection: Selection) -> Manifest {
        Manifest {
            selection,
            subscriptions: vec![subscription],
            has_request_state: false,
        }
    }

    fn named() -> Selection {
        Selection::named("example")
    }

    #[rstest]
    #[case::abort_on_pre_call(Subscription { may_abort: true, ..Subscription::on(Hook::PreCall) })]
    #[case::read_off_path(Subscription { reads: FieldSet::of(&[Field::Usage]), ..Subscription::on(Hook::AttemptEnded) })]
    #[case::awaited_reader(Subscription { awaited: true, ..Subscription::on(Hook::PostCallSuccess) })]
    #[case::reservation_in_place(Subscription {
        reads: FieldSet::of(&[Field::BudgetPlan, Field::BudgetReservation]),
        writes: FieldSet::of(&[Field::BudgetReservation]),
        may_abort: true,
        ..Subscription::on(Hook::PreCall)
    })]
    #[case::merge_headers(Subscription {
        writes: FieldSet::of(&[Field::ResponseHeaders]),
        ..Subscription::on(Hook::ResponseHeaders)
    })]
    fn a_declaration_the_hook_point_allows_validates(#[case] subscription: Subscription) {
        assert_eq!(manifest(subscription, named()).validate(), Ok(()));
    }

    #[rstest]
    #[case::stream(
        Subscription::on(Hook::StreamChunk),
        SubscriptionProblem::NotSubscribable
    )]
    #[case::write_elsewhere(
        Subscription { writes: FieldSet::of(&[Field::Response]), ..Subscription::on(Hook::PreCall) },
        SubscriptionProblem::Writes(FieldSet::of(&[Field::Response]))
    )]
    #[case::derived_write(
        Subscription { writes: FieldSet::of(&[Field::KeyIdentity, Field::Model]), ..Subscription::on(Hook::PreCall) },
        SubscriptionProblem::DerivedWrite(FieldSet::of(&[Field::KeyIdentity]))
    )]
    #[case::derived_write_where_nothing_is_writable(
        Subscription { writes: FieldSet::of(&[Field::Usage]), ..Subscription::on(Hook::AttemptEnded) },
        SubscriptionProblem::DerivedWrite(FieldSet::of(&[Field::Usage]))
    )]
    #[case::abort_off_path(
        Subscription { may_abort: true, ..Subscription::on(Hook::AttemptEnded) },
        SubscriptionProblem::NoOutcome
    )]
    #[case::awaited_abort(
        Subscription { may_abort: true, awaited: true, ..Subscription::on(Hook::PreCall) },
        SubscriptionProblem::AwaitedBlocks
    )]
    #[case::parallel_writer(
        Subscription { parallel: true, writes: FieldSet::of(&[Field::Model]), may_abort: true, ..Subscription::on(Hook::PreCall) },
        SubscriptionProblem::ParallelNotAbortOnly
    )]
    fn a_declaration_the_hook_point_rejects_names_the_problem(
        #[case] subscription: Subscription,
        #[case] problem: SubscriptionProblem,
    ) {
        let hook = subscription.hook;
        assert_eq!(
            manifest(subscription, named()).validate(),
            Err(ManifestError::Subscription {
                name: "example".into(),
                hook,
                problem
            })
        );
    }

    #[rstest]
    #[case::no_log(Selection { honors_no_log: true, ..named() })]
    #[case::caller(Selection { opt_out_by: vec![OptOutBy::Admin, OptOutBy::Caller], ..named() })]
    fn a_blocking_callback_a_caller_could_remove_is_rejected(#[case] selection: Selection) {
        let subscription = Subscription {
            may_abort: true,
            ..Subscription::on(Hook::PreCall)
        };
        assert_eq!(
            manifest(subscription, selection).validate(),
            Err(ManifestError::Removable("example".into()))
        );
    }

    #[rstest]
    #[case::name(Selection::named("S3_v2"), "S3_v2")]
    #[case::alias(Selection { aliases: vec!["s3".into(), "S3".into()], ..named() }, "S3")]
    fn a_selector_that_is_not_lowercase_is_rejected(
        #[case] selection: Selection,
        #[case] selector: &str,
    ) {
        let name = selection.name.clone();
        assert_eq!(
            manifest(Subscription::on(Hook::AttemptEnded), selection).validate(),
            Err(ManifestError::NotLowercase {
                name,
                selector: selector.into()
            })
        );
    }

    #[rstest]
    fn two_handlers_for_one_hook_are_rejected() {
        let mut twice = manifest(Subscription::on(Hook::AttemptEnded), named());
        twice
            .subscriptions
            .push(Subscription::on(Hook::AttemptEnded));
        assert_eq!(
            twice.validate(),
            Err(ManifestError::Duplicate {
                name: "example".into(),
                hook: Hook::AttemptEnded
            })
        );
    }
}
