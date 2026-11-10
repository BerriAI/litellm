//! The proxy policy engine of `litellm/proxy/policy_engine`: which policies apply to a request,
//! and the guardrails and pipelines they resolve to. Pure logic over an immutable snapshot of the
//! policy and attachment registries, so the Python proxy and the gateway can share it.

mod attachment;
mod condition;

use std::collections::{HashMap, HashSet};

pub use attachment::Attachment;
pub use condition::{HostRegex, ModelCondition, PatternSpec};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MatchContext {
    pub team_alias: Option<String>,
    pub key_alias: Option<String>,
    pub model: Option<String>,
    pub tags: Vec<String>,
}

#[derive(Debug, Default)]
pub struct Policy {
    pub inherit: Option<String>,
    pub add: Vec<String>,
    pub remove: Vec<String>,
    pub condition: Option<ModelCondition>,
    pub pipeline_guardrails: Option<Vec<String>>,
}

impl Policy {
    fn applies(&self, context: &MatchContext) -> bool {
        self.condition
            .as_ref()
            .is_none_or(|condition| condition.matches(context.model.as_deref()))
    }
}

#[derive(Debug)]
pub struct PolicyVersion {
    pub id: String,
    pub name: String,
    pub policy: Policy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineSource {
    pub policy: String,
    pub version_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AncestorAdmission {
    pub policy: String,
    pub ancestor: String,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Resolution {
    pub attached_policies: Vec<String>,
    pub unapplied_policies: Vec<String>,
    pub condition_misses: Vec<String>,
    pub policy_guardrails: Vec<(String, Vec<String>)>,
    pub applied_policies: Vec<String>,
    pub policy_sources: Vec<(String, String)>,
    pub guardrails: Vec<String>,
    pub pipelines: Vec<PipelineSource>,
    pub pipeline_managed_guardrails: Vec<String>,
    pub ancestor_admissions: Vec<AncestorAdmission>,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ResolvedPolicy {
    pub guardrails: Vec<String>,
    pub inheritance_chain: Vec<String>,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct AttachmentMatches {
    pub matches: Vec<(String, String)>,
    pub ancestor_admissions: Vec<AncestorAdmission>,
}

#[derive(Debug, Default)]
pub struct PolicyEngine {
    policies: HashMap<String, Policy>,
    versions: HashMap<String, (String, Policy)>,
    attachments: Vec<Attachment>,
}

struct Policies<'a> {
    engine: &'a PolicyEngine,
    overrides: HashMap<&'a str, (&'a str, &'a Policy)>,
}

impl<'a> Policies<'a> {
    fn get(&self, name: &str) -> Option<&'a Policy> {
        self.overrides
            .get(name)
            .map(|(_, policy)| *policy)
            .or_else(|| self.engine.policies.get(name))
    }

    fn version_id(&self, name: &str) -> Option<&'a str> {
        self.overrides.get(name).map(|(id, _)| *id)
    }

    fn inheritance_chain(&self, name: &'a str) -> Vec<&'a str> {
        let mut seen = HashSet::new();
        let mut chain = Vec::new();
        let mut current = Some(name);
        while let Some(name) = current {
            if !seen.insert(name) {
                break;
            }
            let Some(policy) = self.get(name) else {
                break;
            };
            chain.push(name);
            current = policy.inherit.as_deref();
        }
        chain.reverse();
        chain
    }

    fn applying_chain_members(&self, name: &'a str, context: &MatchContext) -> Vec<&'a str> {
        self.inheritance_chain(name)
            .into_iter()
            .filter(|member| {
                self.get(member)
                    .is_some_and(|policy| policy.applies(context))
            })
            .collect()
    }

    fn guardrails(&self, name: &'a str, context: Option<&MatchContext>) -> Vec<&'a str> {
        let mut guardrails = Vec::new();
        for policy in self
            .inheritance_chain(name)
            .into_iter()
            .filter_map(|member| self.get(member))
            .filter(|policy| context.is_none_or(|context| policy.applies(context)))
        {
            for guardrail in &policy.add {
                push_unique(&mut guardrails, guardrail.as_str());
            }
            guardrails.retain(|kept| !policy.remove.iter().any(|removed| removed == kept));
        }
        guardrails
    }
}

fn push_unique<'a>(values: &mut Vec<&'a str>, value: &'a str) {
    if !values.contains(&value) {
        values.push(value);
    }
}

fn owned(values: Vec<&str>) -> Vec<String> {
    values.into_iter().map(str::to_owned).collect()
}

impl PolicyEngine {
    pub fn new(
        policies: impl IntoIterator<Item = (String, Policy)>,
        versions: impl IntoIterator<Item = PolicyVersion>,
        attachments: Vec<Attachment>,
    ) -> Self {
        Self {
            policies: policies.into_iter().collect(),
            versions: versions
                .into_iter()
                .map(|version| (version.id, (version.name, version.policy)))
                .collect(),
            attachments,
        }
    }

    fn attached<'a>(
        &'a self,
        context: &MatchContext,
        policies: Option<&Policies<'a>>,
        admissions: &mut Vec<AncestorAdmission>,
    ) -> Vec<&'a Attachment> {
        let mut in_scope = Vec::new();
        for attachment in &self.attachments {
            if !attachment.in_scope(context) {
                continue;
            }
            let Some(policies) = policies else {
                in_scope.push(attachment);
                continue;
            };
            let applying = policies.applying_chain_members(&attachment.policy, context);
            let Some(first) = applying.first() else {
                continue;
            };
            if !applying.contains(&attachment.policy.as_str()) {
                admissions.push(AncestorAdmission {
                    policy: attachment.policy.clone(),
                    ancestor: (*first).to_owned(),
                });
            }
            in_scope.push(attachment);
        }
        let any_non_default = in_scope.iter().any(|attachment| !attachment.default);
        in_scope.retain(|attachment| attachment.default != any_non_default);
        in_scope.sort_by_key(|attachment| attachment.order());
        let mut seen = HashSet::new();
        in_scope.retain(|attachment| seen.insert(attachment.policy.as_str()));
        in_scope
    }

    fn base(&self) -> Policies<'_> {
        Policies {
            engine: self,
            overrides: HashMap::new(),
        }
    }

    /// Registry policies whose inheritance chain loops back on itself, sorted by name
    pub fn circular_policies(&self) -> Vec<String> {
        let mut circular: Vec<String> = self
            .policies
            .keys()
            .filter(|name| self.inherits_in_a_loop(name))
            .cloned()
            .collect();
        circular.sort_unstable();
        circular
    }

    fn inherits_in_a_loop(&self, name: &str) -> bool {
        let mut seen = HashSet::new();
        let mut current = Some(name);
        while let Some(name) = current {
            if !seen.insert(name) {
                return true;
            }
            current = self
                .policies
                .get(name)
                .and_then(|policy| policy.inherit.as_deref());
        }
        false
    }

    pub fn policy_guardrails(&self, name: &str, context: Option<&MatchContext>) -> ResolvedPolicy {
        let base = self.base();
        ResolvedPolicy {
            guardrails: owned(base.guardrails(name, context)),
            inheritance_chain: owned(base.inheritance_chain(name)),
        }
    }

    pub fn attached_policies(
        &self,
        context: &MatchContext,
        only_applying: bool,
    ) -> AttachmentMatches {
        let base = self.base();
        let mut ancestor_admissions = Vec::new();
        let matches = self
            .attached(
                context,
                only_applying.then_some(&base),
                &mut ancestor_admissions,
            )
            .into_iter()
            .map(|attachment| (attachment.policy.clone(), attachment.match_reason(context)))
            .collect();
        AttachmentMatches {
            matches,
            ancestor_admissions,
        }
    }

    pub fn resolve(
        &self,
        context: &MatchContext,
        names: &[String],
        version_ids: &[String],
    ) -> Resolution {
        let mut overrides = HashMap::new();
        let mut fetched = Vec::new();
        for id in version_ids {
            if let Some((name, policy)) = self.versions.get(id) {
                overrides.insert(name.as_str(), (id.as_str(), policy));
                fetched.push(name.as_str());
            }
        }
        let merged = Policies {
            engine: self,
            overrides,
        };
        let mut ancestor_admissions = Vec::new();
        let attached = self.attached(context, Some(&merged), &mut ancestor_admissions);

        let mut candidates = Vec::new();
        for name in attached
            .iter()
            .map(|attachment| attachment.policy.as_str())
            .chain(names.iter().map(String::as_str))
            .chain(fetched)
        {
            push_unique(&mut candidates, name);
        }
        let (applied, unapplied): (Vec<&str>, Vec<&str>) = candidates
            .into_iter()
            .partition(|name| !merged.applying_chain_members(name, context).is_empty());
        let mut condition_misses = Vec::new();
        for member in applied
            .iter()
            .flat_map(|name| merged.inheritance_chain(name))
            .filter(|member| {
                merged
                    .get(member)
                    .is_some_and(|policy| !policy.applies(context))
            })
        {
            push_unique(&mut condition_misses, member);
        }
        let policy_sources = applied
            .iter()
            .filter_map(|name| {
                attached
                    .iter()
                    .find(|attachment| attachment.policy == *name)
                    .map(|attachment| ((*name).to_owned(), attachment.match_reason(context)))
            })
            .collect();

        let contributions: Vec<(&str, Vec<&str>)> = applied
            .iter()
            .map(|name| (*name, merged.guardrails(name, Some(context))))
            .collect();
        let mut guardrails = Vec::new();
        for guardrail in contributions.iter().flat_map(|(_, added)| added.iter()) {
            push_unique(&mut guardrails, guardrail);
        }

        let pipelines: Vec<(&str, &Vec<String>)> = applied
            .iter()
            .filter_map(|name| {
                let policy = merged.get(name).filter(|policy| policy.applies(context))?;
                policy
                    .pipeline_guardrails
                    .as_ref()
                    .map(|steps| (*name, steps))
            })
            .collect();
        let mut pipeline_managed_guardrails = Vec::new();
        for guardrail in pipelines.iter().flat_map(|(_, steps)| steps.iter()) {
            push_unique(&mut pipeline_managed_guardrails, guardrail.as_str());
        }

        Resolution {
            attached_policies: attached
                .iter()
                .map(|attachment| attachment.policy.clone())
                .collect(),
            unapplied_policies: owned(unapplied),
            condition_misses: owned(condition_misses),
            policy_guardrails: contributions
                .into_iter()
                .map(|(name, added)| (name.to_owned(), owned(added)))
                .collect(),
            applied_policies: owned(applied),
            policy_sources,
            guardrails: owned(guardrails),
            pipelines: pipelines
                .into_iter()
                .map(|(name, _)| PipelineSource {
                    policy: name.to_owned(),
                    version_id: merged.version_id(name).map(str::to_owned),
                })
                .collect(),
            pipeline_managed_guardrails: owned(pipeline_managed_guardrails),
            ancestor_admissions,
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};

    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn policy(inherit: Option<&str>, add: &[&str], remove: &[&str]) -> Policy {
        Policy {
            inherit: inherit.map(str::to_owned),
            add: strings(add),
            remove: strings(remove),
            ..Policy::default()
        }
    }

    fn only_for(model: &str, policy: Policy) -> Policy {
        Policy {
            condition: Some(ModelCondition::new([PatternSpec::Regex {
                text: model.to_owned(),
                host: HostRegex::new(|_| unreachable!()),
            }])),
            ..policy
        }
    }

    fn global(policy: &str) -> Attachment {
        Attachment {
            policy: policy.to_owned(),
            global: true,
            ..Attachment::default()
        }
    }

    fn context(model: &str) -> MatchContext {
        MatchContext {
            model: Some(model.to_owned()),
            ..MatchContext::default()
        }
    }

    #[fixture]
    fn engine() -> PolicyEngine {
        PolicyEngine::new(
            [
                ("base".to_owned(), policy(None, &["pii", "toxicity"], &[])),
                (
                    "child".to_owned(),
                    only_for("gpt-.*", policy(Some("base"), &["audit"], &["toxicity"])),
                ),
                ("loop-a".to_owned(), policy(Some("loop-b"), &["a"], &[])),
                ("loop-b".to_owned(), policy(Some("loop-a"), &["b"], &[])),
                ("orphan".to_owned(), policy(Some("missing"), &["o"], &[])),
                (
                    "piped".to_owned(),
                    Policy {
                        pipeline_guardrails: Some(strings(&["pii", "judge", "pii"])),
                        ..policy(None, &["judge"], &[])
                    },
                ),
            ],
            [PolicyVersion {
                id: "v1".to_owned(),
                name: "base".to_owned(),
                policy: only_for("claude-.*", policy(None, &["draft"], &[])),
            }],
            vec![global("child")],
        )
    }

    #[rstest]
    fn resolution_explains_what_was_attached_skipped_and_contributed(engine: PolicyEngine) {
        let resolution =
            engine.resolve(&context("claude-3"), &strings(&["missing", "orphan"]), &[]);
        assert_eq!(resolution.attached_policies, strings(&["child"]));
        assert_eq!(resolution.applied_policies, strings(&["child", "orphan"]));
        assert_eq!(resolution.unapplied_policies, strings(&["missing"]));
        assert_eq!(resolution.condition_misses, strings(&["child"]));
        assert_eq!(
            resolution.policy_guardrails,
            vec![
                ("child".to_owned(), strings(&["pii", "toxicity"])),
                ("orphan".to_owned(), strings(&["o"])),
            ]
        );
    }

    #[rstest]
    fn child_condition_hit_adds_and_removes_through_the_chain(engine: PolicyEngine) {
        let resolution = engine.resolve(&context("gpt-4"), &[], &[]);
        assert_eq!(resolution.applied_policies, strings(&["child"]));
        assert_eq!(
            resolution.policy_sources,
            vec![("child".to_owned(), "scope:*".to_owned())]
        );
        assert_eq!(resolution.guardrails, strings(&["pii", "audit"]));
        assert!(resolution.ancestor_admissions.is_empty());
    }

    #[rstest]
    fn child_condition_miss_still_applies_through_its_ancestor(engine: PolicyEngine) {
        let resolution = engine.resolve(&context("claude-3"), &[], &[]);
        assert_eq!(resolution.applied_policies, strings(&["child"]));
        assert_eq!(resolution.guardrails, strings(&["pii", "toxicity"]));
        assert_eq!(
            resolution.ancestor_admissions,
            vec![AncestorAdmission {
                policy: "child".to_owned(),
                ancestor: "base".to_owned(),
            }]
        );
    }

    #[rstest]
    #[case::cycle_keeps_the_walk_up_to_the_repeat(&["loop-a"], &["b", "a"])]
    #[case::missing_parent_ends_the_chain(&["orphan"], &["o"])]
    #[case::unknown_request_policy_is_dropped(&["nope"], &[])]
    fn inheritance_edges(#[case] names: &[&str], #[case] expected_guardrails: &[&str]) {
        let engine = PolicyEngine::new(engine().policies.into_iter(), [], Vec::new());
        let resolution = engine.resolve(&context("m"), &strings(names), &[]);
        assert_eq!(resolution.guardrails, strings(expected_guardrails));
        assert!(resolution.policy_sources.is_empty());
    }

    #[rstest]
    fn version_override_replaces_the_named_policy_for_this_request(engine: PolicyEngine) {
        let resolution = engine.resolve(&context("claude-3"), &[], &strings(&["v1", "unknown"]));
        assert_eq!(resolution.applied_policies, strings(&["child", "base"]));
        assert_eq!(resolution.guardrails, strings(&["draft"]));
    }

    #[rstest]
    fn a_version_whose_condition_misses_switches_its_policy_off() {
        let engine = PolicyEngine::new(
            [("base".to_owned(), policy(None, &["base-guard"], &[]))],
            [PolicyVersion {
                id: "v1".to_owned(),
                name: "base".to_owned(),
                policy: only_for("claude-.*", policy(None, &["draft"], &[])),
            }],
            vec![global("base")],
        );
        let with_version = engine.resolve(&context("gpt-4"), &[], &strings(&["v1"]));
        assert_eq!(with_version.applied_policies, Vec::<String>::new());
        assert_eq!(with_version.guardrails, Vec::<String>::new());
        assert_eq!(with_version.unapplied_policies, strings(&["base"]));
        let without_version = engine.resolve(&context("gpt-4"), &[], &[]);
        assert_eq!(without_version.guardrails, strings(&["base-guard"]));
    }

    #[rstest]
    fn pipelines_record_their_policy_and_managed_guardrails(engine: PolicyEngine) {
        let resolution = engine.resolve(&context("m"), &strings(&["piped"]), &[]);
        assert_eq!(
            resolution.pipelines,
            vec![PipelineSource {
                policy: "piped".to_owned(),
                version_id: None,
            }]
        );
        assert_eq!(
            resolution.pipeline_managed_guardrails,
            strings(&["pii", "judge"])
        );
    }

    #[rstest]
    fn defaults_apply_only_when_no_other_attachment_matches() {
        let engine = PolicyEngine::new(
            [
                ("fallback".to_owned(), policy(None, &["f"], &[])),
                ("team".to_owned(), policy(None, &["t"], &[])),
            ],
            [],
            vec![
                Attachment {
                    default: true,
                    ..global("fallback")
                },
                Attachment {
                    policy: "team".to_owned(),
                    teams: strings(&["a"]),
                    ..Attachment::default()
                },
            ],
        );
        let in_team = MatchContext {
            team_alias: Some("a".to_owned()),
            ..MatchContext::default()
        };
        assert_eq!(
            engine.resolve(&in_team, &[], &[]).applied_policies,
            strings(&["team"])
        );
        let elsewhere = engine.resolve(&MatchContext::default(), &[], &[]);
        assert_eq!(elsewhere.applied_policies, strings(&["fallback"]));
        assert_eq!(
            elsewhere.policy_sources,
            vec![("fallback".to_owned(), "default:scope:*".to_owned())]
        );
    }

    #[rstest]
    #[case::without_context_every_chain_member_counts(None, &["pii", "audit"])]
    #[case::with_context_conditions_filter_the_chain(Some("claude-3"), &["pii", "toxicity"])]
    fn one_policy_resolves_through_its_chain(
        engine: PolicyEngine,
        #[case] model: Option<&str>,
        #[case] guardrails: &[&str],
    ) {
        let context = model.map(context);
        let resolved = engine.policy_guardrails("child", context.as_ref());
        assert_eq!(resolved.guardrails, strings(guardrails));
        assert_eq!(resolved.inheritance_chain, strings(&["base", "child"]));
    }

    #[rstest]
    fn unknown_policy_resolves_to_nothing(engine: PolicyEngine) {
        assert_eq!(
            engine.policy_guardrails("nope", None),
            ResolvedPolicy::default()
        );
    }

    #[rstest]
    #[case::every_in_scope_attachment(false, &["ghost", "base"])]
    #[case::only_policies_that_apply(true, &["base"])]
    fn attachment_matching(#[case] only_applying: bool, #[case] expected: &[&str]) {
        let engine = PolicyEngine::new(
            [("base".to_owned(), policy(None, &["pii"], &[]))],
            [],
            vec![global("ghost"), global("base")],
        );
        let matched = engine.attached_policies(&context("m"), only_applying);
        let names: Vec<&str> = matched
            .matches
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        assert_eq!(names, expected);
    }

    #[test]
    fn circular_policies_names_every_policy_whose_chain_loops() {
        let engine = PolicyEngine::new(
            [
                ("a".to_owned(), policy(Some("b"), &[], &[])),
                ("b".to_owned(), policy(Some("a"), &[], &[])),
                ("into-loop".to_owned(), policy(Some("a"), &[], &[])),
                ("self".to_owned(), policy(Some("self"), &[], &[])),
                ("plain".to_owned(), policy(Some("missing"), &[], &[])),
            ],
            [],
            Vec::new(),
        );
        assert_eq!(
            engine.circular_policies(),
            strings(&["a", "b", "into-loop", "self"])
        );
    }
}
