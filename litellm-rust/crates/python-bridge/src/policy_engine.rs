//! The Python face of `litellm-policy-engine`. `litellm/rust_bridge/policy_engine.py` builds one
//! `PolicyEngine` per registry snapshot from plain tuples, reading the pydantic models with the
//! same getters the registries use. A condition pattern arrives with its compiled `re.Pattern`,
//! which Rust calls only for syntax it cannot read the way Python does.
use litellm_policy_engine::{
    Attachment, AttachmentMatches, HostRegex, MatchContext, ModelCondition, PatternSpec, Policy,
    PolicyEngine, PolicyVersion, Resolution, ResolvedPolicy,
};
use pyo3::prelude::*;

type NamedPairs = Vec<(String, String)>;

fn host_regex(compiled: Py<PyAny>) -> HostRegex {
    HostRegex::new(move |value| {
        Python::attach(|py| {
            compiled
                .bind(py)
                .call_method1("fullmatch", (value,))
                .is_ok_and(|matched| !matched.is_none())
        })
    })
}

#[derive(FromPyObject)]
struct PolicyRow(
    Option<String>,
    Vec<String>,
    Vec<String>,
    Option<Vec<(String, Option<Py<PyAny>>)>>,
    Option<Vec<String>>,
);

impl From<PolicyRow> for Policy {
    fn from(row: PolicyRow) -> Self {
        let PolicyRow(inherit, add, remove, condition, pipeline_guardrails) = row;
        Self {
            inherit,
            add,
            remove,
            condition: condition.map(|patterns| {
                ModelCondition::new(patterns.into_iter().map(|(text, compiled)| match compiled {
                    Some(compiled) => PatternSpec::Regex {
                        text,
                        host: host_regex(compiled),
                    },
                    None => PatternSpec::Literal(text),
                }))
            }),
            pipeline_guardrails,
        }
    }
}

#[derive(FromPyObject)]
struct AttachmentRow(
    String,
    bool,
    Vec<String>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
    Option<i32>,
    bool,
);

impl From<AttachmentRow> for Attachment {
    fn from(row: AttachmentRow) -> Self {
        let AttachmentRow(policy, global, teams, keys, models, tags, priority, default) = row;
        Self {
            policy,
            global,
            teams,
            keys,
            models,
            tags,
            priority,
            default,
        }
    }
}

#[derive(FromPyObject)]
struct ContextRow(Option<String>, Option<String>, Option<String>, Vec<String>);

impl From<ContextRow> for MatchContext {
    fn from(row: ContextRow) -> Self {
        let ContextRow(team_alias, key_alias, model, tags) = row;
        Self {
            team_alias,
            key_alias,
            model,
            tags,
        }
    }
}

#[pyclass(frozen, name = "PolicyEngine")]
pub(crate) struct NativePolicyEngine {
    engine: PolicyEngine,
}

#[pymethods]
impl NativePolicyEngine {
    #[new]
    fn new(
        policies: Vec<(String, PolicyRow)>,
        versions: Vec<(String, String, PolicyRow)>,
        attachments: Vec<AttachmentRow>,
    ) -> Self {
        let engine = PolicyEngine::new(
            policies
                .into_iter()
                .map(|(name, row)| (name, Policy::from(row))),
            versions.into_iter().map(|(id, name, row)| PolicyVersion {
                id,
                name,
                policy: row.into(),
            }),
            attachments.into_iter().map(Attachment::from).collect(),
        );
        Self { engine }
    }

    fn resolve(
        &self,
        context: ContextRow,
        names: Vec<String>,
        version_ids: Vec<String>,
    ) -> PolicyResolution {
        self.engine
            .resolve(&context.into(), &names, &version_ids)
            .into()
    }

    #[pyo3(signature = (name, context=None))]
    fn policy_guardrails(
        &self,
        name: &str,
        context: Option<ContextRow>,
    ) -> (Vec<String>, Vec<String>) {
        let context = context.map(MatchContext::from);
        let ResolvedPolicy {
            guardrails,
            inheritance_chain,
        } = self.engine.policy_guardrails(name, context.as_ref());
        (guardrails, inheritance_chain)
    }

    fn attached_policies(
        &self,
        context: ContextRow,
        only_applying: bool,
    ) -> (NamedPairs, NamedPairs) {
        let AttachmentMatches {
            matches,
            ancestor_admissions,
        } = self
            .engine
            .attached_policies(&context.into(), only_applying);
        (
            matches,
            ancestor_admissions
                .into_iter()
                .map(|admission| (admission.policy, admission.ancestor))
                .collect(),
        )
    }
}

#[pyclass(frozen, get_all, name = "PolicyResolution")]
pub(crate) struct PolicyResolution {
    applied_policies: Vec<String>,
    policy_sources: Vec<(String, String)>,
    guardrails: Vec<String>,
    pipelines: Vec<(String, Option<String>)>,
    pipeline_managed_guardrails: Vec<String>,
    ancestor_admissions: Vec<(String, String)>,
}

impl From<Resolution> for PolicyResolution {
    fn from(resolution: Resolution) -> Self {
        Self {
            applied_policies: resolution.applied_policies,
            policy_sources: resolution.policy_sources,
            guardrails: resolution.guardrails,
            pipelines: resolution
                .pipelines
                .into_iter()
                .map(|source| (source.policy, source.version_id))
                .collect(),
            pipeline_managed_guardrails: resolution.pipeline_managed_guardrails,
            ancestor_admissions: resolution
                .ancestor_admissions
                .into_iter()
                .map(|admission| (admission.policy, admission.ancestor))
                .collect(),
        }
    }
}
