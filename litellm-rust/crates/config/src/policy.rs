use serde::Deserialize;

use crate::{Raw, Spelled};

#[macro_rules_attribute::apply(crate::config_section)]
pub struct Policy {
    pub guardrails: Option<PolicyGuardrails>,
    pub condition: Option<PolicyCondition>,
    pub pipeline: Option<PolicyPipeline>,
    pub inherit: Option<String>,
    pub description: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum PolicyGuardrails {
    Added(Box<[String]>),
    Changes(PolicyGuardrailChanges),
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct PolicyGuardrailChanges {
    pub add: Option<Box<[String]>>,
    pub remove: Option<Box<[String]>>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct PolicyCondition {
    pub model: Option<Raw>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct PolicyPipeline {
    pub mode: Option<String>,
    pub steps: Option<Box<[PolicyPipelineStep]>>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct PolicyPipelineStep {
    pub guardrail: Option<String>,
    pub on_fail: Option<String>,
    pub on_pass: Option<String>,
    pub on_error: Option<String>,
    pub pass_data: Option<Spelled<bool>>,
    pub modify_response_message: Option<String>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct PolicyAttachment {
    pub policy: Option<String>,
    pub scope: Option<String>,
    pub teams: Option<Box<[String]>>,
    pub keys: Option<Box<[String]>>,
    pub models: Option<Box<[String]>>,
    pub tags: Option<Box<[String]>>,
    pub priority: Option<Spelled<i64>>,
    pub default: Option<Spelled<bool>>,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::{Config, PolicyGuardrails, Spelled, Value};

    fn parse(yaml: &str) -> Config {
        Config::from_yaml(yaml).unwrap()
    }

    #[rstest]
    #[case::legacy_list("guardrails: [pii, toxicity]", &["pii", "toxicity"], &[])]
    #[case::add_remove("guardrails: {add: [pii], remove: [toxicity]}", &["pii"], &["toxicity"])]
    fn policy_guardrails_accept_both_python_spellings(
        #[case] guardrails: &str,
        #[case] added: &[&str],
        #[case] removed: &[&str],
    ) {
        let config = parse(&format!("policies:\n  safe:\n    {guardrails}\n"));
        let (add, remove) = match config.policies["safe"].guardrails.as_ref().unwrap() {
            PolicyGuardrails::Added(add) => (add.to_vec(), Vec::new()),
            PolicyGuardrails::Changes(changes) => (
                changes.add.as_deref().unwrap_or_default().to_vec(),
                changes.remove.as_deref().unwrap_or_default().to_vec(),
            ),
        };

        assert_eq!(add, added);
        assert_eq!(remove, removed);
    }

    #[rstest]
    fn policies_and_attachments_parse_pipeline_condition_and_scope() {
        let config = parse(
            r#"
policies:
  safe:
    inherit: base
    condition:
      model: [gpt-a, gpt-b]
    pipeline:
      mode: pre_call
      steps:
        - guardrail: pii
          on_fail: next
          pass_data: true
policy_attachments:
  - policy: safe
    teams: [research]
    priority: 10
    default: true
"#,
        );
        let policy = &config.policies["safe"];
        let step = &policy.pipeline.as_ref().unwrap().steps.as_ref().unwrap()[0];
        let attachment = &config.policy_attachments[0];

        assert_eq!(policy.inherit.as_deref(), Some("base"));
        assert!(matches!(
            policy.condition.as_ref().unwrap().model.as_ref().unwrap().get(),
            Value::Sequence(models) if models.len() == 2
        ));
        assert_eq!(step.guardrail.as_deref(), Some("pii"));
        assert_eq!(step.on_fail.as_deref(), Some("next"));
        assert_eq!(step.pass_data, Some(Spelled::Value(true)));
        assert_eq!(attachment.policy.as_deref(), Some("safe"));
        assert_eq!(
            attachment.teams.as_deref(),
            Some(&["research".to_owned()][..])
        );
        assert_eq!(attachment.priority, Some(Spelled::Value(10)));
        assert_eq!(attachment.default, Some(Spelled::Value(true)));
    }

    #[rstest]
    #[case::policy_model(
        r#"
policies:
  safe: {condition: {model: SENTINEL}}
"#
    )]
    fn debug_never_prints_policy_values(#[case] yaml: &str) {
        assert!(!format!("{:?}", parse(yaml)).contains("SENTINEL"));
    }
}
