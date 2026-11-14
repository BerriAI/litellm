use std::cell::RefCell;

use litellm_callbacks::CallbackError;
use serde_json::Value;

use crate::{
    attempt::AttemptFacts,
    emit::{Emitter, failed},
    labels::LabelValues,
    success::{LogLine, Success, get, metadata, stamped},
};

pub struct Failure<'a> {
    pub emit: Emitter<'a>,
    pub facts: &'a AttemptFacts,
    pub logs: &'a RefCell<Vec<LogLine>>,
}

impl Failure<'_> {
    pub fn run(&self) -> Result<(), CallbackError> {
        let facts = self.facts;
        let identity = metadata(facts)?;
        let labels = &facts.residual.payload_labels;
        let user_id = stamped(identity, "user_api_key_user_id", identity.user_id.as_ref())?;
        let hashed_api_key = stamped(identity, "user_api_key_hash", identity.key_hash.as_ref())?;
        let api_key_alias = stamped(identity, "user_api_key_alias", identity.key_alias.as_ref())?;
        let team = stamped(identity, "user_api_key_team_id", identity.team_id.as_ref())?;
        let team_alias = stamped(
            identity,
            "user_api_key_team_alias",
            identity.team_alias.as_ref(),
        )?;
        let values = LabelValues {
            custom_metadata_labels: facts.custom_metadata_labels(),
            ..LabelValues::default()
        }
        .with("end_user", facts.end_user_id())
        .with("hashed_api_key", hashed_api_key)
        .with("api_key_alias", api_key_alias)
        .with("model", facts.model.clone())
        .with("team", team)
        .with("team_alias", team_alias)
        .with("user", user_id)
        .with(
            "model_id",
            labels.get("model_id").cloned().unwrap_or(Value::from("")),
        )
        .with("requested_model", get(labels, "model_group").clone())
        .with("api_provider", get(labels, "custom_llm_provider").clone());
        let context = values.checked_context(facts.settings()).map_err(failed)?;
        self.emit
            .inc("litellm_llm_api_failed_requests_metric", &context, 1.0)?;
        let success = Success {
            emit: Emitter {
                store: self.emit.store,
                settings: self.emit.settings,
            },
            facts,
            input_sequence_length: Value::Null,
            logs: self.logs,
        };
        success.zero_cost(&values)
    }
}
