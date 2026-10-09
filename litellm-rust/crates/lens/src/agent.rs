use crate::{
    Error,
    activity::Tracker,
    evidence::{MAX_TOOL_BYTES, Workspace},
    journal::{Journal, Turn as JournalTurn},
    model, sandbox, wire,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use std::collections::BTreeSet;

#[derive(Deserialize, Serialize)]
#[serde(untagged)]
enum Tool {
    Evidence(wire::EvidenceRequest),
    Python(wire::PythonRequest),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, bound(deserialize = "T: DeserializeOwned"))]
struct Turn<T> {
    #[serde(default)]
    tools: Vec<Tool>,
    checkpoint: Option<String>,
    result: Option<T>,
}

pub fn checks(claim: &wire::Claim) -> Result<Vec<wire::Check>, Error> {
    let mut checks: Vec<_> = claim
        .job
        .settings
        .checks
        .iter()
        .filter(|check| check.enabled)
        .cloned()
        .collect();
    if !claim.job.settings.context.trim().is_empty() {
        checks.insert(0, serde_json::from_value(json!({"id": "expected_behavior", "instruction": "Identify deviations from the expected behavior described in context."}))?);
    }
    Ok(checks)
}

pub trait Output: DeserializeOwned + Send + Sync {
    const SCHEMA: &'static str;
    fn validate(
        &self,
        claim: &wire::Claim,
        workspace: &Workspace,
    ) -> impl std::future::Future<Output = Result<Option<String>, Error>> + Send;
}

async fn evidence(
    claim: &wire::Claim,
    workspace: &Workspace,
    check_id: &str,
    quotes: &[wire::Evidence],
) -> Result<Option<String>, Error> {
    if !checks(claim)?.iter().any(|c| *c.id == check_id) {
        return Ok(Some("Use an enabled check ID".into()));
    }
    if !quotes.iter().any(|q| q.role == wire::EvidenceRole::Support) {
        return Ok(Some("Each finding or observation needs at least one supporting quote from original evidence".into()));
    }
    for quote in quotes {
        match workspace.valid(quote).await {
            Ok(true) => {},
            Ok(false) => return Ok(Some("Every evidence quote must exactly match the cited execution and span in the original recording".into())),
            Err(error) => return Ok(Some(format!("Could not verify a citation: {error}. Inspect other evidence and revise the citation."))),
        }
    }
    Ok(None)
}

impl Output for wire::Extraction {
    const SCHEMA: &'static str = "PythonAgentTurn[Extraction]";
    async fn validate(
        &self,
        claim: &wire::Claim,
        workspace: &Workspace,
    ) -> Result<Option<String>, Error> {
        for observation in &self.observations {
            if let Some(error) = evidence(
                claim,
                workspace,
                &observation.check_id,
                &observation.evidence,
            )
            .await?
            {
                return Ok(Some(error));
            }
        }
        Ok(None)
    }
}

impl Output for wire::Findings {
    const SCHEMA: &'static str = "PythonAgentTurn[Findings]";
    async fn validate(
        &self,
        claim: &wire::Claim,
        workspace: &Workspace,
    ) -> Result<Option<String>, Error> {
        let enabled: BTreeSet<_> = checks(claim)?
            .into_iter()
            .map(|c| c.id.to_string())
            .collect();
        for finding in &self.findings {
            if finding.check_ids.iter().any(|id| !enabled.contains(id)) {
                return Ok(Some("check_ids must contain only enabled check IDs".into()));
            }
            if let Some(error) =
                evidence(claim, workspace, &finding.check_id, &finding.evidence).await?
            {
                return Ok(Some(error));
            }
            if finding.kind == wire::FindingDraftKind::Issue && finding.brief.is_none() {
                return Ok(Some("Issues require a brief containing the problem, user goal, observed outcome, and test cases".into()));
            }
            if finding.existing_finding_id.as_ref().is_some_and(|id| {
                !claim
                    .findings
                    .iter()
                    .any(|f| &f.id == id && f.kind.to_string() == finding.kind.to_string())
            }) {
                return Ok(Some(
                    "Use an existing finding ID of the same kind and cause".into(),
                ));
            }
            if !finding.merged_finding_ids.is_empty() {
                return Ok(Some("Leave merged_finding_ids empty. Finding consolidation handles merging saved findings.".into()));
            }
        }
        Ok(None)
    }
}

pub struct Assignment<'a> {
    pub stage: &'a str,
    pub task: String,
    pub purpose: wire::ModelRequestPurpose,
    pub supplied: Value,
}

pub async fn run<T: Output>(
    claim: &wire::Claim,
    workspace: &Workspace,
    assignment: Assignment<'_>,
    tracker: &Tracker,
) -> Result<T, Error> {
    let existing: Vec<Value> = claim
        .findings
        .iter()
        .map(serde_json::to_value)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|mut finding| {
            if let Some(object) = finding.as_object_mut() {
                for field in ["evidence", "occurrences", "investigation_runs"] {
                    object.remove(field);
                }
            }
            finding
        })
        .collect();
    let initial =
        json!({"evidence": [], "supplied": assignment.supplied, "existing_findings": existing});
    let mut journal = Journal::new(&initial).await?;
    let prompt = json!({
        "stage": assignment.stage, "task": assignment.task,
        "response_instructions": include_str!("../prompts/response_instructions.md"),
        "tool_instructions": include_str!("../prompts/tool_instructions.md"),
        "python_instructions": include_str!("../prompts/python_instructions.md"),
        "context": claim.job.settings.context, "checks": checks(claim)?,
        "catalog_fields": ["span_id", "parent_span_id", "name", "kind", "characters", "start_time", "end_time"],
        "available_sessions": workspace.executions.len(), "available_review_records": workspace.reviews.len(),
        "response_schema": model::schema(T::SCHEMA)?,
    });
    let mut request = model::request(assignment.purpose, prompt)?;
    let task_message = model::message(wire::ModelMessageRole::System, request.prompt.to_string());
    request.messages = vec![task_message.clone(), model::message(wire::ModelMessageRole::User, json!({"initial_evidence": [], "supplied": assignment.supplied, "existing_findings": existing}).to_string())];
    let mut compacted = false;
    let mut rejected = 0;
    loop {
        tracker.change("model", true).await?;
        let result = model::structured::<Turn<T>>(&workspace.client, request.clone(), T::SCHEMA, |turn| {
            if (turn.tools.is_empty() && turn.checkpoint.is_none()) != turn.result.is_some() {
                return Some("Return tools and/or a checkpoint with result=null, or a final result without tools or checkpoint".into());
            }
            if turn.checkpoint.as_ref().is_some_and(|c| c.is_empty()) { return Some("Checkpoint must not be empty".into()); }
            None
        }).await;
        tracker.change("model", false).await?;
        let (turn, responded) = match result {
            Err(Error::Context(previous)) if !compacted => {
                tracker.change("checkpoint", true).await?;
                request.messages =
                    model::compact(&workspace.client, *previous, journal.turns.len() + 1).await?;
                tracker.change("checkpoint", false).await?;
                journal
                    .push(&JournalTurn {
                        response: request.messages[1].content.clone(),
                        tool_results: Vec::new(),
                        validation_error: String::new(),
                    })
                    .await?;
                compacted = true;
                continue;
            }
            Err(Error::Context(_)) => {
                return Err(Error::CompactedContext);
            }
            result => result?,
        };
        compacted = false;
        if let Some(result) = turn.result {
            let Some(invalid) = result.validate(claim, workspace).await? else {
                return Ok(result);
            };
            rejected += 1;
            journal
                .push(&JournalTurn {
                    response: responded
                        .last()
                        .ok_or(Error::InvalidRequest)?
                        .content
                        .clone(),
                    tool_results: Vec::new(),
                    validation_error: invalid.clone(),
                })
                .await?;
            if rejected > 3 {
                return Err(Error::ModelValidation {
                    schema: T::SCHEMA,
                    detail: invalid,
                });
            }
            request.messages = responded;
            request.messages.push(model::message(
                wire::ModelMessageRole::User,
                json!({"journal_turns": journal.turns.len()}).to_string(),
            ));
            request.messages.push(model::message(wire::ModelMessageRole::System, json!({"instruction": "Correct the validation errors using original evidence. Tools remain available. Verify exact quotes and remove claims the evidence cannot support. Continue using the task response_schema.", "validation_errors": invalid}).to_string()));
            continue;
        }
        let mut results = Vec::new();
        let mut archived = Vec::new();
        let mut bytes = 0;
        for tool in turn.tools {
            let operation = match &tool {
                Tool::Evidence(r) => r.action.to_string(),
                Tool::Python(_) => "python".into(),
            };
            tracker.change(&operation, true).await?;
            let result = match &tool {
                Tool::Evidence(request)
                    if request.action == wire::EvidenceRequestAction::History =>
                {
                    journal.reply(request).await
                }
                Tool::Evidence(request) => workspace.respond(request).await,
                Tool::Python(request) => sandbox::execute(workspace, request)
                    .await
                    .map(|output| json!({"request": request, "output": output})),
            };
            tracker.change(&operation, false).await?;
            let result = match result {
                Ok(value) => value.to_string(),
                Err(error) => json!({"request": tool, "error": error.to_string()}).to_string(),
            };
            archived.push(match &tool {
                Tool::Evidence(r) => journal.reference(r).unwrap_or_else(|| result.clone()),
                _ => result.clone(),
            });
            bytes += result.len();
            if bytes > MAX_TOOL_BYTES {
                let error = json!({"request": tool, "error": "Combined tool output exceeds 8 MiB. Request smaller ranges or fewer tools per turn."}).to_string();
                results.push(error);
                continue;
            }
            results.push(result);
        }
        journal
            .push(&JournalTurn {
                response: responded
                    .last()
                    .ok_or(Error::InvalidRequest)?
                    .content
                    .clone(),
                tool_results: archived,
                validation_error: String::new(),
            })
            .await?;
        request.messages = if let Some(checkpoint) = turn.checkpoint {
            tracker.change("checkpoint", true).await?;
            let messages = vec![
                task_message.clone(),
                model::message(
                    wire::ModelMessageRole::User,
                    json!({"working_notes": checkpoint, "initial_context_archived": true})
                        .to_string(),
                ),
                responded.last().ok_or(Error::InvalidRequest)?.clone(),
            ];
            tracker.change("checkpoint", false).await?;
            messages
        } else {
            responded
        };
        request.messages.push(model::message(
            wire::ModelMessageRole::User,
            json!({"journal_turns": journal.turns.len(), "tool_results": results}).to_string(),
        ));
        if request
            .messages
            .iter()
            .map(|m| m.content.len())
            .sum::<usize>()
            > 16 * 1024 * 1024
        {
            request.messages =
                model::compact(&workspace.client, request.clone(), journal.turns.len()).await?;
            compacted = true;
        }
    }
}
