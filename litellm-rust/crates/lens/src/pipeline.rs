use crate::{
    Error,
    activity::Tracker,
    agent::{self, Assignment},
    control::JobClient,
    evidence::{Workspace, character_range},
    grouping, wire,
};
use futures_util::{StreamExt, stream};
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Instant,
};
use tokio::sync::Mutex;

struct Outcome {
    review: wire::Review,
    error: String,
}

struct ReviewProgress {
    coverage: wire::Coverage,
    reading: Vec<wire::InFlight>,
}

impl ReviewProgress {
    async fn publish(&self, client: &JobClient, review: Option<wire::Review>) -> Result<(), Error> {
        client
            .progress(&wire::Progress {
                stage: Some("Reading executions".into()),
                coverage: Some(self.coverage.clone()),
                reading: Some(self.reading.clone()),
                review,
                ..Default::default()
            })
            .await
    }
}

async fn review(
    claim: &wire::Claim,
    workspace: &Workspace,
    execution: &wire::Execution,
    progress: &Mutex<ReviewProgress>,
) -> Result<Outcome, Error> {
    let started = Instant::now();
    {
        let mut progress = progress.lock().await;
        progress.reading.push(wire::InFlight {
            execution_id: execution.id.clone(),
            trace_id: execution.trace_id.clone(),
            agent: if execution.service.is_empty() {
                execution.name.clone()
            } else {
                execution.service.clone()
            },
            started_at: chrono::Utc::now(),
        });
        progress.publish(&workspace.client, None).await?;
    }
    let tracker = Tracker::start(
        &workspace.client,
        format!("review:{}", execution.id),
        wire::ActivityPhase::Review,
        execution.name.clone(),
        vec![execution.id.clone()],
    )
    .await?;
    let version = workspace.fingerprint(execution).await;
    let previous = version.as_ref().ok().and_then(|version| {
        claim.reviews.as_ref()?.iter().find(|r| {
            r.execution_id == execution.id
                && &r.content_version == version
                && r.extraction.is_some()
        })
    });
    let (extraction, error) = if let Some(previous) = previous {
        (
            previous.extraction.clone().unwrap_or_default(),
            String::new(),
        )
    } else if let Err(error) = &version {
        (
            wire::Extraction {
                cannot_assess: true,
                ..Default::default()
            },
            error.to_string(),
        )
    } else {
        let mut local_claim = claim.clone();
        let mut local_workspace = workspace.clone();
        if claim.reviews.is_some() {
            local_claim.findings.clear();
            local_workspace.executions = vec![execution.clone()];
        }
        let result = agent::run::<wire::Extraction>(&local_claim, &local_workspace, Assignment {
            stage: "context_review", purpose: wire::ModelRequestPurpose::Extract,
            task: format!("{}\nReview the assigned execution, including its recorded subagents. Original evidence is available through tools. Inspect actual trace evidence before concluding there are no issues; metadata alone is not enough. The result field follows the Extraction schema.", include_str!("../../../../litellm/proxy/lens/prompts/review.md")),
            supplied: json!({"execution": execution, "characters": null, "recorded_spans": execution.span_count, "partial": workspace.partial(execution)}),
        }, &tracker).await;
        match result {
            Ok(extraction) => (extraction, String::new()),
            Err(error) if error.is_control_failure() => {
                tracker.finish().await?;
                return Err(error);
            }
            Err(error) => (
                wire::Extraction {
                    cannot_assess: true,
                    ..Default::default()
                },
                error.to_string(),
            ),
        }
    };
    let tool_calls = tracker.finish().await?;
    let (extraction, error) = if workspace.read_failed(&execution.id) {
        (
            wire::Extraction {
                cannot_assess: true,
                ..Default::default()
            },
            Error::EvidenceUnavailable.to_string(),
        )
    } else {
        (extraction, error)
    };
    let reasoning = if error.is_empty() {
        extraction.reasoning.to_string()
    } else {
        character_range(&error, 0, Some(800))
    };
    let content_version = version.unwrap_or_default();
    let review: wire::Review = serde_json::from_value(json!({
        "execution_id": execution.id, "trace_id": execution.trace_id, "agent": if execution.service.is_empty() { &execution.name } else { &execution.service }, "name": execution.name,
        "spans": previous.map(|review| review.spans.clone()).unwrap_or_else(|| workspace.previews(&execution.id)), "reasoning": reasoning,
        "verdicts": extraction.observations.iter().filter(|o| o.evidence.iter().any(|q| q.execution_id == execution.id && q.role == wire::EvidenceRole::Support)).map(|o| json!({"check_id": o.check_id, "kind": o.kind, "summary": character_range(&o.summary, 0, Some(300))})).collect::<Vec<_>>(),
        "cannot_assess": extraction.cannot_assess, "model": claim.job.settings.model, "duration_ms": started.elapsed().as_millis() as u64, "at": chrono::Utc::now(), "tool_calls": tool_calls,
        "extraction": if !content_version.is_empty() && error.is_empty() { Some(&extraction) } else { None }, "content_version": content_version,
        "reused": previous.is_some(), "consolidated": previous.is_some_and(|r| r.consolidated), "partial": workspace.partial(execution) || previous.is_some_and(|r| r.partial),
    }))?;
    {
        let mut progress = progress.lock().await;
        progress.coverage.screened += 1;
        progress.coverage.reused += u64::from(previous.is_some());
        progress.coverage.reusable += u64::from(previous.is_some());
        progress.reading.retain(|r| r.execution_id != execution.id);
        progress
            .publish(&workspace.client, Some(review.clone()))
            .await?;
    }
    Ok(Outcome { review, error })
}

fn result(coverage: wire::Coverage) -> wire::Result {
    wire::Result {
        coverage,
        findings: Vec::new(),
        assessments: Vec::new(),
        review_versions: Vec::new(),
        error: String::new(),
    }
}

pub async fn analyze(
    claim: &wire::Claim,
    sample: wire::Sample,
    client: JobClient,
) -> Result<wire::Result, Error> {
    let mut result = result(wire::Coverage {
        eligible: sample.eligible,
        selected: sample.executions.len() as i64,
        ..Default::default()
    });
    if sample.executions.is_empty() {
        return Ok(result);
    }
    let mut workspace = Workspace::new(sample.executions, client.clone());
    let concurrency = (claim.job.settings.concurrency.get() as usize).clamp(1, 16);
    let progress = Arc::new(Mutex::new(ReviewProgress {
        coverage: result.coverage.clone(),
        reading: Vec::new(),
    }));
    progress.lock().await.publish(&client, None).await?;
    let mut completed = BTreeMap::new();
    let mut errors = BTreeSet::new();
    {
        let jobs: Vec<_> = workspace
            .executions
            .iter()
            .map(|execution| review(claim, &workspace, execution, &progress))
            .collect();
        let calls = stream::iter(jobs).buffer_unordered(concurrency);
        futures_util::pin_mut!(calls);
        while let Some(review) = calls.next().await {
            match review {
                Ok(outcome) => {
                    completed.insert(outcome.review.execution_id.clone(), outcome);
                }
                Err(error) => {
                    errors.insert(error.to_string());
                    break;
                }
            }
        }
    }
    client
        .progress(&wire::Progress {
            reading: Some(Vec::new()),
            ..Default::default()
        })
        .await?;
    let outcomes: Vec<_> = workspace
        .executions
        .iter()
        .filter_map(|execution| completed.remove(&execution.id))
        .collect();
    result.coverage.screened = outcomes.len() as i64;
    result.coverage.partial = outcomes.iter().filter(|o| o.review.partial).count() as i64;
    result.coverage.unassessable =
        outcomes.iter().filter(|o| o.review.cannot_assess).count() as i64;
    result.coverage.failed_tasks = outcomes.iter().filter(|o| !o.error.is_empty()).count() as u64;
    result.coverage.reused = outcomes.iter().filter(|o| o.review.reused).count() as u64;
    result.coverage.reusable = result.coverage.reused;
    let observations: Vec<_> = outcomes
        .iter()
        .filter_map(|o| o.review.extraction.as_ref())
        .flat_map(|e| &e.observations)
        .collect();
    result.assessments = outcomes
        .iter()
        .map(|o| wire::RunAssessment {
            execution_id: o.review.execution_id.clone(),
            cannot_assess: o.review.cannot_assess,
            issue_checks: observations
                .iter()
                .filter(|ob| {
                    ob.kind == wire::ObservationKind::Issue
                        && ob.evidence.iter().any(|q| {
                            q.execution_id == o.review.execution_id
                                && q.role == wire::EvidenceRole::Support
                        })
                })
                .map(|ob| ob.check_id.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
            pattern_checks: observations
                .iter()
                .filter(|ob| {
                    ob.kind == wire::ObservationKind::Pattern
                        && ob.evidence.iter().any(|q| {
                            q.execution_id == o.review.execution_id
                                && q.role == wire::EvidenceRole::Support
                        })
                })
                .map(|ob| ob.check_id.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
        })
        .collect();
    result.review_versions = outcomes
        .iter()
        .filter(|o| {
            o.error.is_empty()
                && !o.review.content_version.is_empty()
                && !workspace.read_failed(&o.review.execution_id)
        })
        .map(|o| wire::ReviewVersion {
            execution_id: o.review.execution_id.clone(),
            content_version: o.review.content_version.clone(),
        })
        .collect();
    let pending: Vec<_> = outcomes
        .iter()
        .filter(|o| !o.review.consolidated)
        .filter_map(|o| o.review.extraction.as_ref())
        .flat_map(|e| e.observations.iter().cloned())
        .collect();
    let stopped = !errors.is_empty();
    errors.extend(
        outcomes
            .iter()
            .filter(|o| !o.error.is_empty())
            .map(|o| o.error.clone()),
    );
    if stopped || pending.is_empty() {
        if stopped {
            result.review_versions.clear();
        }
        errors.extend(workspace.errors());
        result.error = errors.into_iter().collect::<Vec<_>>().join("\n\n");
        return Ok(result);
    }
    workspace.reviews = outcomes
        .iter()
        .filter_map(|o| o.review.extraction.as_ref().map(|e| (&o.review, e)))
        .map(|(r, e)| {
            Ok(wire::ReviewRecord {
                execution_id: r.execution_id.clone(),
                phase: wire::ReviewRecordPhase::Initial,
                content: serde_json::to_string(e)?,
            })
        })
        .collect::<Result<_, Error>>()?;
    let candidates =
        match grouping::group(&client, &pending, &mut result.coverage, concurrency).await {
            Ok(candidates) => candidates,
            Err(error) => {
                result.review_versions.clear();
                errors.insert(error.to_string());
                result.error = errors.into_iter().collect::<Vec<_>>().join("\n\n");
                return Ok(result);
            }
        };
    result.coverage.candidates = candidates.len() as i64;
    client
        .progress(&wire::Progress {
            stage: Some("Checking original evidence".into()),
            coverage: Some(result.coverage.clone()),
            ..Default::default()
        })
        .await?;
    let jobs: Vec<_> = candidates.iter().enumerate().map(|(index, candidate)| {
        let workspace = &workspace;
        let client = &client;
        async move {
            let tracker = Tracker::start(client, format!("investigate:{index}"), wire::ActivityPhase::Investigate, candidate.title.clone(), candidate.execution_ids.clone()).await?;
            let result = agent::run::<wire::Findings>(claim, workspace, Assignment {
                stage: "context_investigation", purpose: wire::ModelRequestPurpose::Investigate,
                task: format!("{}\nInvestigate the supplied candidate against original evidence, including counterexamples. Use read_reviews for the candidate sessions and search_reviews to compare other sessions. All sampled sessions and nested agents remain available. Finalize findings about this candidate's check and underlying causes. Unrelated successes are context or counterevidence, not additional findings. Preserve distinct supported causes if the candidate conflates them. Return every supported finding, or an empty findings list if unsupported.", include_str!("../prompts/findings.md")),
                supplied: serde_json::to_value(candidate)?,
            }, &tracker).await;
            tracker.finish().await?;
            Ok::<_, Error>((index, result))
        }
    }).collect();
    let calls = stream::iter(jobs).buffer_unordered(concurrency);
    futures_util::pin_mut!(calls);
    let mut drafts = BTreeMap::new();
    let mut unfinished = BTreeSet::new();
    while let Some(outcome) = calls.next().await {
        let (index, outcome) = match outcome {
            Ok(outcome) => outcome,
            Err(error) if error.is_control_failure() => return Err(error),
            Err(error) => {
                errors.insert(error.to_string());
                result.review_versions.clear();
                break;
            }
        };
        result.coverage.investigated += 1;
        match outcome {
            Ok(findings) => {
                result.coverage.inconclusive += i64::from(findings.findings.is_empty());
                drafts.insert(index, findings.findings);
            }
            Err(error) if error.is_control_failure() => return Err(error),
            Err(error) => {
                result.coverage.failed_tasks += 1;
                result.coverage.inconclusive += 1;
                unfinished.extend(candidates[index].execution_ids.iter().cloned());
                errors.insert(error.to_string());
            }
        }
        client
            .progress(&wire::Progress {
                stage: Some("Checking original evidence".into()),
                coverage: Some(result.coverage.clone()),
                ..Default::default()
            })
            .await?;
    }
    client
        .progress(&wire::Progress {
            stage: Some("Consolidating findings across runs".into()),
            ..Default::default()
        })
        .await?;
    match grouping::consolidate(
        &client,
        drafts.into_values().flatten().collect(),
        &claim.findings,
    )
    .await
    {
        Ok(findings) => result.findings = findings,
        Err(error) => {
            result.review_versions.clear();
            errors.insert(format!("Finding consolidation is incomplete: {error}"));
        }
    }
    result.review_versions.retain(|r| {
        !unfinished.contains(&r.execution_id) && !workspace.read_failed(&r.execution_id)
    });
    result.coverage.partial = workspace
        .executions
        .iter()
        .filter(|e| workspace.partial(e))
        .count() as i64;
    errors.extend(workspace.errors());
    result.error = errors.into_iter().collect::<Vec<_>>().join("\n\n");
    Ok(result)
}
