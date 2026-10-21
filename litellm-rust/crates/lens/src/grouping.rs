use crate::{Error, activity::Tracker, control::JobClient, model, wire};
use futures_util::{StreamExt, stream};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

async fn merge(
    client: &JobClient,
    candidates: &[wire::Candidate],
    prior_count: usize,
) -> Result<(Vec<wire::Candidate>, Vec<wire::Candidate>), Error> {
    let inputs: BTreeMap<_, _> = candidates
        .iter()
        .enumerate()
        .map(|(i, candidate)| (format!("p{i}"), (i, candidate)))
        .collect();
    let request = model::request(
        wire::ModelRequestPurpose::Cluster,
        json!({
            "task": include_str!("../../../../litellm/proxy/lens/prompts/cluster.md"),
            "response_schema": model::schema("Clusters")?,
            "candidates": inputs.iter().map(|(id, (_, c))| wire::Candidate { execution_ids: vec![id.clone()], ..(*c).clone() }).collect::<Vec<_>>(),
        }),
    )?;
    let (groups, _) = model::structured::<wire::Clusters>(client, request, "Clusters", |groups| {
        let mut seen = BTreeSet::new();
        if groups.candidates.iter().flat_map(|c| &c.execution_ids).any(|id| !seen.insert(id)) { Some("Each input reference must appear in exactly one group. Do not duplicate references.".into()) } else { None }
    }).await?;
    let mut used = BTreeSet::new();
    let mut expanded = Vec::new();
    for mut group in groups.candidates {
        if group.execution_ids.is_empty()
            || group.execution_ids.iter().any(|id| {
                inputs
                    .get(id)
                    .is_none_or(|(_, c)| c.check_id != group.check_id || c.kind != group.kind)
            })
        {
            continue;
        }
        let active = group
            .execution_ids
            .iter()
            .any(|id| inputs[id].0 >= prior_count);
        used.extend(group.execution_ids.iter().cloned());
        group.execution_ids = group
            .execution_ids
            .iter()
            .flat_map(|id| inputs[id].1.execution_ids.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        expanded.push((group, active));
    }
    expanded.extend(
        inputs
            .into_iter()
            .filter(|(id, _)| !used.contains(id))
            .map(|(_, (index, candidate))| (candidate.clone(), index >= prior_count)),
    );
    let (active, preserved): (Vec<_>, Vec<_>) =
        expanded.into_iter().partition(|(_, active)| *active);
    Ok((
        active.into_iter().map(|(c, _)| c).collect(),
        preserved.into_iter().map(|(c, _)| c).collect(),
    ))
}

async fn registry(
    client: &JobClient,
    candidates: Vec<wire::Candidate>,
) -> Result<Vec<wire::Candidate>, Error> {
    let mut registry = Vec::new();
    for candidate in candidates {
        if registry.is_empty() {
            registry.push(candidate);
            continue;
        }
        let mut pending = VecDeque::from([std::mem::take(&mut registry)]);
        let mut active = vec![candidate];
        while let Some(prior) = pending.pop_front() {
            let combined: Vec<_> = prior.iter().chain(&active).cloned().collect();
            match merge(client, &combined, prior.len()).await {
                Ok((continued, preserved)) => {
                    active = continued;
                    registry.extend(preserved);
                }
                Err(Error::Context(_)) if prior.len() > 1 => {
                    let midpoint = prior.len() / 2;
                    pending.push_front(prior[midpoint..].to_vec());
                    pending.push_front(prior[..midpoint].to_vec());
                }
                Err(Error::Context(_)) => {
                    return Err(Error::CandidateContext);
                }
                Err(error) => return Err(error),
            }
        }
        registry.extend(active);
    }
    Ok(registry)
}

async fn reconcile_candidates(
    client: &JobClient,
    candidates: Vec<wire::Candidate>,
) -> Result<Vec<wire::Candidate>, Error> {
    match merge(client, &candidates, 0).await {
        Ok((mut active, preserved)) => {
            active.extend(preserved);
            Ok(active)
        }
        Err(Error::Context(_)) => registry(client, candidates).await,
        Err(error) => Err(error),
    }
}

pub async fn group(
    client: &JobClient,
    observations: &[wire::Observation],
    coverage: &mut wire::Coverage,
    concurrency: usize,
) -> Result<Vec<wire::Candidate>, Error> {
    let mut ordered = observations.to_vec();
    ordered.sort_by(|a, b| (&a.check_id, a.kind).cmp(&(&b.check_id, b.kind)));
    let mut batches = Vec::<Vec<wire::Observation>>::new();
    let mut size = 0;
    for observation in ordered {
        let length = serde_json::to_string(&observation)?.chars().count();
        if batches.is_empty() || (size + length > 16000 && size > 0) {
            batches.push(Vec::new());
            size = 0;
        }
        size += length;
        if let Some(batch) = batches.last_mut() {
            batch.push(observation);
        }
    }
    coverage.grouping_batches = batches.len() as i64;
    client
        .progress(&wire::Progress {
            stage: Some("Grouping observations".into()),
            coverage: Some(coverage.clone()),
            ..Default::default()
        })
        .await?;
    let calls = stream::iter(batches.into_iter().enumerate().map(
        |(index, observations)| async move {
            let candidates = observations
                .into_iter()
                .map(|observation| {
                    Ok(wire::Candidate {
                        check_id: observation.check_id,
                        title: observation.summary.clone(),
                        hypothesis: format!("{}: {}", observation.kind, observation.summary),
                        kind: serde_json::from_value(serde_json::to_value(observation.kind)?)?,
                        execution_ids: observation
                            .evidence
                            .iter()
                            .filter(|q| q.role == wire::EvidenceRole::Support)
                            .map(|q| q.execution_id.clone())
                            .collect::<BTreeSet<_>>()
                            .into_iter()
                            .collect(),
                        existing_finding_id: None,
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let tracker = Tracker::start(
                client,
                format!("group:{index}"),
                wire::ActivityPhase::Group,
                format!("Compare observation batch {}", index + 1),
                candidates
                    .iter()
                    .flat_map(|c| c.execution_ids.iter().cloned())
                    .collect(),
            )
            .await?;
            let result = reconcile_candidates(client, candidates).await;
            tracker.finish().await?;
            Ok::<_, Error>((index, result?))
        },
    ))
    .buffer_unordered(concurrency);
    futures_util::pin_mut!(calls);
    let mut completed = BTreeMap::new();
    while let Some(result) = calls.next().await {
        let (index, candidates) = result?;
        completed.insert(index, candidates);
        coverage.grouped_batches += 1;
        client
            .progress(&wire::Progress {
                stage: Some("Grouping observations".into()),
                coverage: Some(coverage.clone()),
                ..Default::default()
            })
            .await?;
    }
    let mut candidates: Vec<_> = completed.into_values().flatten().collect();
    if coverage.grouping_batches < 2 {
        return Ok(candidates);
    }
    candidates.sort_by(|a, b| (&a.check_id, a.kind).cmp(&(&b.check_id, b.kind)));
    let tracker = Tracker::start(
        client,
        "reconcile".into(),
        wire::ActivityPhase::Reconcile,
        "Compare candidate patterns".into(),
        candidates
            .iter()
            .flat_map(|c| c.execution_ids.iter().cloned())
            .collect(),
    )
    .await?;
    let result = reconcile_candidates(client, candidates).await;
    tracker.finish().await?;
    result
}

struct Finding {
    draft: wire::FindingDraft,
    saved: Option<wire::Finding>,
}

pub async fn consolidate(
    client: &JobClient,
    drafts: Vec<wire::FindingDraft>,
    prior: &[wire::Finding],
) -> Result<Vec<wire::FindingDraft>, Error> {
    if drafts.is_empty() || (drafts.len() == 1 && prior.is_empty()) {
        return Ok(drafts);
    }
    let mut findings: BTreeMap<String, Finding> = drafts
        .into_iter()
        .enumerate()
        .map(|(i, draft)| (format!("new:{i}"), Finding { draft, saved: None }))
        .collect();
    let properties = model::schema("FindingDraft")?["properties"]
        .as_object()
        .ok_or(Error::InvalidRequest)?
        .clone();
    for saved in prior {
        let mut value = serde_json::to_value(saved)?;
        value
            .as_object_mut()
            .ok_or(Error::InvalidRequest)?
            .retain(|key, _| properties.contains_key(key));
        findings.insert(
            format!("saved:{}", saved.id),
            Finding {
                draft: serde_json::from_value(value)?,
                saved: Some(saved.clone()),
            },
        );
    }
    let request = model::request(
        wire::ModelRequestPurpose::Cluster,
        json!({
            "task": include_str!("../prompts/consolidate.md"), "response_schema": model::schema("FindingGroups")?,
            "findings": findings.iter().map(|(reference, f)| json!({"reference": reference, "title": f.draft.title, "description": f.draft.description, "brief": f.draft.brief, "kind": f.draft.kind, "checks": std::iter::once(&f.draft.check_id).chain(&f.draft.check_ids).collect::<BTreeSet<_>>(), "suggestion": f.draft.suggestion, "feedback": f.saved.as_ref().map(|s| json!({"status": s.status, "reason": s.reason})) })).collect::<Vec<_>>(),
        }),
    )?;
    let (response, _) = model::structured::<wire::FindingGroups>(client, request, "FindingGroups", |response| {
        let members: Vec<_> = response.groups.iter().flat_map(|g| &g.members).collect();
        if members.len() != findings.len() || members.iter().copied().collect::<BTreeSet<_>>() != findings.keys().collect() { return Some("Partition every input reference exactly once without inventing or omitting references".into()); }
        for group in &response.groups {
            if !group.members.contains(&group.representative) { return Some("Each representative must be a member of its group".into()); }
            if group.members.iter().map(|id| findings[id].draft.kind).collect::<BTreeSet<_>>().len() != 1 { return Some("Keep issues and positive patterns separate".into()); }
            if group.members.iter().filter_map(|id| findings[id].saved.as_ref()).map(|s| (s.status, &s.reason)).collect::<BTreeSet<_>>().len() > 1 { return Some("Keep saved findings with conflicting user feedback separate".into()); }
        }
        None
    }).await?;
    let mut merged = Vec::new();
    for group in response.groups {
        let incoming: Vec<_> = group
            .members
            .iter()
            .filter(|id| id.starts_with("new:"))
            .map(|id| &findings[id].draft)
            .collect();
        let Some(first) = incoming.first() else {
            continue;
        };
        let mut saved: Vec<_> = group
            .members
            .iter()
            .filter_map(|id| findings[id].saved.as_ref())
            .collect();
        saved.sort_by(|a, b| (&a.first_seen, &a.id).cmp(&(&b.first_seen, &b.id)));
        let mut presentation = findings[&group.representative].draft.clone();
        presentation.existing_finding_id = saved.first().map(|f| f.id.clone());
        presentation.merged_finding_ids = saved.iter().skip(1).map(|f| f.id.clone()).collect();
        presentation.check_id = first.check_id.clone();
        presentation.check_ids = incoming
            .iter()
            .flat_map(|f| std::iter::once(f.check_id.clone()).chain(f.check_ids.clone()))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut seen = BTreeSet::new();
        presentation.evidence = incoming
            .iter()
            .flat_map(|f| f.evidence.iter().cloned())
            .filter(|q| {
                seen.insert((
                    q.execution_id.clone(),
                    q.span_id.clone(),
                    q.quote.to_string(),
                    q.role,
                ))
            })
            .collect();
        merged.push(presentation);
    }
    Ok(merged)
}
