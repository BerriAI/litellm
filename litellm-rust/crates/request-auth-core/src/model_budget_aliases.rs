use serde::{Deserialize, Serialize};

use crate::model_budget_data::{Snapshot, check};

#[derive(Clone, Debug, Deserialize)]
pub struct Row {
    pub key: String,
    pub names: Vec<String>,
    pub prefix_known: Option<bool>,
    pub provider_read: bool,
    pub provider: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Facts {
    pub model: String,
    pub rows: Vec<Row>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, strum::VariantArray)]
pub enum Read {
    Entry,
    Prefix,
    Provider,
}

#[derive(Clone, Debug, Serialize)]
pub struct Decision {
    pub read: Option<Read>,
    pub key: String,
    pub models: Vec<String>,
}

fn reading(read: Read, key: &str) -> Decision {
    Decision {
        read: Some(read),
        key: key.to_owned(),
        models: vec![],
    }
}

fn row<'a>(facts: &'a Facts, key: &str) -> Result<&'a Row, Decision> {
    facts
        .rows
        .iter()
        .find(|row| row.key == key)
        .ok_or_else(|| reading(Read::Entry, key))
}

fn first(row: &Row) -> Option<&str> {
    row.names
        .first()
        .filter(|name| **name != row.key)
        .map(String::as_str)
}

fn provider(row: &Row) -> Result<Option<&str>, Decision> {
    if let Some((prefix, _)) = row.key.split_once('/') {
        match row.prefix_known {
            Some(true) => return Ok(Some(prefix)),
            Some(false) => (),
            None => return Err(reading(Read::Prefix, &row.key)),
        }
    }
    if !row.provider_read {
        return Err(reading(Read::Provider, &row.key));
    }
    Ok(row.provider.as_deref())
}

fn target(facts: &Facts, key: &str, requested_provider: &str) -> Result<Option<String>, Decision> {
    let row = row(facts, key)?;
    let Some(name) = first(row) else {
        return Ok(None);
    };
    if provider(row)? != Some(requested_provider) {
        return Ok(None);
    }
    Ok(Some(
        name.strip_prefix(&format!("{requested_provider}/"))
            .unwrap_or(name)
            .to_owned(),
    ))
}

fn models(facts: &Facts) -> Result<Vec<String>, Decision> {
    let Some((provider, name)) = facts.model.split_once('/') else {
        return Ok(match first(row(facts, &facts.model)?) {
            Some(target) => vec![facts.model.clone(), target.to_owned()],
            None => vec![facts.model.clone()],
        });
    };
    let direct = target(facts, &facts.model, provider)?.filter(|name| !name.is_empty());
    let resolved = match direct {
        Some(target) => Some(target),
        None => target(facts, name, provider)?.filter(|name| !name.is_empty()),
    };
    Ok(match resolved {
        Some(target) if target != name => vec![facts.model.clone(), format!("{provider}/{target}")],
        _ => vec![facts.model.clone()],
    })
}

pub fn decide(facts: &Facts) -> Decision {
    match models(facts) {
        Ok(models) => Decision {
            read: None,
            key: facts.model.clone(),
            models,
        },
        Err(read) => read,
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct ScopeFacts {
    pub models: Vec<String>,
    pub snapshots: Vec<Snapshot>,
    pub budget_present: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ScopeDecision {
    pub read_index: Option<usize>,
    pub result_index: Option<usize>,
}

pub fn scope(facts: &ScopeFacts) -> ScopeDecision {
    if let Some(index) = facts.snapshots.iter().position(|value| check(value).denied) {
        return ScopeDecision {
            read_index: None,
            result_index: Some(index),
        };
    }
    if !facts.budget_present {
        return ScopeDecision {
            read_index: None,
            result_index: None,
        };
    }
    if facts.snapshots.len() < facts.models.len() {
        return ScopeDecision {
            read_index: Some(facts.snapshots.len()),
            result_index: None,
        };
    }
    ScopeDecision {
        read_index: None,
        result_index: facts.snapshots.len().checked_sub(1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::{fixture, rstest};

    #[fixture]
    fn alias() -> Row {
        Row {
            key: "provider/latest".into(),
            names: vec!["provider/target".into()],
            prefix_known: Some(true),
            provider_read: false,
            provider: None,
        }
    }

    #[rstest]
    #[case::catalog_first(vec![], Some(Read::Entry), "provider/latest", vec![])]
    #[case::prefix_before_metadata(vec![Row { prefix_known: None, ..alias() }], Some(Read::Prefix), "provider/latest", vec![])]
    #[case::metadata_for_unknown_provider(vec![Row { prefix_known: Some(false), ..alias() }], Some(Read::Provider), "provider/latest", vec![])]
    #[case::known_prefix_ignores_metadata(vec![alias()], None, "provider/latest", vec!["provider/latest", "provider/target"])]
    #[case::absent_qualified_reads_bare(vec![Row { names: vec![], ..alias() }], Some(Read::Entry), "latest", vec![])]
    #[case::matching_metadata(vec![Row { prefix_known: Some(false), provider_read: true, provider: Some("provider".into()), ..alias() }], None, "provider/latest", vec!["provider/latest", "provider/target"])]
    #[case::wrong_provider_reads_bare(vec![Row { prefix_known: Some(false), provider_read: true, provider: Some("other".into()), ..alias() }], Some(Read::Entry), "latest", vec![])]
    fn lazy_alias_reads(
        #[case] rows: Vec<Row>,
        #[case] read: Option<Read>,
        #[case] key: &str,
        #[case] models: Vec<&str>,
    ) {
        let result = decide(&Facts {
            model: "provider/latest".into(),
            rows,
        });
        assert_eq!(
            (result.read, result.key.as_str(), result.models),
            (read, key, models.into_iter().map(str::to_owned).collect())
        );
    }

    #[rstest]
    #[case::bare_alias("latest", vec!["target"], vec!["latest", "target"])]
    #[case::self("latest", vec!["latest", "other"], vec!["latest"])]
    #[case::empty_row("latest", vec![], vec!["latest"])]
    #[case::empty_bare_target("latest", vec![""], vec!["latest", ""])]
    fn bare_alias(#[case] model: &str, #[case] names: Vec<&str>, #[case] expected: Vec<&str>) {
        let result = decide(&Facts {
            model: model.into(),
            rows: vec![Row {
                key: model.into(),
                names: names.into_iter().map(str::to_owned).collect(),
                ..alias()
            }],
        });
        assert_eq!(result.read, None);
        assert_eq!(result.models, expected);
    }

    fn snapshot(limit: f64, spend: f64) -> Snapshot {
        Snapshot {
            max_budget: Some(limit),
            local_spend: Some(spend),
            legacy_spend: None,
            shared_spend: None,
            shared_started_at: None,
            shared_window_seconds: None,
            now: 0.0,
        }
    }

    #[rstest]
    #[case::first(vec![], true, Some(0), None)]
    #[case::target_after_allowed(vec![snapshot(1.0, 0.0)], true, Some(1), None)]
    #[case::alias_denial(vec![snapshot(1.0, 1.0)], true, None, Some(0))]
    #[case::target_denial(vec![snapshot(1.0, 0.0), snapshot(2.0, 3.0)], true, None, Some(1))]
    #[case::all_allowed(vec![snapshot(1.0, 0.0), snapshot(2.0, 0.0)], true, None, Some(1))]
    #[case::no_budgets(vec![], false, None, None)]
    #[case::retain_denial(vec![snapshot(1.0, 1.0)], false, None, Some(0))]
    #[case::nan_continues(vec![snapshot(f64::NAN, 1.0)], true, Some(1), None)]
    fn scope_order(
        #[case] snapshots: Vec<Snapshot>,
        #[case] budget_present: bool,
        #[case] read_index: Option<usize>,
        #[case] result_index: Option<usize>,
    ) {
        assert_eq!(
            scope(&ScopeFacts {
                models: vec!["alias".into(), "target".into()],
                snapshots,
                budget_present
            }),
            ScopeDecision {
                read_index,
                result_index
            }
        );
    }
}
