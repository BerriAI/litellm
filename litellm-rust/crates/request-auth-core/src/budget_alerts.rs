use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Failure {
    pub kind: String,
    pub message: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ThresholdFacts {
    pub key: String,
    pub digit: bool,
    pub digits: Option<String>,
    pub ratio: Option<f64>,
    pub integer_failure: Option<Failure>,
    pub ratio_failure: Option<Failure>,
    pub text: Option<String>,
    pub items: Option<Vec<Option<String>>>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ConfigFacts {
    pub global_entries: Vec<ThresholdFacts>,
    pub local_entries: Vec<ThresholdFacts>,
    pub global_failure: Option<Failure>,
    pub local_failure: Option<Failure>,
    pub order: Option<Vec<String>>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Facts {
    pub member: bool,
    pub spend: Option<f64>,
    pub limit: Option<f64>,
    pub default_fraction: f64,
    pub config: Option<ConfigFacts>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Threshold {
    pub key: String,
    pub recipients: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct Decision {
    pub read_config: bool,
    pub read_order: bool,
    pub global_keys: Vec<String>,
    pub local_keys: Vec<String>,
    pub emit: bool,
    pub thresholds: Option<Vec<Threshold>>,
    pub failure: Option<Failure>,
}

fn member_entry(entry: &ThresholdFacts) -> Result<bool, Failure> {
    if !entry.digit || entry.key.chars().count() > 3 {
        return Ok(false);
    }
    if let Some(error) = &entry.integer_failure {
        return Err(error.clone());
    }
    Ok(entry
        .digits
        .as_deref()
        .and_then(|digits| digits.parse::<u32>().ok())
        .is_some_and(|n| (1..=100).contains(&n)))
}

fn eligible(entries: &[ThresholdFacts], member: bool) -> Result<Vec<&ThresholdFacts>, Failure> {
    entries.iter().try_fold(Vec::new(), |mut selected, entry| {
        if !member || member_entry(entry)? {
            selected.push(entry);
        }
        Ok(selected)
    })
}

fn recipients(entry: &ThresholdFacts) -> Vec<String> {
    let strings: Vec<&str> = if let Some(text) = &entry.text {
        text.split(',').collect()
    } else {
        entry
            .items
            .as_deref()
            .unwrap_or_default()
            .iter()
            .filter_map(Option::as_deref)
            .collect()
    };
    strings
        .into_iter()
        .map(|s| s.trim_matches(|c: char| c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')))
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

fn merge(key: &str, global: &[&ThresholdFacts], local: &[&ThresholdFacts]) -> Threshold {
    let recipients = global
        .iter()
        .chain(local)
        .filter(|entry| entry.key == key)
        .flat_map(|entry| recipients(entry))
        .fold(Vec::new(), |mut merged, email| {
            if !merged.contains(&email) {
                merged.push(email);
            }
            merged
        });
    Threshold {
        key: key.to_owned(),
        recipients,
    }
}

fn evaluate(facts: &Facts) -> Result<Decision, Failure> {
    if !facts.member && !(facts.limit.is_some() && facts.spend.is_some_and(|spend| spend > 0.0)) {
        return Ok(Decision::default());
    }
    let Some(config) = &facts.config else {
        return Ok(Decision {
            read_config: true,
            ..Decision::default()
        });
    };
    if let Some(failure) = &config.global_failure {
        return Err(failure.clone());
    }
    if !facts.member
        && let Some(failure) = &config.local_failure
    {
        return Err(failure.clone());
    }
    let global = eligible(&config.global_entries, false)?;
    let local = if facts.member && config.local_failure.is_some() {
        vec![]
    } else {
        eligible(&config.local_entries, facts.member)?
    };
    if global.is_empty() && local.is_empty() {
        let emit = !facts.member
            && matches!((facts.spend, facts.limit), (Some(spend), Some(limit)) if spend >= limit * facts.default_fraction && spend < limit);
        return Ok(Decision {
            emit,
            ..Decision::default()
        });
    }
    let Some(order) = &config.order else {
        return Ok(Decision {
            read_order: true,
            global_keys: global.iter().map(|entry| entry.key.clone()).collect(),
            local_keys: local.iter().map(|entry| entry.key.clone()).collect(),
            ..Decision::default()
        });
    };
    let thresholds: Vec<Threshold> = order
        .iter()
        .map(|key| merge(key, &global, &local))
        .collect();
    if facts.member && facts.spend.is_some_and(|spend| spend <= 0.0) {
        return Ok(Decision::default());
    }
    let mut minimum: Option<&ThresholdFacts> = None;
    for key in order {
        let Some(entry) = global.iter().chain(&local).find(|entry| entry.key == *key) else {
            continue;
        };
        if !entry.digit {
            continue;
        }
        if let Some(failure) = &entry.integer_failure {
            return Err(failure.clone());
        }
        let Some(digits) = entry.digits.as_ref() else {
            continue;
        };
        if minimum
            .and_then(|entry| entry.digits.as_ref())
            .is_none_or(|current| (digits.len(), digits) < (current.len(), current))
        {
            minimum = Some(entry);
        }
    }
    let Some(minimum) = minimum else {
        return Ok(Decision::default());
    };
    if let Some(failure) = &minimum.ratio_failure {
        return Err(failure.clone());
    }
    let emit = matches!((facts.spend, facts.limit, minimum.ratio), (Some(spend), Some(limit), Some(ratio)) if spend.partial_cmp(&(limit * ratio)) != Some(std::cmp::Ordering::Less));
    Ok(Decision {
        emit,
        thresholds: Some(thresholds),
        ..Decision::default()
    })
}

pub fn plan(facts: &Facts) -> Decision {
    evaluate(facts).unwrap_or_else(|failure| Decision {
        failure: Some(failure),
        ..Decision::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::before(7.9, false)]
    #[case::at(8.0, true)]
    #[case::at_limit(10.0, false)]
    #[case::nan(f64::NAN, false)]
    #[case::zero(0.0, false)]
    fn default_key_threshold(#[case] spend: f64, #[case] emit: bool) {
        let facts = Facts {
            member: false,
            spend: Some(spend),
            limit: Some(10.0),
            default_fraction: 0.8,
            config: Some(ConfigFacts {
                global_entries: vec![],
                local_entries: vec![],
                global_failure: None,
                local_failure: None,
                order: None,
            }),
        };
        assert_eq!(plan(&facts).emit, emit);
    }
}
