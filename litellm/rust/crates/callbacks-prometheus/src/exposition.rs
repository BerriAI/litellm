//! The store as `prometheus_client.generate_latest` writes a registry in the text format, version
//! 0.0.4: each family in the order it was registered, its samples, then its `_created` samples as
//! their own gauge.

use std::fmt::Write;

use crate::{
    pyfmt::float_to_go_string,
    store::{FamilySnapshot, Kind, Series, Store},
};

/// The content type `prometheus_client` answers a scrape in this format with.
pub const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// What a scrape asks of the store.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Scrape {
    /// The sample names asked for by `name[]`, every family when empty, as
    /// `registry.restricted_registry` keeps a family any of whose sample names is asked for.
    pub names: Vec<String>,
    /// Whether `_created` samples are written, as `PROMETHEUS_DISABLE_CREATED_SERIES` leaves them.
    pub created: bool,
}

/// Whether `prometheus_client` writes `_created` samples under `variable`, the value of
/// `PROMETHEUS_DISABLE_CREATED_SERIES`.
pub fn created_series(variable: Option<&str>) -> bool {
    !matches!(variable, Some("True" | "true" | "1"))
}

struct Sample<'a> {
    name: String,
    labels: Vec<(&'a str, &'a str)>,
    extra: Option<(&'static str, String)>,
    value: f64,
}

fn help(text: &str) -> String {
    text.replace('\\', r"\\").replace('\n', r"\n")
}

fn label_value(text: &str) -> String {
    text.replace('\\', r"\\")
        .replace('\n', r"\n")
        .replace('"', r#"\""#)
}

fn line(out: &mut String, sample: &Sample<'_>) {
    let mut labels: Vec<(&str, String)> = sample
        .labels
        .iter()
        .map(|(name, value)| (*name, label_value(value)))
        .collect();
    if let Some((name, value)) = &sample.extra {
        labels.push((name, value.clone()));
    }
    labels.sort_by(|left, right| left.0.cmp(right.0));
    let _ = write!(out, "{}", sample.name);
    if !labels.is_empty() {
        let joined: Vec<String> = labels
            .iter()
            .map(|(name, value)| format!("{name}=\"{value}\""))
            .collect();
        let _ = write!(out, "{{{}}}", joined.join(","));
    }
    let _ = writeln!(out, " {}", float_to_go_string(sample.value));
}

fn samples<'a>(
    family: &'a FamilySnapshot,
    values: &'a [String],
    series: &'a Series,
) -> Vec<Sample<'a>> {
    let labels: Vec<(&str, &str)> = family
        .labelnames
        .iter()
        .map(String::as_str)
        .zip(values.iter().map(String::as_str))
        .collect();
    let sample = |suffix: &str, extra: Option<(&'static str, String)>, value: f64| Sample {
        name: format!("{}{suffix}", family.name),
        labels: labels.clone(),
        extra,
        value,
    };
    match family.kind {
        Kind::Counter => vec![sample("_total", None, series.value)],
        Kind::Gauge => vec![sample("", None, series.value)],
        Kind::Histogram => {
            let mut cumulative = 0.0;
            let mut written: Vec<Sample<'a>> = family
                .buckets
                .iter()
                .zip(&series.buckets)
                .map(|(bound, count)| {
                    cumulative += count;
                    sample(
                        "_bucket",
                        Some(("le", float_to_go_string(*bound))),
                        cumulative,
                    )
                })
                .collect();
            written.push(sample("_count", None, cumulative));
            if family.buckets.first().is_some_and(|first| *first >= 0.0) {
                written.push(sample("_sum", None, series.sum));
            }
            written
        }
    }
}

fn created_samples<'a>(family: &'a FamilySnapshot) -> Vec<Sample<'a>> {
    if family.kind == Kind::Gauge {
        return Vec::new();
    }
    family
        .series
        .iter()
        .map(|(values, series)| Sample {
            name: format!("{}_created", family.name),
            labels: family
                .labelnames
                .iter()
                .map(String::as_str)
                .zip(values.iter().map(String::as_str))
                .collect(),
            extra: None,
            value: series.created,
        })
        .collect()
}

fn family_text(out: &mut String, family: &FamilySnapshot, scrape: &Scrape) {
    let asked =
        |sample: &Sample<'_>| scrape.names.is_empty() || scrape.names.contains(&sample.name);
    let written: Vec<Sample<'_>> = family
        .series
        .iter()
        .flat_map(|(values, series)| samples(family, values, series))
        .filter(asked)
        .collect();
    let created: Vec<Sample<'_>> = if scrape.created {
        created_samples(family).into_iter().filter(asked).collect()
    } else {
        Vec::new()
    };
    if !scrape.names.is_empty() && written.is_empty() && created.is_empty() {
        return;
    }
    let shown = match family.kind {
        Kind::Counter => format!("{}_total", family.name),
        Kind::Gauge | Kind::Histogram => family.name.clone(),
    };
    let kind: &'static str = family.kind.into();
    let documentation = help(&family.documentation);
    let _ = writeln!(out, "# HELP {shown} {documentation}");
    let _ = writeln!(out, "# TYPE {shown} {kind}");
    for sample in &written {
        line(out, sample);
    }
    if !created.is_empty() {
        let name = format!("{}_created", family.name);
        let _ = writeln!(out, "# HELP {name} {documentation}");
        let _ = writeln!(out, "# TYPE {name} gauge");
        for sample in &created {
            line(out, sample);
        }
    }
}

/// The text a scrape of `store` answers with.
pub fn render(store: &Store, scrape: &Scrape) -> String {
    let mut out = String::new();
    for family in store.snapshots() {
        family_text(&mut out, &family, scrape);
    }
    out
}
