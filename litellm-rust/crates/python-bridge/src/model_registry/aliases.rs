use std::collections::HashMap;

use litellm_model_catalog::{AliasClaims, Unclaimed};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

/// A skipped alias, carrying the objects main's warning formats
#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) enum AliasWarning {
    NotAList { model: Py<PyAny>, type_name: String },
    ShadowsModel { alias: Py<PyAny>, model: Py<PyAny> },
    Taken { alias: Py<PyAny>, model: Py<PyAny> },
}

/// Expands every row's `aliases` into top-level keys sharing one `entry_type` row, as main's `_expand_model_aliases`.
/// Returns the warnings in main's order and the error main raises mid-scan, in which case `model_cost` is untouched
#[pyfunction]
pub(crate) fn expand_model_aliases<'py>(
    model_cost: &Bound<'py, PyDict>,
    entry_type: &Bound<'py, PyAny>,
) -> PyResult<(Vec<AliasWarning>, Option<PyErr>)> {
    let scan = match scan(model_cost) {
        Ok(scan) => scan,
        Err(Failed { warnings, error }) => return Ok((warnings, Some(error))),
    };
    apply(model_cost, entry_type, &scan)?;
    Ok((scan.warnings, None))
}

struct Scan<'py> {
    warnings: Vec<AliasWarning>,
    with_aliases: Vec<Bound<'py, PyAny>>,
    owners: Vec<Bound<'py, PyAny>>,
    /// Claimed aliases by their class under Python equality, so `1`, `1.0` and `True` collide as they do in a dict
    aliases: HashMap<usize, Bound<'py, PyAny>>,
    claims: AliasClaims<usize, usize>,
}

struct Failed {
    warnings: Vec<AliasWarning>,
    error: PyErr,
}

fn scan<'py>(model_cost: &Bound<'py, PyDict>) -> Result<Scan<'py>, Failed> {
    let py = model_cost.py();
    let classes = PyDict::new(py);
    let mut scan = Scan {
        warnings: Vec::new(),
        with_aliases: Vec::new(),
        owners: Vec::new(),
        aliases: HashMap::new(),
        claims: AliasClaims::new(),
    };
    for (name, info) in model_cost.iter() {
        if let Err(error) = scan_row(model_cost, &classes, &mut scan, name, &info) {
            return Err(Failed {
                warnings: scan.warnings,
                error,
            });
        }
    }
    Ok(scan)
}

fn scan_row<'py>(
    model_cost: &Bound<'py, PyDict>,
    classes: &Bound<'py, PyDict>,
    scan: &mut Scan<'py>,
    name: Bound<'py, PyAny>,
    info: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let py = model_cost.py();
    let aliases = info.call_method1(intern!(py, "get"), (intern!(py, "aliases"),))?;
    if aliases.is_none() {
        return Ok(());
    }
    scan.with_aliases.push(name.clone());
    let Ok(aliases) = aliases.cast::<PyList>() else {
        scan.warnings.push(AliasWarning::NotAList {
            model: name.unbind(),
            type_name: aliases.get_type().name()?.to_string(),
        });
        return Ok(());
    };
    let owner = scan.owners.len();
    scan.owners.push(name.clone());
    for alias in aliases.iter() {
        let is_model = model_cost.contains(&alias)?;
        let class = class_of(classes, &alias)?;
        match scan.claims.claim(&class, &owner, is_model) {
            Ok(()) => {
                scan.aliases.insert(class, alias);
            }
            Err(Unclaimed::ShadowsModel) => scan.warnings.push(AliasWarning::ShadowsModel {
                alias: alias.unbind(),
                model: name.clone().unbind(),
            }),
            Err(Unclaimed::TakenBy(_)) => scan.warnings.push(AliasWarning::Taken {
                alias: alias.unbind(),
                model: name.clone().unbind(),
            }),
        }
    }
    Ok(())
}

fn class_of(classes: &Bound<'_, PyDict>, alias: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Some(class) = classes.get_item(alias)? {
        return class.extract();
    }
    let class = classes.len();
    classes.set_item(alias, class)?;
    Ok(class)
}

fn apply<'py>(
    model_cost: &Bound<'py, PyDict>,
    entry_type: &Bound<'py, PyAny>,
    scan: &Scan<'py>,
) -> PyResult<()> {
    let py = model_cost.py();
    for name in &scan.with_aliases {
        model_cost
            .as_any()
            .get_item(name)?
            .call_method1(intern!(py, "pop"), (intern!(py, "aliases"), py.None()))?;
    }
    for (owner, aliases) in scan.claims.groups() {
        let owner = &scan.owners[*owner];
        let names: Vec<_> = std::iter::once(owner.clone())
            .chain(aliases.into_iter().map(|class| scan.aliases[class].clone()))
            .collect();
        let names = PyTuple::new(py, names)?;
        let entry = entry_type.call1((model_cost.as_any().get_item(owner)?, &names))?;
        for name in names.iter() {
            model_cost.set_item(name, &entry)?;
        }
    }
    Ok(())
}
