use std::collections::HashMap;

use litellm_host_python::from_py;
use litellm_model_catalog::wire::Fields;
use litellm_model_catalog::{
    RegistrationCandidate, RegistrationCandidates, misses_cache_prices, overlay, registration_base,
    registration_target,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use serde_json::Value;

use super::CatalogHandle;
use super::py_values::{Opaque, without_opaque};
use super::resolve::LiveKeys;

const CANDIDATES_MODULE: &str = "litellm.rust_bridge.host.model_names";

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct Registered {
    /// Always the caller's keys
    #[pyo3(get)]
    written: Vec<String>,
    /// Caller key to the row it went on top of
    #[pyo3(get)]
    inherited_from: HashMap<String, Option<String>>,
    /// Keys that get main's "cache cost will default to 0" warning
    #[pyo3(get)]
    missing_cache_prices: Vec<String>,
}

/// Which layer a registration lands in
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Layer {
    /// 2, `persist=False`
    Transient,
    /// 3, `persist=True`
    Persisted,
    /// An install replaying layer 3, which records nothing new
    Replay,
}

struct Written {
    inherited_from: Option<String>,
    misses_cache_prices: bool,
}

/// The candidate names, the capability seed and the provider-info skip for one key, from the host module
struct Inheritance {
    candidates: RegistrationCandidates,
    seed: Option<Fields>,
}

fn inheritance(
    py: Python<'_>,
    key: &str,
    row_provider: Option<&str>,
    provider: Option<&str>,
) -> PyResult<Inheritance> {
    let answer = py
        .import(CANDIDATES_MODULE)?
        .call_method1("registration_candidates", (key, row_provider, provider))?;
    let answer = answer.cast::<PyTuple>()?;
    let candidates = answer
        .get_item(0)?
        .extract::<Vec<(String, Option<String>, bool, bool, bool)>>()?
        .into_iter()
        .map(
            |(name, provider, requires_provider, names_the_row, seeded)| RegistrationCandidate {
                name,
                provider,
                requires_provider,
                names_the_row,
                seeded,
            },
        )
        .collect();
    let seed = answer.get_item(1)?;
    let seed = if seed.is_none() {
        None
    } else {
        Some(from_py::<Fields>(&seed)?)
    };
    Ok(Inheritance {
        candidates: RegistrationCandidates(candidates),
        seed,
    })
}

impl CatalogHandle {
    /// One row of `register`. The caller's key is written, on top of the row it inherits from (B1)
    #[expect(
        clippy::too_many_arguments,
        reason = "one row of a register call, its opaque values and where it lands"
    )]
    fn register_one(
        &self,
        py: Python<'_>,
        dict: &Bound<'_, PyDict>,
        key: &str,
        registration: &Fields,
        opaque: &mut Opaque,
        provider: Option<&str>,
        layer: Layer,
    ) -> PyResult<Written> {
        let keys = LiveKeys {
            dict,
            index: &self.index,
        };
        let row_provider = registration.get("litellm_provider").and_then(Value::as_str);
        let generation = self.install_generation();
        let inheritance = inheritance(py, key, row_provider, provider)?;
        let created =
            dict.get_item(key)?.is_none() || self.lock().layers.registrations().created(key);
        let candidates = RegistrationCandidates(
            inheritance
                .candidates
                .0
                .iter()
                .filter(|candidate| !(created && candidate.name == key))
                .cloned()
                .collect(),
        );
        let target = registration_target(&keys, key, &candidates);
        if self.install_generation() != generation {
            return self.register_one(py, dict, key, registration, opaque, provider, layer);
        }
        let mut row_of = |name: &str| -> PyResult<Option<Fields>> {
            match dict.get_item(name)? {
                Some(row) if row.is_instance_of::<PyDict>() => opaque.fields(&row).map(Some),
                _ => Ok(None),
            }
        };
        let inherited_row = match target.inherit_from.as_deref() {
            Some(source) => row_of(source)?,
            None => None,
        };
        let base = registration_base(
            &target,
            inherited_row.as_ref(),
            inheritance.seed.as_ref(),
            registration,
        );
        let mut written = row_of(key)?.unwrap_or_default();
        written.extend(overlay(&base, registration));
        let current = dict.get_item(key)?;
        let group = self.lock().layers.alias_group(key);
        let sharing: Vec<String> = group
            .into_iter()
            .filter(|name| {
                name == key
                    || current.as_ref().is_some_and(|current| {
                        dict.get_item(name.as_str())
                            .ok()
                            .flatten()
                            .is_some_and(|row| row.is(current))
                    })
            })
            .chain([key.to_owned()])
            .collect();
        let row = opaque.to_py(py, &serde_json::Value::Object(written.clone()))?;
        for name in &sharing {
            dict.set_item(name.as_str(), row.clone_ref(py))?;
        }
        {
            let mut state = self.lock();
            if let Some(record) = state.record.as_mut() {
                for name in &sharing {
                    record.note_written(name, &written);
                }
            }
            let layers = &mut state.layers;
            if created {
                layers.registrations_mut().note_created(key);
            }
            for name in &sharing {
                layers.registrations_mut().note_fields(name, registration);
            }
            match layer {
                Layer::Transient => layers.record_transient(key, &without_opaque(registration)),
                Layer::Persisted => layers
                    .registrations_mut()
                    .record(key, &without_opaque(registration)),
                Layer::Replay => {}
            }
        }
        Ok(Written {
            inherited_from: target.inherit_from.clone(),
            misses_cache_prices: misses_cache_prices(&target, registration),
        })
    }

    pub(crate) fn register_rows(
        &self,
        py: Python<'_>,
        rows: &Bound<'_, PyDict>,
        persist: bool,
        provider: Option<&str>,
    ) -> PyResult<Registered> {
        let dict = self.bound_dict(py)?;
        let layer = if persist {
            Layer::Persisted
        } else {
            Layer::Transient
        };
        let mut registered = Registered {
            written: Vec::new(),
            inherited_from: HashMap::new(),
            missing_cache_prices: Vec::new(),
        };
        for (key, row) in rows.iter() {
            let key: String = key.str()?.extract()?;
            let mut opaque = Opaque::default();
            let registration = opaque.fields(&row)?;
            let written =
                self.register_one(py, &dict, &key, &registration, &mut opaque, provider, layer)?;
            if written.misses_cache_prices {
                registered.missing_cache_prices.push(key.clone());
            }
            registered
                .inherited_from
                .insert(key.clone(), written.inherited_from);
            registered.written.push(key);
        }
        Ok(registered)
    }

    /// After an install, main's replay order. Live router deployments first, then the persisted registrations over them
    pub(crate) fn replay_after_install(&self, py: Python<'_>) -> PyResult<()> {
        py.import(CANDIDATES_MODULE)?
            .call_method0("replay_live_deployments")?;
        let persisted: Vec<(String, Fields)> = self
            .lock()
            .layers
            .registrations()
            .replay()
            .map(|(key, fields)| (key.to_owned(), fields.clone()))
            .collect();
        let dict = self.bound_dict(py)?;
        for (key, registration) in persisted {
            let mut opaque = Opaque::default();
            self.register_one(
                py,
                &dict,
                &key,
                &registration,
                &mut opaque,
                None,
                Layer::Replay,
            )?;
        }
        Ok(())
    }

    /// The row a registration of `key` inherits from, or the capability seed when it inherits from nothing
    pub(crate) fn base_for(
        &self,
        py: Python<'_>,
        key: &str,
        row_provider: Option<&str>,
    ) -> PyResult<Fields> {
        let dict = self.bound_dict(py)?;
        let keys = LiveKeys {
            dict: &dict,
            index: &self.index,
        };
        let inheritance = inheritance(py, key, row_provider, None)?;
        let target = registration_target(&keys, key, &inheritance.candidates);
        let inherited_row = target
            .inherit_from
            .as_deref()
            .and_then(|source| keys.fields_of(source));
        let registration: Fields = row_provider
            .map(|provider| {
                (
                    "litellm_provider".to_owned(),
                    Value::String(provider.to_owned()),
                )
            })
            .into_iter()
            .collect();
        Ok(registration_base(
            &target,
            inherited_row.as_ref(),
            inheritance.seed.as_ref(),
            &registration,
        ))
    }
}
