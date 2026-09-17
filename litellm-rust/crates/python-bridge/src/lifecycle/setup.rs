use litellm_core::call_lifecycle::CallbackId;
use litellm_core::call_lifecycle::registration::{
    Candidate, DynamicSuccessSlot, Entry, NamedEvent, Registration, RegistrationFacts, Registry,
    RegistryMutation, classify_dynamic_success, plan_registration,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

use super::bindings::PythonLogger;

const SETUP_MODULE: &str = "litellm.rust_bridge.setup";

struct Targets<'py> {
    objects: Vec<Bound<'py, PyAny>>,
}

impl<'py> Targets<'py> {
    fn new() -> Self {
        Self {
            objects: Vec::new(),
        }
    }

    fn intern(&mut self, object: &Bound<'py, PyAny>) -> CallbackId {
        if let Some(index) = self
            .objects
            .iter()
            .position(|existing| existing.is(object) || existing.eq(object).unwrap_or(false))
        {
            return CallbackId(index as u64);
        }
        self.objects.push(object.clone());
        CallbackId((self.objects.len() - 1) as u64)
    }

    fn get(&self, id: CallbackId) -> PyResult<&Bound<'py, PyAny>> {
        self.objects
            .get(id.0 as usize)
            .ok_or_else(super::missing_state)
    }
}

fn registration<'py>(
    setup: &Bound<'py, PyModule>,
    object: &Bound<'py, PyAny>,
) -> PyResult<Registration> {
    if let Ok(name) = object.cast::<PyString>() {
        let name = name.to_str()?;
        let known = setup.getattr("is_known_name")?.call1((name,))?.extract()?;
        return Ok(Registration::Named {
            known,
            async_only: matches!(name, "dynamodb" | "openmeter"),
        });
    }
    let asynchronous = setup
        .getattr("is_async_callable")?
        .call1((object,))?
        .extract()?;
    Ok(Registration::Object { asynchronous })
}

fn entries<'py>(
    setup: &Bound<'py, PyModule>,
    targets: &mut Targets<'py>,
    list: &Bound<'py, PyAny>,
) -> PyResult<Vec<Entry>> {
    list.try_iter()?
        .map(|object| {
            let object = object?;
            Ok(Entry {
                id: targets.intern(&object),
                registration: registration(setup, &object)?,
            })
        })
        .collect()
}

fn registry_name(registry: Registry) -> &'static str {
    match registry {
        Registry::Input => "input",
        Registry::AsyncInput => "async_input",
        Registry::Success => "success",
        Registry::AsyncSuccess => "async_success",
        Registry::Failure => "failure",
        Registry::AsyncFailure => "async_failure",
    }
}

fn read_registry<'py>(setup: &Bound<'py, PyModule>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    setup.getattr("registry")?.call1((name,))
}

fn read_candidates<'py>(
    setup: &Bound<'py, PyModule>,
    targets: &mut Targets<'py>,
    dynamic: Option<Bound<'py, PyAny>>,
) -> PyResult<Vec<Candidate>> {
    let mut candidates = Vec::new();
    let global = read_registry(setup, "callbacks")?;
    let sources = std::iter::once(global).chain(dynamic);
    for source in sources {
        for object in source.try_iter()? {
            let object = object?;
            let candidate = if object.is_instance_of::<PyString>() {
                let resolved = setup
                    .getattr("resolve_named_integration")?
                    .call1((&object,))?;
                if resolved.is_none() {
                    Candidate {
                        resolved: None,
                        duplicate_type: false,
                    }
                } else {
                    let duplicate_type = setup
                        .getattr("async_success_registry_has_type")?
                        .call1((&resolved,))?
                        .extract()?;
                    Candidate {
                        resolved: Some(Entry {
                            id: targets.intern(&resolved),
                            registration: registration(setup, &resolved)?,
                        }),
                        duplicate_type,
                    }
                }
            } else {
                Candidate {
                    resolved: Some(Entry {
                        id: targets.intern(&object),
                        registration: registration(setup, &object)?,
                    }),
                    duplicate_type: false,
                }
            };
            candidates.push(candidate);
        }
    }
    Ok(candidates)
}

fn apply<'py>(
    setup: &Bound<'py, PyModule>,
    targets: &Targets<'py>,
    mutations: &[RegistryMutation],
    function_id: Option<&Bound<'py, PyAny>>,
) -> PyResult<()> {
    for mutation in mutations {
        match mutation {
            RegistryMutation::Append(registry, id) => {
                setup
                    .getattr("append_registry")?
                    .call1((registry_name(*registry), targets.get(*id)?))?;
            }
            RegistryMutation::Remove(registry, id) => {
                setup
                    .getattr("remove_registry")?
                    .call1((registry_name(*registry), targets.get(*id)?))?;
            }
            RegistryMutation::ExpandNamed(event, id) => {
                let event = match event {
                    NamedEvent::Success => "success",
                    NamedEvent::Failure => "failure",
                };
                setup
                    .getattr("expand_named")?
                    .call1((targets.get(*id)?, event))?;
            }
            RegistryMutation::Bootstrap => {
                setup.getattr("bootstrap")?.call1((function_id,))?;
            }
        }
    }
    Ok(())
}

struct DynamicLists<'py> {
    success: Option<Bound<'py, PyList>>,
    async_success: Option<Bound<'py, PyList>>,
    failure: Option<Bound<'py, PyList>>,
}

fn split_dynamic<'py>(
    py: Python<'py>,
    setup: &Bound<'py, PyModule>,
    targets: &mut Targets<'py>,
    kwargs: &Bound<'py, PyDict>,
) -> PyResult<DynamicLists<'py>> {
    let success = match kwargs.get_item("success_callback")? {
        Some(value) if value.is_instance_of::<PyList>() => {
            let list = value.cast_into::<PyList>()?;
            let sync = PyList::empty(py);
            let asynchronous = PyList::empty(py);
            for object in list.iter() {
                let entry = Entry {
                    id: targets.intern(&object),
                    registration: registration(setup, &object)?,
                };
                let named_async = object
                    .cast::<PyString>()
                    .ok()
                    .and_then(|name| {
                        name.to_str()
                            .ok()
                            .map(|name| matches!(name, "dynamodb" | "s3"))
                    })
                    .unwrap_or(false);
                match classify_dynamic_success(entry, named_async) {
                    DynamicSuccessSlot::Sync => sync.append(&object)?,
                    DynamicSuccessSlot::Async => asynchronous.append(&object)?,
                }
            }
            kwargs.del_item("success_callback")?;
            Some((sync, (!asynchronous.is_empty()).then_some(asynchronous)))
        }
        _ => None,
    };
    let failure = match kwargs.get_item("failure_callback")? {
        Some(value) if value.is_instance_of::<PyList>() => {
            kwargs.del_item("failure_callback")?;
            Some(value.cast_into::<PyList>()?)
        }
        _ => None,
    };
    let (success, async_success) = match success {
        Some((sync, asynchronous)) => (Some(sync), asynchronous),
        None => (None, None),
    };
    Ok(DynamicLists {
        success,
        async_success,
        failure,
    })
}

pub(super) struct Setup {
    pub logger: PythonLogger,
    pub kwargs: Py<PyDict>,
    pub supplied: bool,
}

pub(super) fn setup(
    py: Python<'_>,
    call_type: &str,
    args: &Py<PyTuple>,
    kwargs: &Py<PyDict>,
    start: &Py<PyAny>,
    asynchronous: bool,
) -> PyResult<Setup> {
    let setup = py.import(SETUP_MODULE)?;
    let kwargs = kwargs.bind(py).copy()?;
    if !kwargs.contains("litellm_call_id")? {
        let call_id = py.import("uuid")?.call_method0("uuid4")?.str()?;
        kwargs.set_item("litellm_call_id", call_id)?;
    }
    if let Some(supplied) = kwargs.get_item("litellm_logging_obj")? {
        let logging_class = py
            .import("litellm.litellm_core_utils.litellm_logging")?
            .getattr("Logging")?;
        if supplied.is_instance(&logging_class)? {
            return Ok(Setup {
                logger: supplied.extract()?,
                kwargs: kwargs.unbind(),
                supplied: true,
            });
        }
    }

    setup.getattr("prepare_environment")?.call0()?;
    let guardrails = setup.getattr("applied_guardrails")?.call1((&kwargs,))?;
    let function_id = kwargs.get_item("id")?;

    let mut targets = Targets::new();
    let dynamic = match kwargs.get_item("callbacks")? {
        Some(value) => {
            kwargs.del_item("callbacks")?;
            (!value.is_none()).then_some(value)
        }
        None => None,
    };
    let candidates = read_candidates(&setup, &mut targets, dynamic)?;
    let facts = RegistrationFacts {
        candidates,
        input: entries(&setup, &mut targets, &read_registry(&setup, "input")?)?,
        success: entries(&setup, &mut targets, &read_registry(&setup, "success")?)?,
        failure: entries(&setup, &mut targets, &read_registry(&setup, "failure")?)?,
        async_success: entries(
            &setup,
            &mut targets,
            &read_registry(&setup, "async_success")?,
        )?
        .into_iter()
        .map(|entry| entry.id)
        .collect(),
        async_failure: entries(
            &setup,
            &mut targets,
            &read_registry(&setup, "async_failure")?,
        )?
        .into_iter()
        .map(|entry| entry.id)
        .collect(),
        bootstrap_pending: setup.getattr("bootstrap_pending")?.call0()?.extract()?,
    };
    apply(
        &setup,
        &targets,
        &plan_registration(&facts),
        function_id.as_ref(),
    )?;

    let dynamic = split_dynamic(py, &setup, &mut targets, &kwargs)?;
    setup.getattr("breadcrumb")?.call1((&kwargs,))?;
    if let Some(logger_fn) = kwargs.get_item("logger_fn")? {
        setup.getattr("logger_fn")?.call1((logger_fn,))?;
    }
    let model = match args.bind(py).get_item(0) {
        Ok(model) => Some(model),
        Err(_) => kwargs.get_item("model")?,
    };
    let build = setup.getattr("build_logging")?;
    let build_kwargs = PyDict::new(py);
    build_kwargs.set_item("call_type", call_type)?;
    build_kwargs.set_item("model", model)?;
    build_kwargs.set_item("kwargs", &kwargs)?;
    build_kwargs.set_item("start_time", start)?;
    build_kwargs.set_item("asynchronous", asynchronous)?;
    build_kwargs.set_item("dynamic_success", dynamic.success)?;
    build_kwargs.set_item("dynamic_async_success", dynamic.async_success)?;
    build_kwargs.set_item("dynamic_failure", dynamic.failure)?;
    build_kwargs.set_item("guardrails", guardrails)?;
    let logger = build.call((), Some(&build_kwargs))?;
    Ok(Setup {
        logger: logger.extract()?,
        kwargs: kwargs.unbind(),
        supplied: false,
    })
}
