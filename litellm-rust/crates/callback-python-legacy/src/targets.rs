use litellm_callback_protocol::CallbackId;
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyString;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetShape {
    StructuredLogger,
    Callable { internal: bool },
    Named { known: bool },
    Opaque,
}

pub struct Targets {
    objects: Vec<Py<PyAny>>,
    shapes: Vec<TargetShape>,
}

impl Targets {
    pub fn read(
        py: Python<'_>,
        lists: &[Bound<'_, PyAny>],
    ) -> PyResult<(Self, Vec<Vec<CallbackId>>)> {
        let custom_logger = py
            .import("litellm.integrations.custom_logger")?
            .getattr("CustomLogger")?;
        let known = py
            .import("litellm")?
            .getattr("_known_custom_logger_compatible_callbacks")?;
        let mut targets = Self {
            objects: Vec::new(),
            shapes: Vec::new(),
        };
        let mut ids = Vec::with_capacity(lists.len());
        for list in lists {
            let mut family = Vec::new();
            for object in list.try_iter()? {
                family.push(targets.intern(py, &object?, &custom_logger, &known)?);
            }
            ids.push(family);
        }
        Ok((targets, ids))
    }

    pub fn shapes(&self, ids: &[CallbackId]) -> Vec<TargetShape> {
        ids.iter().map(|id| self.shapes[id.0 as usize]).collect()
    }

    pub fn object<'py>(&self, py: Python<'py>, id: CallbackId) -> &Bound<'py, PyAny> {
        self.objects[id.0 as usize].bind(py)
    }

    pub fn shape(&self, id: CallbackId) -> TargetShape {
        self.shapes[id.0 as usize]
    }

    pub fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        for object in &self.objects {
            visit.call(object)?;
        }
        Ok(())
    }

    fn intern(
        &mut self,
        py: Python<'_>,
        object: &Bound<'_, PyAny>,
        custom_logger: &Bound<'_, PyAny>,
        known: &Bound<'_, PyAny>,
    ) -> PyResult<CallbackId> {
        for (index, existing) in self.objects.iter().enumerate() {
            let existing = existing.bind(py);
            if existing.is(object) || existing.eq(object)? {
                return Ok(CallbackId(index as u64));
            }
        }
        let shape = if object.is_instance(custom_logger)? {
            TargetShape::StructuredLogger
        } else if let Ok(name) = object.cast::<PyString>() {
            TargetShape::Named {
                known: known.contains(name)?,
            }
        } else if object.is_callable() {
            TargetShape::Callable {
                internal: internal_callable(object)?,
            }
        } else {
            TargetShape::Opaque
        };
        self.objects.push(object.clone().unbind());
        self.shapes.push(shape);
        Ok(CallbackId((self.objects.len() - 1) as u64))
    }
}

fn internal_callable(object: &Bound<'_, PyAny>) -> PyResult<bool> {
    let name = if let Ok(name) = object.getattr("__name__") {
        name.extract::<String>()?
    } else if let Ok(function) = object.getattr("__func__") {
        function.getattr("__name__")?.extract::<String>()?
    } else {
        object.get_type().name()?.to_string()
    };
    Ok([
        "_PROXY",
        "_service_logger.ServiceLogging",
        "sync_deployment_callback_on_success",
    ]
    .iter()
    .any(|prefix| name.contains(prefix)))
}
