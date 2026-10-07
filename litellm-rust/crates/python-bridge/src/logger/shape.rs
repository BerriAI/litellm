use litellm_tracing::{PayloadShape, ShapeLimits, ShapeSource, ShapeVisitor};
use pyo3::{
    prelude::*,
    types::{PyDict, PyList, PyString, PyTuple},
};

struct PythonShape<'a, 'py> {
    value: &'a Bound<'py, PyAny>,
    model_type: &'a Bound<'py, PyAny>,
}

impl PythonShape<'_, '_> {
    fn child(
        &self,
        value: &Bound<'_, PyAny>,
        visit: impl FnOnce(&PythonShape<'_, '_>) -> bool,
    ) -> bool {
        visit(&PythonShape {
            value,
            model_type: self.model_type,
        })
    }

    fn fields(&self, fields: &Bound<'_, PyDict>, visitor: &mut ShapeVisitor<'_>) -> bool {
        fields.iter().all(|(key, child)| {
            let Ok(key) = key.cast::<PyString>() else {
                return true;
            };
            let Ok(key) = key.to_str() else {
                return false;
            };
            self.child(&child, |child| visitor.field(key, child))
        })
    }

    fn model(&self, visitor: &mut ShapeVisitor<'_>) -> PyResult<bool> {
        let fields = self.value.get_type().getattr("model_fields")?;
        let fields = fields.cast::<PyDict>()?;
        let values = self.value.getattr("__dict__")?;
        let values = values.cast::<PyDict>()?;
        for (name, field) in fields.iter() {
            let name = name.cast::<PyString>()?.to_str()?;
            if name.starts_with('_')
                || field.getattr("exclude")?.extract::<Option<bool>>()? == Some(true)
            {
                continue;
            }
            let Some(child) = values.get_item(name)? else {
                continue;
            };
            let alias = field.getattr("serialization_alias")?;
            let key = if alias.is_none() {
                name
            } else {
                alias.cast::<PyString>()?.to_str()?
            };
            if !self.child(&child, |child| visitor.field(key, child)) {
                return Ok(false);
            }
        }
        let extras = self.value.getattr("__pydantic_extra__")?;
        Ok(extras.is_none() || self.fields(extras.cast::<PyDict>()?, visitor))
    }
}

impl ShapeSource for PythonShape<'_, '_> {
    fn visit(&self, visitor: &mut ShapeVisitor<'_>) -> bool {
        if let Ok(fields) = self.value.cast::<PyDict>() {
            return self.fields(fields, visitor);
        }
        if let Ok(items) = self.value.cast::<PyList>() {
            return items
                .iter()
                .all(|child| self.child(&child, |child| visitor.item(child)));
        }
        if let Ok(items) = self.value.cast::<PyTuple>() {
            return items
                .iter()
                .all(|child| self.child(&child, |child| visitor.item(child)));
        }
        match self.value.is_instance(self.model_type) {
            Ok(true) => self.model(visitor).unwrap_or(false),
            Ok(false) => true,
            Err(_) => false,
        }
    }
}

pub(super) fn extract(value: &Bound<'_, PyAny>, limits: ShapeLimits) -> PyResult<PayloadShape> {
    let model_type = value.py().import("pydantic")?.getattr("BaseModel")?;
    Ok(PayloadShape::extract_source(
        &PythonShape {
            value,
            model_type: &model_type,
        },
        limits,
    ))
}
