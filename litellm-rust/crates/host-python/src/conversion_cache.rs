use std::collections::{HashMap, hash_map::Entry};

use pyo3::prelude::*;

pub struct ToPythonCache<'a, 'py, T> {
    entries: HashMap<usize, (&'a T, Bound<'py, PyAny>)>,
}

impl<T> Default for ToPythonCache<'_, '_, T> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }
}

impl<'a, 'py, T> ToPythonCache<'a, 'py, T> {
    pub fn get_or_try_insert_with(
        &mut self,
        value: &'a T,
        convert: impl FnOnce(&'a T) -> PyResult<Bound<'py, PyAny>>,
    ) -> PyResult<&Bound<'py, PyAny>> {
        let identity = std::ptr::from_ref(value) as usize;
        let entry = match self.entries.entry(identity) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert((value, convert(value)?)),
        };
        Ok(&entry.1)
    }
}

pub struct FromPythonCache<'py, T> {
    entries: HashMap<usize, (Bound<'py, PyAny>, T)>,
}

impl<T> Default for FromPythonCache<'_, T> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }
}

impl<'py, T> FromPythonCache<'py, T> {
    pub fn get_or_try_insert_with(
        &mut self,
        value: &Bound<'py, PyAny>,
        convert: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<T>,
    ) -> PyResult<&T> {
        let identity = value.as_ptr() as usize;
        let entry = match self.entries.entry(identity) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert((value.clone(), convert(value)?)),
        };
        Ok(&entry.1)
    }
}
