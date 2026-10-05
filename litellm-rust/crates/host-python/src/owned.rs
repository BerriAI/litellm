use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
};

pub trait PythonOwned: Send + Sync {
    fn close(&mut self, py: Python<'_>);
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError>;
}
