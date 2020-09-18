use core::fmt;

use datafusion::error::ExecutionError;
use pyo3::{exceptions, PyErr};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DataFusionError {
    ExecutionError(ExecutionError),
}

impl fmt::Display for DataFusionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DataFusionError::ExecutionError(e) => write!(f, "THIS {:?}", e),
        }
    }
}

impl From<DataFusionError> for PyErr {
    fn from(err: DataFusionError) -> PyErr {
        exceptions::Exception::py_err(err.to_string())
    }
}

impl From<ExecutionError> for DataFusionError {
    fn from(err: ExecutionError) -> DataFusionError {
        DataFusionError::ExecutionError(err)
    }
}
