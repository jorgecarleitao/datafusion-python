use pyo3::prelude::*;
use pyo3::PyErr;
use pyo3::exceptions;

use std::collections::HashSet;

use datafusion::execution::context::ExecutionContext as _ExecutionContext;
use datafusion::error::ExecutionError;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum DataStoreError {
    #[error(transparent)]
    ExecutionError(#[from] ExecutionError),
}

impl From<DataStoreError> for PyErr {
    fn from(err: DataStoreError) -> PyErr {
        exceptions::Exception::py_err(err.to_string())
    }
}


#[pyclass]
struct ExecutionContext {
    ctx: _ExecutionContext
}

fn wrap<T>(a: Result<T, ExecutionError>) -> Result<T, DataStoreError> {
    return Ok(a?)
}

#[pymethods]
impl ExecutionContext {
    #[new]
     fn new() -> Self {
        ExecutionContext {
            ctx: _ExecutionContext::new()
        }
    }

    fn sql(&mut self, query: &str, batch_size: usize) -> PyResult<usize> {
        Ok(wrap(self.ctx.sql(query, batch_size))?.len())
    }

    fn register_parquet(&mut self, name: &str, path: &str) -> PyResult<()> {
        wrap(self.ctx.register_parquet(name, path))?;
        Ok(())
    }

    fn tables(&self) -> HashSet<String> {
        self.ctx.tables()
    }
}

#[pymodule]
fn datafusion(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ExecutionContext>()?;

    Ok(())
}
