use std::sync::Arc;

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::PyErr;

use std::collections::HashSet;

use datafusion::error::ExecutionError;
use datafusion::execution::context::ExecutionContext as _ExecutionContext;
use datafusion::logical_plan::create_udf;

use arrow::datatypes::DataType;
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

#[pyclass(unsendable)]
struct ExecutionContext {
    ctx: _ExecutionContext,
}

fn wrap<T>(a: Result<T, ExecutionError>) -> Result<T, DataStoreError> {
    return Ok(a?);
}

#[pymethods]
impl ExecutionContext {
    #[new]
    fn new() -> Self {
        ExecutionContext {
            ctx: _ExecutionContext::new(),
        }
    }

    fn sql(&mut self, query: &str) -> PyResult<PyObject> {
        let batches = wrap(wrap(self.ctx.sql(query))?.collect())?;
        to_py::to_py(&batches)
    }

    fn register_parquet(&mut self, name: &str, path: &str) -> PyResult<()> {
        wrap(self.ctx.register_parquet(name, path))?;
        Ok(())
    }

    fn register_udf(
        &mut self,
        name: &str,
        func: PyObject,
        args_types: Vec<&str>,
        return_type: &str,
    ) -> PyResult<()> {
        // map strings for declared types to cases from DataType
        let args_types: Vec<DataType> = wrap(
            args_types
                .iter()
                .map(|x| types::data_type(&x))
                .collect::<Result<Vec<DataType>, ExecutionError>>(),
        )?;
        let return_type = Arc::new(wrap(types::data_type(return_type))?);

        self.ctx.register_udf(create_udf(
            name.into(),
            args_types.clone(),
            return_type.clone(),
            udf::udf(func, args_types, return_type),
        ));
        Ok(())
    }

    fn tables(&self) -> HashSet<String> {
        self.ctx.tables()
    }
}

#[pymodule]
fn datafusion(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ExecutionContext>()?;

    for data_type in types::DATA_TYPES {
        m.add(data_type, data_type)?;
    }

    Ok(())
}

mod to_py;
mod types;
mod udf;
