use rand::distributions::Alphanumeric;
use rand::Rng;

use std::sync::Arc;

use pyo3::prelude::*;

use std::collections::HashSet;

use arrow::{datatypes::DataType, record_batch::RecordBatch};
use datafusion::execution::context::ExecutionContext as _ExecutionContext;
use datafusion::logical_plan::create_udf;
use datafusion::{datasource::MemTable, error::ExecutionError};

mod dataframe;
mod errors;
mod expression;
mod functions;
mod to_py;
mod to_rust;
mod types;
mod udf;

#[pyclass(unsendable)]
struct ExecutionContext {
    ctx: _ExecutionContext,
}

#[pymethods]
impl ExecutionContext {
    #[new]
    fn new() -> Self {
        ExecutionContext {
            ctx: _ExecutionContext::new(),
        }
    }

    /// Returns a DataFrame whose plan corresponds to the SQL statement.
    fn sql(&mut self, query: &str) -> PyResult<dataframe::DataFrame> {
        let df = self
            .ctx
            .sql(query)
            .map_err(|e| -> errors::DataFusionError { e.into() })?;
        Ok(dataframe::DataFrame::new(
            self.ctx.state.clone(),
            df.to_logical_plan(),
        ))
    }

    fn create_dataframe(
        &mut self,
        partitions: Vec<Vec<PyObject>>,
        py: Python,
    ) -> PyResult<dataframe::DataFrame> {
        let partitions: Vec<Vec<RecordBatch>> = partitions
            .iter()
            .map(|batches| {
                batches
                    .iter()
                    .map(|batch| to_rust::to_rust_batch(batch.as_ref(py)))
                    .collect()
            })
            .collect::<PyResult<_>>()?;

        let table = errors::wrap(MemTable::new(partitions[0][0].schema(), partitions))?;

        // generate a random (unique) name for this table
        let name = rand::thread_rng()
            .sample_iter(&Alphanumeric)
            .take(10)
            .collect::<String>();

        self.ctx.register_table(&name, Box::new(table));
        Ok(dataframe::DataFrame::new(
            self.ctx.state.clone(),
            errors::wrap(self.ctx.table(&name))?.to_logical_plan(),
        ))
    }

    fn register_parquet(&mut self, name: &str, path: &str) -> PyResult<()> {
        errors::wrap(self.ctx.register_parquet(name, path))?;
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
        let args_types: Vec<DataType> = errors::wrap(
            args_types
                .iter()
                .map(|x| types::data_type(&x))
                .collect::<Result<Vec<DataType>, ExecutionError>>(),
        )?;
        let return_type = Arc::new(errors::wrap(types::data_type(return_type))?);

        self.ctx.register_udf(create_udf(
            name.into(),
            args_types.clone(),
            return_type.clone(),
            udf::udf(func, args_types, return_type),
        ));
        Ok(())
    }

    fn register_array_udf(
        &mut self,
        name: &str,
        func: PyObject,
        args_types: Vec<&str>,
        return_type: &str,
    ) -> PyResult<()> {
        // map strings for declared types to cases from DataType
        let args_types: Vec<DataType> = errors::wrap(
            args_types
                .iter()
                .map(|x| types::data_type(&x))
                .collect::<Result<Vec<DataType>, ExecutionError>>(),
        )?;
        let return_type = Arc::new(errors::wrap(types::data_type(return_type))?);

        self.ctx.register_udf(create_udf(
            name.into(),
            args_types.clone(),
            return_type.clone(),
            udf::array_udf(func, args_types),
        ));
        Ok(())
    }

    fn tables(&self) -> HashSet<String> {
        self.ctx.tables()
    }
}

/// DataFusion.
#[pymodule]
fn datafusion(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ExecutionContext>()?;
    m.add_class::<dataframe::DataFrame>()?;
    m.add_class::<expression::Expression>()?;

    let functions = PyModule::new(py, "functions")?;
    functions::init(functions)?;
    m.add_submodule(functions)?;

    for data_type in types::DATA_TYPES {
        m.add(data_type, *data_type)?;
    }

    Ok(())
}
