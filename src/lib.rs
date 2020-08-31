use std::sync::Arc;

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::PyErr;

use std::collections::{HashMap, HashSet};

use datafusion::error::ExecutionError;
use datafusion::execution::context::ExecutionContext as _ExecutionContext;
use datafusion::execution::physical_plan::udf::ScalarFunction;

use arrow::array;
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

#[pyclass]
struct ExecutionContext {
    ctx: _ExecutionContext,
}

fn wrap<T>(a: Result<T, ExecutionError>) -> Result<T, DataStoreError> {
    return Ok(a?);
}

macro_rules! to_primitive {
    ($VALUES:ident, $ARRAY_TY:ident) => {{
        $VALUES
            .as_any()
            .downcast_ref::<array::$ARRAY_TY>()
            .ok_or_else(|| ExecutionError::General(format!("Bla.").to_owned()))?
    }};
}

macro_rules! to_native {
    ($VALUES:ident, $BUILDER:ident, $TY:ident, $SIZE_HINT:ident) => {{
        let mut builder = array::$BUILDER::new($SIZE_HINT);
        $VALUES
            .iter()
            .map(|x| x.extract::<$TY>().unwrap())
            .for_each(|x| builder.append_value(x).unwrap());
        Ok(Arc::new(builder.finish()))
    }};
}

#[pymethods]
impl ExecutionContext {
    #[new]
    fn new() -> Self {
        ExecutionContext {
            ctx: _ExecutionContext::new(),
        }
    }

    fn sql(&mut self, query: &str) -> PyResult<HashMap<String, PyObject>> {
        let batches = wrap(wrap(self.ctx.sql(query))?.collect())?;
        Ok(wrap(to_py::to_py(&batches))?)
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
        let return_type = wrap(types::data_type(return_type))?;
        let return_type1 = return_type.clone();

        // construct the argument fields. Their name does not matter. 2x because one needs to be copied to ScalarFunction
        //let fields = args_types.iter().map(|x| Field::new("n", x.clone(), true)).collect();
        let args_types1 = args_types.clone();

        self.ctx.register_udf(ScalarFunction::new(
            name.into(),
            args_types.clone(),
            return_type,
            Arc::new(
                move |args: &[array::ArrayRef]| -> Result<array::ArrayRef, ExecutionError> {
                    let capacity = args[0].len();

                    // get GIL
                    let gil = pyo3::Python::acquire_gil();
                    let py = gil.python();

                    // for each row
                    // 1. cast args to PyTuple
                    // 2. call function
                    // 3. cast the arguments back to Rust-Native

                    let mut final_values = Vec::with_capacity(capacity);
                    for i in 0..args[0].len() {
                        // 1. cast args to PyTuple
                        let mut values = Vec::with_capacity(args_types.len());
                        for arg_i in 0..args_types.len() {
                            let arg = &args[arg_i];
                            let values_i = match &args_types1[arg_i] {
                                DataType::Float64 => {
                                    Ok(to_primitive!(arg, Float64Array).value(i).to_object(py))
                                }
                                DataType::Float32 => {
                                    Ok(to_primitive!(arg, Float32Array).value(i).to_object(py))
                                }
                                DataType::Int32 => {
                                    Ok(to_primitive!(arg, Int32Array).value(i).to_object(py))
                                }
                                DataType::UInt32 => {
                                    Ok(to_primitive!(arg, UInt32Array).value(i).to_object(py))
                                }
                                other => Err(ExecutionError::NotImplemented(
                                    format!("Datatype \"{:?}\" is still not implemented.", other)
                                        .to_owned(),
                                )),
                            }?;
                            values.push(values_i);
                        }
                        let values = PyTuple::new(py, values);

                        // 2. call function
                        let any = func.as_ref(py);
                        let value = any.call(values, None);
                        let value = match value {
                            Ok(n) => Ok(n),
                            Err(data) => {
                                Err(ExecutionError::General(format!("{:?}", data).to_owned()))
                            }
                        }?;
                        final_values.push(value);
                    }

                    // 3. cast the arguments back to Rust-Native
                    match &return_type1 {
                        DataType::Float64 => {
                            to_native!(final_values, Float64Builder, f64, capacity)
                        }
                        DataType::Float32 => {
                            to_native!(final_values, Float32Builder, f32, capacity)
                        }
                        DataType::Int32 => to_native!(final_values, Int32Builder, i32, capacity),
                        DataType::Int64 => to_native!(final_values, Int64Builder, i64, capacity),
                        other => Err(ExecutionError::NotImplemented(
                            format!(
                                "Datatype \"{:?}\" is still not implemented as a return type.",
                                other
                            )
                            .to_owned(),
                        )),
                    }
                },
            ),
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
