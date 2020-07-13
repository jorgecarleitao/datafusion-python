use pyo3::prelude::*;
use pyo3::PyErr;
use pyo3::exceptions;

use numpy::PyArray1;

use std::collections::{HashSet, HashMap};

use datafusion::execution::context::ExecutionContext as _ExecutionContext;
use datafusion::error::ExecutionError;

use thiserror::Error;
use arrow::record_batch::RecordBatch;
use arrow::array::Array;
use arrow::datatypes::DataType;
use arrow::array;

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

macro_rules! to_py_numpy {
    ($ARRAY:ident, $ARRAY_TY:ident) => {{
        let gil = pyo3::Python::acquire_gil();
        let casted = $ARRAY.as_any().downcast_ref::<array::$ARRAY_TY>().unwrap();
        let mut values = Vec::with_capacity(casted.len());
        for i in 0..$ARRAY.len() {
            values.push(casted.value(i));
        }
        Ok(PyObject::from(PyArray1::from_iter(gil.python(), values)))
    }};
}

fn to_py(record: &RecordBatch) -> Result<HashMap<String, PyObject>, ExecutionError> {
    let mut map: HashMap<String, PyObject> = HashMap::new();

    for column_index in 0..record.schema().fields().len() {
        let column = record.column(column_index);
        let value = match column.data_type() {
            //DataType::Null: no NullArray in arrow
            DataType::Boolean => to_py_numpy!(column, BooleanArray),
            DataType::Int8 => to_py_numpy!(column, Int8Array),
            DataType::Int16 => to_py_numpy!(column, Int16Array),
            DataType::Int32 => to_py_numpy!(column, Int32Array),
            DataType::Int64 => to_py_numpy!(column, Int64Array),
            DataType::UInt8 => to_py_numpy!(column, UInt8Array),
            DataType::UInt16 => to_py_numpy!(column, UInt16Array),
            DataType::UInt32 => to_py_numpy!(column, UInt32Array),
            DataType::UInt64 => to_py_numpy!(column, UInt64Array),
            //DataType::Float16 is not represented in rust arrow
            DataType::Float32 => to_py_numpy!(column, Float32Array),
            DataType::Float64 => to_py_numpy!(column, Float64Array),
            /*
            None of the below are currently supported by rust-numpy
            DataType::Timestamp(_, _) => {}
            DataType::Date32(_) => {}
            DataType::Date64(_) => {}
            DataType::Time32(_) => {}
            DataType::Time64(_) => {}
            DataType::Duration(_) => {}
            DataType::Interval(_) => {}
            DataType::Binary => {}
            DataType::FixedSizeBinary(_) => {}
            DataType::LargeBinary => {}
            DataType::Utf8 => {}
            DataType::LargeUtf8 => {}
            DataType::List(_) => {}
            DataType::FixedSizeList(_, _) => {}
            DataType::LargeList(_) => {}
            DataType::Struct(_) => {}
            DataType::Union(_) => {}
            DataType::Dictionary(_, _) => {}*/
            other => Err(ExecutionError::NotImplemented(format!("Type {:?} is still not valid.", other).to_owned())),
        };
        map.insert(record.schema().field(column_index).name().clone(), value?);
    };
    Ok(map)
}

#[pymethods]
impl ExecutionContext {
    #[new]
     fn new() -> Self {
        ExecutionContext {
            ctx: _ExecutionContext::new()
        }
    }

    fn sql(&mut self, query: &str, batch_size: usize) -> PyResult<HashMap<String, PyObject>> {
        let batches = wrap(self.ctx.sql(query, batch_size))?;
        // this is wrong: we should iterate over all batches
        Ok(wrap(to_py(&batches[0]))?)
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
