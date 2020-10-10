use std::sync::Arc;

use pyo3::{prelude::*, types::PyTuple};

use arrow::array;
use arrow::datatypes::DataType;

use datafusion::error::ExecutionError;
use datafusion::physical_plan::functions::ScalarFunctionImplementation;

use crate::to_py::to_py_array;
use crate::to_rust::to_rust;

/// creates a DataFusion's UDF implementation from a python function that expects pyarrow arrays
/// This is more efficient as it performs a zero-copy of the contents.
pub fn array_udf(func: PyObject, args_types: Vec<DataType>) -> ScalarFunctionImplementation {
    Arc::new(
        move |args: &[array::ArrayRef]| -> Result<array::ArrayRef, ExecutionError> {
            // get GIL
            let gil = pyo3::Python::acquire_gil();
            let py = gil.python();

            // 1. cast args to Pyarrow arrays
            // 2. call function
            // 3. cast to arrow::array::Array

            // 1.
            let mut py_args = Vec::with_capacity(args_types.len());
            for arg_i in 0..args_types.len() {
                let arg = &args[arg_i];
                // remove unwrap
                py_args.push(to_py_array(arg, py).unwrap());
            }
            let py_args = PyTuple::new(py, py_args);

            // 2.
            let value = func.as_ref(py).call(py_args, None);
            let value = match value {
                Ok(n) => Ok(n),
                Err(error) => Err(ExecutionError::General(format!("{:?}", error).to_owned())),
            }?;

            let array = to_rust(value).unwrap();
            Ok(array)
        },
    )
}
