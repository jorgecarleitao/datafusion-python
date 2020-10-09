use std::sync::Arc;

use pyo3::{prelude::*, types::PyTuple};

use arrow::array;
use arrow::array::PrimitiveArrayOps;
use arrow::datatypes::DataType;

use datafusion::error::ExecutionError;
use datafusion::physical_plan::functions::ScalarFunctionImplementation;

use crate::to_py::to_py_array;
use crate::to_rust::to_rust;

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
        $VALUES.iter().for_each(|x| {
            if x.is_none() {
                builder.append_null().unwrap();
            } else {
                builder.append_value(x.extract::<$TY>().unwrap()).unwrap();
            }
        });
        Ok(Arc::new(builder.finish()))
    }};
}

/// creates a DataFusion's UDF implementation from a python function
pub fn udf(
    func: PyObject,
    args_types: Vec<DataType>,
    return_type: Arc<DataType>,
) -> ScalarFunctionImplementation {
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
                    let value_i = if arg.is_null(i) {
                        py.None().to_object(py)
                    } else {
                        match &args_types[arg_i] {
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
                            DataType::Boolean => {
                                Ok(to_primitive!(arg, BooleanArray).value(i).to_object(py))
                            }
                            other => Err(ExecutionError::NotImplemented(
                                format!("Datatype \"{:?}\" is still not implemented.", other)
                                    .to_owned(),
                            )),
                        }?
                    };
                    values.push(value_i);
                }
                let values = PyTuple::new(py, values);

                // 2. call function
                let any = func.as_ref(py);
                let value = any.call(values, None);
                let value = match value {
                    Ok(n) => Ok(n),
                    Err(data) => Err(ExecutionError::General(format!("{:?}", data).to_owned())),
                }?;
                final_values.push(value);
            }

            // 3. cast the result to arrow::array::Array
            match return_type.as_ref() {
                DataType::Float64 => to_native!(final_values, Float64Builder, f64, capacity),
                DataType::Float32 => to_native!(final_values, Float32Builder, f32, capacity),
                DataType::Int32 => to_native!(final_values, Int32Builder, i32, capacity),
                DataType::Int64 => to_native!(final_values, Int64Builder, i64, capacity),
                DataType::Boolean => to_native!(final_values, BooleanBuilder, bool, capacity),
                other => Err(ExecutionError::NotImplemented(
                    format!(
                        "Datatype \"{:?}\" is still not implemented as a return type.",
                        other
                    )
                    .to_owned(),
                )),
            }
        },
    )
}

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
