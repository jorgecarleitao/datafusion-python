use pyo3::prelude::*;

use datafusion::error::ExecutionError;

use numpy::PyArray1;

use std::collections::HashMap;

use arrow::array;
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;


fn to_py_str(batches: &Vec<RecordBatch>, column_index: usize) -> Result<PyObject, ExecutionError> {
    let mut values = Vec::with_capacity(batches.len() * batches[0].num_rows());
    for batch in batches {
        let column = batch.column(column_index);
        let casted = column.as_any().downcast_ref::<array::StringArray>().unwrap();
        for i in 0..column.len() {
            values.push(casted.value(i));
        }
    };
    let gil = pyo3::Python::acquire_gil();
    let py = gil.python();

    // use a Python call to construct the array, since rust-numpy does not support this type yet.
    let numpy = PyModule::import(py, "numpy").expect("Numpy is not installed.");
    let values = numpy.call1("array", (values, "O")).expect("asas");

    Ok(PyObject::from(values))
}

//numpy.array

macro_rules! to_py_numpy {
    ($BATCHES:ident, $COLUMN_INDEX:ident, $ARRAY_TY:ident) => {{
        let mut values = Vec::with_capacity($BATCHES.len() * $BATCHES[0].num_rows());
        for batch in $BATCHES {
            let column = batch.column($COLUMN_INDEX);
            let casted = column.as_any().downcast_ref::<array::$ARRAY_TY>().unwrap();
            for i in 0..column.len() {
                values.push(casted.value(i));
            }
        };
        let gil = pyo3::Python::acquire_gil();
        Ok(PyObject::from(PyArray1::from_iter(gil.python(), values)))
    }};
}

/// Converts a Vec<RecordBatch> into HashMap<String, PyObject>
pub fn to_py(batches: &Vec<RecordBatch>) -> Result<HashMap<String, PyObject>, ExecutionError> {
    let mut map: HashMap<String, PyObject> = HashMap::new();

    let schema = batches[0].schema();

    for column_index in 0..schema.fields().len() {
        let column_name = schema.field(column_index).name().clone();
        let column_type = schema.field(column_index).data_type();

        let value = match column_type {
            //DataType::Null: no NullArray in arrow
            DataType::Boolean => to_py_numpy!(batches, column_index, BooleanArray),
            DataType::Int8 => to_py_numpy!(batches, column_index, Int8Array),
            DataType::Int16 => to_py_numpy!(batches, column_index, Int16Array),
            DataType::Int32 => to_py_numpy!(batches, column_index, Int32Array),
            DataType::Int64 => to_py_numpy!(batches, column_index, Int64Array),
            DataType::UInt8 => to_py_numpy!(batches, column_index, UInt8Array),
            DataType::UInt16 => to_py_numpy!(batches, column_index, UInt16Array),
            DataType::UInt32 => to_py_numpy!(batches, column_index, UInt32Array),
            DataType::UInt64 => to_py_numpy!(batches, column_index, UInt64Array),
            //DataType::Float16 is not represented in rust arrow
            DataType::Float32 => to_py_numpy!(batches, column_index, Float32Array),
            DataType::Float64 => to_py_numpy!(batches, column_index, Float64Array),
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
            */
            DataType::Utf8 => to_py_str(batches, column_index),
            DataType::LargeUtf8 => to_py_str(batches, column_index),
            /*
            DataType::List(_) => {}
            DataType::FixedSizeList(_, _) => {}
            DataType::LargeList(_) => {}
            DataType::Struct(_) => {}
            DataType::Union(_) => {}
            DataType::Dictionary(_, _) => {}*/
            other => Err(ExecutionError::NotImplemented(
                format!("Type {:?} is still not valid.", other).to_owned(),
            )),
        };
        map.insert(column_name, value?);
    }
    Ok(map)
}
