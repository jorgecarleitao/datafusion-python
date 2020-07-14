use pyo3::prelude::*;

use datafusion::error::ExecutionError;

use numpy::PyArray1;

use std::collections::HashMap;

use arrow::array;
use arrow::datatypes::{TimeUnit, DateUnit};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;


/// Maps a numpy dtype to a PyObject denoting its null representation.
fn py_null(numpy_type: &str, py: &Python, numpy: &PyModule) -> Result<PyObject, ExecutionError> {
    match numpy_type {
        "O" => Ok(py.None()),
        "datetime64[s]" => Ok(PyObject::from(numpy.call("datetime64", ("NaT",), None).unwrap())),
        "datetime64[us]" => Ok(PyObject::from(numpy.call("datetime64", ("NaT",), None).unwrap())),
        "datetime64[ms]" => Ok(PyObject::from(numpy.call("datetime64", ("NaT",), None).unwrap())),
        "datetime64[ns]" => Ok(PyObject::from(numpy.call("datetime64", ("NaT",), None).unwrap())),
        "datetime64[D]" => Ok(PyObject::from(numpy.call("datetime64", ("NaT",), None).unwrap())),
        "timedelta64[ms]" => Ok(PyObject::from(numpy.call("timedelta64", ("NaT",), None).unwrap())),
        "timedelta64[us]" => Ok(PyObject::from(numpy.call("timedelta64", ("NaT",), None).unwrap())),
        "timedelta64[ns]" => Ok(PyObject::from(numpy.call("timedelta64", ("NaT",), None).unwrap())),
        "timedelta64[s]" => Ok(PyObject::from(numpy.call("timedelta64", ("NaT",), None).unwrap())),
        _ => Err(ExecutionError::NotImplemented(
            format!("Unknown null value of type \"{}\" ", numpy_type).to_owned(),
        ))
    }
}


macro_rules! to_py_numpy_generic {
    ($BATCHES:ident, $COLUMN_INDEX:ident, $ARRAY_TY:ident, $NUMPY_TY:expr) => {{
        let mut values = Vec::with_capacity($BATCHES.len() * $BATCHES[0].num_rows());
        let mut mask = Vec::with_capacity($BATCHES.len() * $BATCHES[0].num_rows());
        for batch in $BATCHES {
            let column = batch.column($COLUMN_INDEX);
            let casted = column.as_any().downcast_ref::<array::$ARRAY_TY>().unwrap();
            for i in 0..column.len() {
                mask.push(column.is_null(i));
                values.push(casted.value(i));
            }
        };
        let gil = pyo3::Python::acquire_gil();
        let py = gil.python();

        // use a Python call to construct the array, since rust-numpy does not support this type yet.
        let numpy = PyModule::import(py, "numpy").expect("Numpy is not installed.");
        let values = numpy.call1("array", (values, $NUMPY_TY)).expect("asas");

        // apply null mask to the values.
        let values = numpy.call1("where", (mask, py_null($NUMPY_TY, &py, &numpy)?, values)).expect("asas");

        Ok(PyObject::from(values))
    }};
}


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
            // ignoring time zones for now...
            DataType::Timestamp(TimeUnit::Second, _) => to_py_numpy_generic!(batches, column_index, TimestampSecondArray, "datetime64[s]"),
            DataType::Timestamp(TimeUnit::Millisecond, _) => to_py_numpy_generic!(batches, column_index, TimestampMillisecondArray, "datetime64[ms]"),
            DataType::Timestamp(TimeUnit::Microsecond, _) => to_py_numpy_generic!(batches, column_index, TimestampMicrosecondArray, "datetime64[us]"),
            DataType::Timestamp(TimeUnit::Nanosecond, _) => to_py_numpy_generic!(batches, column_index, TimestampNanosecondArray, "datetime64[ns]"),
            DataType::Date32(DateUnit::Day) => to_py_numpy_generic!(batches, column_index, Date32Array, "datetime64[D]"),
            DataType::Date64(DateUnit::Millisecond) => to_py_numpy_generic!(batches, column_index, Date64Array, "datetime64[ms]"),
            DataType::Duration(TimeUnit::Second) => to_py_numpy_generic!(batches, column_index, DurationSecondArray, "timedelta64[s]"),
            DataType::Duration(TimeUnit::Millisecond) => to_py_numpy_generic!(batches, column_index, DurationMillisecondArray, "timedelta64[ms]"),
            DataType::Duration(TimeUnit::Microsecond) => to_py_numpy_generic!(batches, column_index, DurationMicrosecondArray, "timedelta64[us]"),
            DataType::Duration(TimeUnit::Nanosecond) => to_py_numpy_generic!(batches, column_index, DurationNanosecondArray, "timedelta64[ns]"),
            /*
            No native type in numpy
            DataType::Time32(_) => {}
            DataType::Time64(_) => {}
            */
            /*
            None of the below are currently supported by rust-numpy
            DataType::Interval(_) => {}
            DataType::Binary => {}
            DataType::FixedSizeBinary(_) => {}
            DataType::LargeBinary => {}
            */
            DataType::Utf8 => to_py_numpy_generic!(batches, column_index, StringArray, "O"),
            DataType::LargeUtf8 => to_py_numpy_generic!(batches, column_index, StringArray, "O"),
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
