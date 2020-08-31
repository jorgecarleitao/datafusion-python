use pyo3::prelude::*;

use datafusion::error::ExecutionError;

use std::collections::HashMap;

use arrow::array;
use arrow::datatypes::DataType;
use arrow::datatypes::{DateUnit, TimeUnit};
use arrow::record_batch::RecordBatch;

/// Maps a numpy dtype to a PyObject denoting its null representation.
fn py_null(numpy_type: &str, py: &Python, numpy: &PyModule) -> Option<PyObject> {
    match numpy_type {
        "O" => Some(py.None()),
        "float32" => Some(PyObject::from(numpy.get("NaN").unwrap())),
        "float64" => Some(PyObject::from(numpy.get("NaN").unwrap())),
        "datetime64[s]" => Some(PyObject::from(
            numpy.call("datetime64", ("NaT",), None).unwrap(),
        )),
        "datetime64[us]" => Some(PyObject::from(
            numpy.call("datetime64", ("NaT",), None).unwrap(),
        )),
        "datetime64[ms]" => Some(PyObject::from(
            numpy.call("datetime64", ("NaT",), None).unwrap(),
        )),
        "datetime64[ns]" => Some(PyObject::from(
            numpy.call("datetime64", ("NaT",), None).unwrap(),
        )),
        "datetime64[D]" => Some(PyObject::from(
            numpy.call("datetime64", ("NaT",), None).unwrap(),
        )),
        "timedelta64[ms]" => Some(PyObject::from(
            numpy.call("timedelta64", ("NaT",), None).unwrap(),
        )),
        "timedelta64[us]" => Some(PyObject::from(
            numpy.call("timedelta64", ("NaT",), None).unwrap(),
        )),
        "timedelta64[ns]" => Some(PyObject::from(
            numpy.call("timedelta64", ("NaT",), None).unwrap(),
        )),
        "timedelta64[s]" => Some(PyObject::from(
            numpy.call("timedelta64", ("NaT",), None).unwrap(),
        )),
        _ => None,
    }
}

macro_rules! to_py_numpy {
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
        let values = numpy.call1("array", (values, $NUMPY_TY)).expect("Could not create numpy array");

        // apply null mask to the values when a null value exists
        let values = match py_null($NUMPY_TY, &py, &numpy) {
            Some(null) => numpy.call1("where", (mask, null, values)).expect("Could not apply null mask"),
            None => values,
        };

        Ok(PyObject::from(values))
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
            DataType::Boolean => to_py_numpy!(batches, column_index, BooleanArray, "?"),
            DataType::Int8 => to_py_numpy!(batches, column_index, Int8Array, "int8"),
            DataType::Int16 => to_py_numpy!(batches, column_index, Int16Array, "int16"),
            DataType::Int32 => to_py_numpy!(batches, column_index, Int32Array, "int32"),
            DataType::Int64 => to_py_numpy!(batches, column_index, Int64Array, "int64"),
            DataType::UInt8 => to_py_numpy!(batches, column_index, UInt8Array, "uint8"),
            DataType::UInt16 => to_py_numpy!(batches, column_index, UInt16Array, "uint16"),
            DataType::UInt32 => to_py_numpy!(batches, column_index, UInt32Array, "uint32"),
            DataType::UInt64 => to_py_numpy!(batches, column_index, UInt64Array, "uint64"),
            //DataType::Float16 is not represented in rust arrow
            DataType::Float32 => to_py_numpy!(batches, column_index, Float32Array, "float32"),
            DataType::Float64 => to_py_numpy!(batches, column_index, Float64Array, "float64"),
            // numpy does not support timezones, thus we ignore them
            DataType::Timestamp(TimeUnit::Second, _) => {
                to_py_numpy!(batches, column_index, TimestampSecondArray, "datetime64[s]")
            }
            DataType::Timestamp(TimeUnit::Millisecond, _) => to_py_numpy!(
                batches,
                column_index,
                TimestampMillisecondArray,
                "datetime64[ms]"
            ),
            DataType::Timestamp(TimeUnit::Microsecond, _) => to_py_numpy!(
                batches,
                column_index,
                TimestampMicrosecondArray,
                "datetime64[us]"
            ),
            DataType::Timestamp(TimeUnit::Nanosecond, _) => to_py_numpy!(
                batches,
                column_index,
                TimestampNanosecondArray,
                "datetime64[ns]"
            ),
            DataType::Date32(DateUnit::Day) => {
                to_py_numpy!(batches, column_index, Date32Array, "datetime64[D]")
            }
            DataType::Date64(DateUnit::Millisecond) => {
                to_py_numpy!(batches, column_index, Date64Array, "datetime64[ms]")
            }
            DataType::Duration(TimeUnit::Second) => {
                to_py_numpy!(batches, column_index, DurationSecondArray, "timedelta64[s]")
            }
            DataType::Duration(TimeUnit::Millisecond) => to_py_numpy!(
                batches,
                column_index,
                DurationMillisecondArray,
                "timedelta64[ms]"
            ),
            DataType::Duration(TimeUnit::Microsecond) => to_py_numpy!(
                batches,
                column_index,
                DurationMicrosecondArray,
                "timedelta64[us]"
            ),
            DataType::Duration(TimeUnit::Nanosecond) => to_py_numpy!(
                batches,
                column_index,
                DurationNanosecondArray,
                "timedelta64[ns]"
            ),
            DataType::Binary => to_py_numpy!(batches, column_index, BinaryArray, "u8"),
            DataType::FixedSizeBinary(byte_width) => to_py_numpy!(
                batches,
                column_index,
                BinaryArray,
                &*format!("u{}", byte_width * 8)
            ),
            DataType::LargeBinary => to_py_numpy!(batches, column_index, LargeBinaryArray, "u8"),
            /*
            No native type in numpy
            DataType::Time32(_) => {}
            DataType::Time64(_) => {}
            DataType::Interval(_) => {}
            */
            DataType::Utf8 => to_py_numpy!(batches, column_index, StringArray, "O"),
            DataType::LargeUtf8 => to_py_numpy!(batches, column_index, StringArray, "O"),
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
