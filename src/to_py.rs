use pyo3::exceptions;
use pyo3::PyErr;

use pyo3::prelude::*;

use std::convert::From;
use std::mem;

use arrow::datatypes::DataType;
use arrow::datatypes::{DateUnit, TimeUnit};
use arrow::record_batch::RecordBatch;
use arrow::{array::ArrayRef, buffer::Buffer};

#[derive(Debug)]
pub enum DataStoreError {
    ExecutionError(String),
}

impl From<DataStoreError> for PyErr {
    fn from(err: DataStoreError) -> PyErr {
        match err {
            DataStoreError::ExecutionError(err) => exceptions::Exception::py_err(err.to_string()),
        }
    }
}

/// maps a Rust's DataType to its name in pyarrow
fn type_to_type<'a>(data_type: &DataType, pyarrow: &'a PyModule) -> Result<&'a PyAny, PyErr> {
    match data_type {
        DataType::Boolean => pyarrow.call0("bool_"),
        DataType::Int8 => pyarrow.call0("int8"),
        DataType::Int16 => pyarrow.call0("int16"),
        DataType::Int32 => pyarrow.call0("int32"),
        DataType::Int64 => pyarrow.call0("int64"),
        DataType::UInt8 => pyarrow.call0("uint8"),
        DataType::UInt16 => pyarrow.call0("uint16"),
        DataType::UInt32 => pyarrow.call0("uint32"),
        DataType::UInt64 => pyarrow.call0("uint64"),
        DataType::Float32 => pyarrow.call0("float32"),
        DataType::Float64 => pyarrow.call0("float64"),
        DataType::Binary => pyarrow.call0("binary"),
        DataType::LargeBinary => pyarrow.call0("large_binary"),
        DataType::FixedSizeBinary(t) => {
            let binary = pyarrow.getattr("binary")?;
            binary.call1((*t,))
        }
        DataType::Date32(t) => match t {
            DateUnit::Day => pyarrow.call0("date32"),
            other => {
                return Err(DataStoreError::ExecutionError(
                    format!("Type Date32({:?}) is still not valid.", other).to_owned(),
                )
                .into())
            }
        },
        DataType::Timestamp(t, _) => {
            let timestamp = pyarrow.getattr("timestamp")?;
            match t {
                TimeUnit::Second => timestamp.call1(("s",)),
                TimeUnit::Millisecond => timestamp.call1(("ms",)),
                TimeUnit::Microsecond => timestamp.call1(("us",)),
                TimeUnit::Nanosecond => timestamp.call1(("ns",)),
            }
        }
        DataType::Duration(t) => {
            let duration = pyarrow.getattr("duration")?;
            match t {
                TimeUnit::Second => duration.call1(("s",)),
                TimeUnit::Millisecond => duration.call1(("ms",)),
                TimeUnit::Microsecond => duration.call1(("us",)),
                TimeUnit::Nanosecond => duration.call1(("ns",)),
            }
        }
        DataType::Utf8 => pyarrow.call0("utf8"),
        DataType::LargeUtf8 => pyarrow.call0("large_utf8"),
        other => {
            return Err(DataStoreError::ExecutionError(
                format!("Type {:?} is still not valid.", other).to_owned(),
            )
            .into())
        }
    }
}

fn init_py_buffer<'a>(buffer: &Buffer, pyarrow: &'a PyModule) -> Result<&'a PyAny, PyErr> {
    // this assumes a 64 bit system
    let pointer = buffer.raw_data() as i64;
    pyarrow.call1("foreign_buffer", (pointer, buffer.len()))
}

/// performs a zero-copy conversion between a Rust's Array to a Pyarrow array.
pub fn to_py_array<'a>(
    array: &ArrayRef,
    py: Python,
    pyarrow: &'a PyModule,
) -> Result<PyObject, PyErr> {
    let a = pyarrow.getattr("Array")?;
    let none = py.None();

    let column_type = array.data_type();
    let data_type = type_to_type(column_type, pyarrow)?;

    let data = array.data();

    let null_buffer = data.null_buffer().map_or_else(
        || none.extract(py),
        |buffer| init_py_buffer(buffer, pyarrow),
    )?;

    let mut py_buffers: Vec<&PyAny> = vec![null_buffer];
    for buffer in data.buffers() {
        let buffer = init_py_buffer(buffer, pyarrow)?;

        py_buffers.push(buffer);
    }

    // "leak" the data, as the buffer now owns it.
    // todo: is this the correct way of FFI with rust?
    // todo: does Python own it?
    // todo: maybe a phantom object passed to the foreign_buffer?
    mem::forget(data);

    let array = a.call_method1("from_buffers", (data_type, array.len(), py_buffers))?;

    Ok(PyObject::from(array))
}

fn to_py_batch<'a>(
    batch: &RecordBatch,
    py: Python,
    pyarrow: &'a PyModule,
) -> Result<PyObject, PyErr> {
    let mut py_arrays = vec![];
    let mut py_names = vec![];

    let schema = batch.schema();
    for (array, field) in batch.columns().iter().zip(schema.fields().iter()) {
        let array = to_py_array(array, py, pyarrow)?;

        py_arrays.push(array);
        py_names.push(field.name());
    }

    let record = pyarrow
        .getattr("RecordBatch")?
        .call_method1("from_arrays", (py_arrays, py_names))?;

    Ok(PyObject::from(record))
}

/// Converts a Vec<RecordBatch> into a RecordBatch represented in Python
pub fn to_py(batches: &Vec<RecordBatch>) -> Result<PyObject, PyErr> {
    let gil = pyo3::Python::acquire_gil();
    let py = gil.python();
    let pyarrow = PyModule::import(py, "pyarrow")?;
    let builtins = PyModule::import(py, "builtins")?;

    let mut py_batches = vec![];
    for batch in batches {
        py_batches.push(to_py_batch(batch, py, pyarrow)?);
    }
    let result = builtins.call1("list", (py_batches,))?;
    Ok(PyObject::from(result))
}
