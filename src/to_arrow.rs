use std::sync::Arc;

use arrow::buffer::Buffer;
use arrow::{
    array::make_array,
    array::{ArrayData, ArrayRef},
    datatypes::DataType,
};
use pyo3::prelude::*;

/// converts a pyarrow Buffer into a Rust Buffer
fn to_buffer(buffer: &PyAny) -> Result<Option<Buffer>, PyErr> {
    if buffer.is_none() {
        return Ok(None);
    }
    // this address should be aligned, but it is not. Why??
    let address = buffer.getattr("address")?.extract::<i64>()?;
    let size = buffer.getattr("size")?.extract::<i64>()? as usize;

    Ok(Some(unsafe {
        Buffer::from_raw_parts(address as *const u8, size, size)
    }))
}

/// converts a pyarrow Array into a Rust Array
pub fn to_arrow(array: &PyAny, data_type: &DataType, py: Python) -> Result<ArrayRef, PyErr> {
    let builtins = PyModule::import(py, "builtins")?;

    let len = builtins.call1("len", (array,))?.extract()?;

    let py_buffers = array.call_method0("buffers")?;
    let buffers_len = builtins.call1("len", (py_buffers,))?.extract()?;

    let mut buffers = vec![];
    let mut null_buffer = None;
    for i in 0..buffers_len {
        let buffer = py_buffers.call_method1("__getitem__", (i,))?;

        let buffer = to_buffer(buffer)?;
        if i == 0 {
            null_buffer = buffer;
        } else {
            buffers.push(buffer.unwrap())
        }
    }

    let data = ArrayData::new(
        data_type.clone(),
        len,
        None,
        null_buffer,
        0,
        buffers,
        vec![],
    );
    Ok(make_array(Arc::new(data)))
}
