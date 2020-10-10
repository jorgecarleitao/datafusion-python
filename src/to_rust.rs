use std::sync::Arc;

use arrow::{
    array::{make_array_from_raw, ArrayRef},
    datatypes::Field,
    datatypes::Schema,
    ffi,
    record_batch::RecordBatch,
};
use pyo3::{libc::uintptr_t, prelude::*};

use crate::{errors, types::data_type_id};

/// converts a pyarrow Array into a Rust Array
pub fn to_rust(ob: &PyAny) -> PyResult<ArrayRef> {
    // prepare a pointer to receive the Array struct
    let (array_pointer, schema_pointer) =
        ffi::ArrowArray::into_raw(unsafe { ffi::ArrowArray::empty() });

    // make the conversion through PyArrow's private API
    // this changes the pointer's memory and is thus unsafe. In particular, `_export_to_c` can go out of bounds
    ob.call_method1(
        "_export_to_c",
        (array_pointer as uintptr_t, schema_pointer as uintptr_t),
    )?;

    let array = unsafe { make_array_from_raw(array_pointer, schema_pointer) }
        .map_err(|e| errors::DataFusionError::from(e))?;
    Ok(array)
}

pub fn to_rust_batch(batch: &PyAny) -> PyResult<RecordBatch> {
    let schema = batch.getattr("schema")?;
    let names = schema.getattr("names")?.extract::<Vec<String>>()?;

    let fields = names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let field = schema.call_method1("field", (i,))?;
            let nullable = field.getattr("nullable")?.extract::<bool>()?;
            let t = field.getattr("type")?.getattr("id")?.extract::<i32>()?;
            let data_type = data_type_id(&t)?;
            Ok(Field::new(name, data_type, nullable))
        })
        .collect::<PyResult<_>>()?;

    let schema = Arc::new(Schema::new(fields));

    let arrays = (0..names.len())
        .map(|i| {
            let array = batch.call_method1("column", (i,))?;
            to_rust(array)
        })
        .collect::<PyResult<_>>()?;

    let batch =
        RecordBatch::try_new(schema, arrays).map_err(|e| errors::DataFusionError::from(e))?;
    Ok(batch)
}
