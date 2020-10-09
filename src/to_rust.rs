use arrow::{
    array::{make_array_from_raw, ArrayRef},
    ffi,
};
use pyo3::{libc::uintptr_t, prelude::*};

use crate::errors;

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
