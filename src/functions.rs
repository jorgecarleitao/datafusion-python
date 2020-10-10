use std::sync::Arc;

use arrow::datatypes::DataType;
use pyo3::{prelude::*, wrap_pyfunction};

use datafusion::logical_plan;

use crate::udf;
use crate::{expression, types::PyDataType};

/// Expression representing a column on the existing plan.
#[pyfunction]
#[text_signature = "(name)"]
fn col(name: &str) -> expression::Expression {
    return expression::Expression {
        expr: logical_plan::col(name),
    };
}

/// Expression representing a constant value
#[pyfunction]
#[text_signature = "(value)"]
fn lit(value: i32) -> expression::Expression {
    expression::Expression {
        expr: logical_plan::lit(value),
    }
}

pub(crate) fn create_udf(
    fun: PyObject,
    input_types: Vec<PyDataType>,
    return_type: PyDataType,
    name: &str,
) -> PyResult<expression::ScalarUDF> {
    let input_types: Vec<DataType> = input_types.iter().map(|d| d.data_type.clone()).collect();
    let return_type = Arc::new(return_type.data_type);

    Ok(expression::ScalarUDF {
        function: logical_plan::create_udf(
            name,
            input_types.clone(),
            return_type,
            udf::array_udf(fun, input_types),
        ),
    })
}

/// Creates a new udf.
#[pyfunction]
fn udf(
    fun: PyObject,
    input_types: Vec<PyDataType>,
    return_type: PyDataType,
    py: Python,
) -> PyResult<expression::ScalarUDF> {
    let name = fun.getattr(py, "__qualname__")?.extract::<String>(py)?;

    create_udf(fun, input_types, return_type, &name)
}

pub fn init(module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(col, module)?)?;
    module.add_function(wrap_pyfunction!(lit, module)?)?;
    module.add_function(wrap_pyfunction!(udf, module)?)?;
    Ok(())
}
