use pyo3::{prelude::*, wrap_pyfunction};

use datafusion::logical_plan;

use crate::expression;

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
    return expression::Expression {
        expr: logical_plan::lit(value),
    };
}

pub fn init(module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(col, module)?)?;
    module.add_function(wrap_pyfunction!(lit, module)?)?;
    Ok(())
}
