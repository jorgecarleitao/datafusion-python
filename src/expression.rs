use pyo3::prelude::*;

use datafusion::logical_plan::Expr as _Expr;

/// An expression that can be used on a DataFrame
#[pyclass]
#[derive(Debug, Clone)]
pub(crate) struct Expression {
    pub(crate) expr: _Expr,
}
