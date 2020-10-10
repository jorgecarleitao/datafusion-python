use pyo3::{prelude::*, types::PyTuple, PyNumberProtocol};

use datafusion::logical_plan::Expr as _Expr;
use datafusion::physical_plan::udf::ScalarUDF as _ScalarUDF;

/// An expression that can be used on a DataFrame
#[pyclass]
#[derive(Debug, Clone)]
pub(crate) struct Expression {
    pub(crate) expr: _Expr,
}

/// converts a tuple of expressions into a vector of Expressions
pub(crate) fn from_tuple(value: &PyTuple) -> PyResult<Vec<Expression>> {
    value
        .iter()
        .map(|e| e.extract::<Expression>())
        .collect::<PyResult<_>>()
}

#[pyproto]
impl PyNumberProtocol for Expression {
    fn __add__(lhs: Expression, rhs: Expression) -> PyResult<Expression> {
        Ok(Expression {
            expr: lhs.expr + rhs.expr,
        })
    }

    fn __sub__(lhs: Expression, rhs: Expression) -> PyResult<Expression> {
        Ok(Expression {
            expr: lhs.expr - rhs.expr,
        })
    }
}

// these are here until https://github.com/PyO3/pyo3/issues/1219 is closed and released
#[pymethods]
impl Expression {
    /// operator ">"
    pub fn gt(&self, rhs: Expression) -> PyResult<Expression> {
        Ok(Expression {
            expr: self.expr.gt(rhs.expr),
        })
    }
}

/// Represents a ScalarUDF
#[pyclass]
#[derive(Debug, Clone)]
pub struct ScalarUDF {
    pub(crate) function: _ScalarUDF,
}

#[pymethods]
impl ScalarUDF {
    /// creates a new expression with the call of the udf
    #[call]
    #[args(args = "*")]
    fn __call__(&self, args: &PyTuple) -> PyResult<Expression> {
        let args = from_tuple(args)?.iter().map(|e| e.expr.clone()).collect();

        Ok(Expression {
            expr: self.function.call(args),
        })
    }
}
