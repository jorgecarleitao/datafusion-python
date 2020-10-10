use pyo3::{prelude::*, PyNumberProtocol};

use datafusion::logical_plan::Expr as _Expr;

/// An expression that can be used on a DataFrame
#[pyclass]
#[derive(Debug, Clone)]
pub(crate) struct Expression {
    pub(crate) expr: _Expr,
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
