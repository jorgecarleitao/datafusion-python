use std::sync::Arc;

use logical_plan::LogicalPlan;
use pyo3::{prelude::*, wrap_pyfunction};
use tokio::runtime::Runtime;

use std::collections::HashSet;

use datafusion::execution::context::ExecutionContext as _ExecutionContext;
use datafusion::logical_plan::create_udf;
use datafusion::{
    error::ExecutionError, logical_plan::Expr as _Expr, logical_plan::LogicalPlanBuilder,
};
use datafusion::{execution::context::ExecutionContextState, logical_plan};

use arrow::datatypes::DataType;
mod errors;

/// An expression that can be used on a DataFrame
#[pyclass]
#[derive(Debug, Clone)]
struct Expr {
    expr: _Expr,
}

#[pyclass(unsendable)]
struct DataFrame {
    ctx_state: ExecutionContextState,
    plan: LogicalPlan,
}

#[pymethods]
impl DataFrame {
    fn select(&self, expressions: Vec<Expr>) -> PyResult<Self> {
        let builder = LogicalPlanBuilder::from(&self.plan);
        let builder = wrap(builder.project(expressions.iter().map(|e| e.expr.clone()).collect()))?;
        let plan = wrap(builder.build())?;

        Ok(DataFrame {
            ctx_state: self.ctx_state.clone(),
            plan,
        })
    }

    fn collect(&self) -> PyResult<PyObject> {
        let mut rt = Runtime::new().unwrap();

        let ctx = _ExecutionContext::from(self.ctx_state.clone());
        let plan = ctx
            .optimize(&self.plan)
            .map_err(|e| -> errors::DataFusionError { e.into() })?;
        let plan = ctx
            .create_physical_plan(&plan)
            .map_err(|e| -> errors::DataFusionError { e.into() })?;

        let batches = rt.block_on(async {
            ctx.collect(plan)
                .await
                .map_err(|e| -> errors::DataFusionError { e.into() })
        })?;
        to_py::to_py(&batches)
    }
}

#[pyclass(unsendable)]
struct ExecutionContext {
    ctx: _ExecutionContext,
}

fn wrap<T>(a: Result<T, ExecutionError>) -> Result<T, errors::DataFusionError> {
    Ok(a?)
}

#[pymethods]
impl ExecutionContext {
    #[new]
    fn new() -> Self {
        ExecutionContext {
            ctx: _ExecutionContext::new(),
        }
    }

    /// Returns a DataFrame whose plan corresponds to the SQL statement.
    fn sql(&mut self, query: &str) -> PyResult<DataFrame> {
        let df = self
            .ctx
            .sql(query)
            .map_err(|e| -> errors::DataFusionError { e.into() })?;
        Ok(DataFrame {
            ctx_state: self.ctx.state.clone(),
            plan: df.to_logical_plan(),
        })
    }

    fn register_parquet(&mut self, name: &str, path: &str) -> PyResult<()> {
        wrap(self.ctx.register_parquet(name, path))?;
        Ok(())
    }

    fn register_udf(
        &mut self,
        name: &str,
        func: PyObject,
        args_types: Vec<&str>,
        return_type: &str,
    ) -> PyResult<()> {
        // map strings for declared types to cases from DataType
        let args_types: Vec<DataType> = wrap(
            args_types
                .iter()
                .map(|x| types::data_type(&x))
                .collect::<Result<Vec<DataType>, ExecutionError>>(),
        )?;
        let return_type = Arc::new(wrap(types::data_type(return_type))?);

        self.ctx.register_udf(create_udf(
            name.into(),
            args_types.clone(),
            return_type.clone(),
            udf::udf(func, args_types, return_type),
        ));
        Ok(())
    }

    fn register_array_udf(
        &mut self,
        name: &str,
        func: PyObject,
        args_types: Vec<&str>,
        return_type: &str,
    ) -> PyResult<()> {
        // map strings for declared types to cases from DataType
        let args_types: Vec<DataType> = wrap(
            args_types
                .iter()
                .map(|x| types::data_type(&x))
                .collect::<Result<Vec<DataType>, ExecutionError>>(),
        )?;
        let return_type = Arc::new(wrap(types::data_type(return_type))?);

        self.ctx.register_udf(create_udf(
            name.into(),
            args_types.clone(),
            return_type.clone(),
            udf::array_udf(func, args_types),
        ));
        Ok(())
    }

    fn tables(&self) -> HashSet<String> {
        self.ctx.tables()
    }
}

/// Expression representing a column
#[pyfunction]
#[text_signature = "(name)"]
fn col(name: &str) -> Expr {
    return Expr {
        expr: logical_plan::col(name),
    };
}

/// Returns a literal expression
#[pyfunction]
#[text_signature = "(value)"]
fn lit(value: i32) -> Expr {
    return Expr {
        expr: logical_plan::lit(value),
    };
}

/// The module DataFusion...
#[pymodule]
fn datafusion(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ExecutionContext>()?;
    m.add_class::<DataFrame>()?;
    m.add_class::<Expr>()?;

    // expressions
    m.add_wrapped(wrap_pyfunction!(lit))?;
    m.add_wrapped(wrap_pyfunction!(col))?;

    for data_type in types::DATA_TYPES {
        m.add(data_type, *data_type)?;
    }

    Ok(())
}

mod to_py;
mod to_rust;
mod types;
mod udf;
