use logical_plan::LogicalPlan;
use pyo3::prelude::*;
use tokio::runtime::Runtime;

use datafusion::execution::context::ExecutionContext as _ExecutionContext;
use datafusion::logical_plan::LogicalPlanBuilder;
use datafusion::{execution::context::ExecutionContextState, logical_plan};

use crate::expression;
use crate::{errors, to_py};

#[pyclass(unsendable)]
pub(crate) struct DataFrame {
    ctx_state: ExecutionContextState,
    plan: LogicalPlan,
}

impl DataFrame {
    /// creates a new DataFrame
    pub fn new(ctx_state: ExecutionContextState, plan: LogicalPlan) -> Self {
        Self { ctx_state, plan }
    }
}

#[pymethods]
impl DataFrame {
    fn select(&self, expressions: Vec<expression::Expression>) -> PyResult<Self> {
        let builder = LogicalPlanBuilder::from(&self.plan);
        let builder =
            errors::wrap(builder.project(expressions.iter().map(|e| e.expr.clone()).collect()))?;
        let plan = errors::wrap(builder.build())?;

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
