use logical_plan::LogicalPlan;
use pyo3::{prelude::*, types::PyTuple};
use tokio::runtime::Runtime;

use datafusion::execution::context::ExecutionContext as _ExecutionContext;
use datafusion::logical_plan::LogicalPlanBuilder;
use datafusion::{execution::context::ExecutionContextState, logical_plan};

use crate::expression;
use crate::{errors, to_py};

/// A DataFrame is a representation of a logical plan and an API to compose statements.
/// Use it to build a plan and `.collect()` to execute the plan and collect the result.
/// The actual execution of a plan runs natively on Rust and Arrow on a multi-threaded environment.
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
    /// Select `expressions` from the existing DataFrame.
    #[args(args = "*")]
    fn select(&self, args: &PyTuple) -> PyResult<Self> {
        let expressions = expression::from_tuple(args)?;
        let builder = LogicalPlanBuilder::from(&self.plan);
        let builder =
            errors::wrap(builder.project(expressions.iter().map(|e| e.expr.clone()).collect()))?;
        let plan = errors::wrap(builder.build())?;

        Ok(DataFrame {
            ctx_state: self.ctx_state.clone(),
            plan,
        })
    }

    /// Filter according to the `predicate` expression
    fn filter(&self, predicate: expression::Expression) -> PyResult<Self> {
        let builder = LogicalPlanBuilder::from(&self.plan);
        let builder = errors::wrap(builder.filter(predicate.expr))?;
        let plan = errors::wrap(builder.build())?;

        Ok(DataFrame {
            ctx_state: self.ctx_state.clone(),
            plan,
        })
    }

    /// Aggregates using expressions
    fn aggregate(
        &self,
        group_by: Vec<expression::Expression>,
        aggs: Vec<expression::Expression>,
    ) -> PyResult<Self> {
        let builder = LogicalPlanBuilder::from(&self.plan);
        let builder = errors::wrap(builder.aggregate(
            group_by.iter().map(|e| e.expr.clone()).collect(),
            aggs.iter().map(|e| e.expr.clone()).collect(),
        ))?;
        let plan = errors::wrap(builder.build())?;

        Ok(DataFrame {
            ctx_state: self.ctx_state.clone(),
            plan,
        })
    }

    /// Limits the plan to return at most `count` rows
    fn limit(&self, count: usize) -> PyResult<Self> {
        let builder = LogicalPlanBuilder::from(&self.plan);
        let builder = errors::wrap(builder.limit(count))?;
        let plan = errors::wrap(builder.build())?;

        Ok(DataFrame {
            ctx_state: self.ctx_state.clone(),
            plan,
        })
    }

    /// Executes the plan, returning a list of `RecordBatch`es.
    /// Unless some order is specified in the plan, there is no guarantee of the order of the result
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
