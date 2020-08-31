use arrow::datatypes::DataType;
use datafusion::error::ExecutionError;

pub fn data_type(string: &str) -> Result<DataType, ExecutionError> {
    match string {
        "bool" => Ok(DataType::Boolean),
        "int32" => Ok(DataType::Int32),
        "int64" => Ok(DataType::Int64),
        "int" => Ok(DataType::Int64),
        "float32" => Ok(DataType::Float32),
        "float64" => Ok(DataType::Float64),
        "float" => Ok(DataType::Float64),
        other => Err(ExecutionError::General(format!(
            "The type {} is not valid",
            other
        ))),
    }
}

pub const DATA_TYPES: &'static [&'static str] = &[
    "bool", "int32", "int64", "int", "float32", "float64", "float",
];
