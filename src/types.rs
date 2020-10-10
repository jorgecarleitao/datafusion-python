use arrow::datatypes::DataType;
use datafusion::error::ExecutionError;

use crate::errors;

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

pub fn data_type_id(id: &i32) -> Result<DataType, errors::DataFusionError> {
    // see https://github.com/apache/arrow/blob/3694794bdfd0677b95b8c95681e392512f1c9237/python/pyarrow/includes/libarrow.pxd
    Ok(match id {
        1 => DataType::Boolean,
        2 => DataType::UInt8,
        3 => DataType::Int8,
        4 => DataType::UInt16,
        5 => DataType::Int16,
        6 => DataType::UInt32,
        7 => DataType::Int32,
        8 => DataType::UInt64,
        9 => DataType::Int64,

        10 => DataType::Float16,
        11 => DataType::Float32,
        12 => DataType::Float64,

        //13 => DataType::Decimal,

        // 14 => DataType::Date32(),
        // 15 => DataType::Date64(),
        // 16 => DataType::Timestamp(),
        // 17 => DataType::Time32(),
        // 18 => DataType::Time64(),
        // 19 => DataType::Duration()
        20 => DataType::Binary,
        21 => DataType::Utf8,
        22 => DataType::LargeBinary,
        23 => DataType::LargeUtf8,

        other => {
            return Err(errors::DataFusionError::Common(format!(
                "The type {} is not valid",
                other
            )))
        }
    })
}
