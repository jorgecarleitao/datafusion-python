## Datafusion with Python

This is a Python library that binds to Apache's Arrow in-memory rust-based query engine [datafusion](https://github.com/apache/arrow/tree/master/rust/datafusion).
It allows you to build a Logical Plan through a DataFrame API against parquet or CSV files, and obtain the result back.

Being written in rust, this code has strong assumptions about thread safety and lack of memory leaks.

We lock the GIL to convert the results back to pyarrow arrays and to run UFDs.

## How to use it

Simple usage:

```python
import datafusion
import pyarrow

# an alias
f = datafusion.functions

# create a context
ctx = datafusion.ExecutionContext()

# create a RecordBatch and a new DataFrame from it
batch = pyarrow.RecordBatch.from_arrays(
    [pyarrow.array([1, 2, 3]), pyarrow.array([4, 5, 6])],
    names=["a", "b"],
)
df = ctx.create_dataframe([[batch]])

# create a new statement
df = df.select(
    f.col("a") + f.col("b"),
    f.col("a") - f.col("b"),
)

# execute and collect the first (and only) batch
result = df.collect()[0]

assert result.column(0) == pyarrow.array([5, 7, 9])
assert result.column(1) == pyarrow.array([-3, -3, -3])
```

UDF usage:

```python
# name, function, input types, output types
ctx.register_udf('my_abs', lambda x: abs(x), ['float64'], 'float64')

result = ctx.sql("SELECT my_abs(a) FROM t")
```

## How to install

We haven't configured CI/CD to publish wheels in pip yet and thus you can only install it in development.
It requires cargo and rust. See below.

## How to develop

This assumes that you have rust and cargo installed. We use the workflow recommended by [pyo3](https://github.com/PyO3/pyo3) and [maturin](https://github.com/PyO3/maturin).

Bootstrap:

```bash
# fetch this repo
git clone git@github.com:jorgecarleitao/datafusion-python.git

cd datafusion-python

# prepare development environment (used to build wheel / install in development)
python -m venv venv
venv/bin/pip install maturin==0.8.2 toml==0.10.1

# used for testing
venv/bin/pip install pyarrow==1.0.0
```

Whenever rust code changes (your changes or via git pull):

```bash
venv/bin/maturin develop
venv/bin/python -m unittest discover tests
```

## TODOs

* [x] Add support to Python UDFs
* [x] Add support to nulls
* [x] Add support to numeric types
* [x] Add support to binary types
* [x] Add support to strings
* [x] Add support to datetime (`datetime64`)
* [x] Add support to timedelta
* [ ] Add CI/CD, including publish to Mac and manylinux via official docker
* [ ] benchmarks
