## Datafusion with Python

This is a Python library that binds to Apache's Arrow in-memory rust-based query engine [datafusion](https://github.com/apache/arrow/tree/master/rust/datafusion).
It allows you to execute SQL queries against parquet or CSV files, and have the results converted back to
numpy arrays.

Being written in rust, this code has strong assumptions about thread safety and lack of memory leaks.

We lock the GIL to convert the results back to numpy arrays and to run UFDs.

Known limitations:

* timezones are stripped from datetimes as numpy does not support timezone-aware dates
* null value information is discarded for types that do not support them (int and uint) and instead contain the default value of the type (typically a 0)

## How to use it

Simple usage:

```
import datafusion


ctx = datafusion.ExecutionContext()

ctx.register_parquet('t', path)

result = ctx.sql('SELECT (a > 50), COUNT(a) FROM t GROUP BY CAST((a > 10.0) AS int)')
# result is a dictionary with two keys, CAST and COUNT, whose values are numpy arrays.
```

UDF usage:

```
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
