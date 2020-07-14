## Datafusion with Python

This is a Python library that binds to Apache's Arrow in-memory rust-based query engine [datafusion](https://github.com/apache/arrow/tree/master/rust/datafusion).
It allows you to execute SQL queries against parquet or CSV files, and have the results converted back to
numpy arrays.

Being written in rust, this code has strong assumptions about thread safety and lack of memory leaks.

We lock the GIL to convert the results back to Python, when building numpy arrays.

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

Known issues: 

* timezones are currently stripped form datetimes.

## How to install

We haven't configured CI/CD to publish wheels in pip yet and thus you can only install it in development.
It requires cargo and rust. See below.

## How to develop

This assumes that you have rust and cargo installed. We use the workflow recommended by [pyo3](https://github.com/PyO3/pyo3) and [maturin](https://github.com/PyO3/maturin).

Bootstrap:

```bash
# fetch arrow
git clone git@github.com:apache/arrow.git
# fetch this repo
git clone git@github.com:jorgecarleitao/datafusion-python.git

cd datafusion-python

# prepare development environment (used to build wheel / install in development)
python -m venv venv
venv/bin/pip install maturin==0.8.2 toml==0.10.1

# used for testing
venv/bin/python install pyarrow==0.17.1
```

Whenever rust code changes (your changes or via git pull):

```bash
venv/bin/maturin develop
venv/bin/python -m unittest discover tests
```
