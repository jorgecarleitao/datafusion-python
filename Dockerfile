FROM rust:1.44.1-buster

# build Rust dependencies
COPY Cargo.toml Cargo.toml
COPY rust-toolchain rust-toolchain

RUN mkdir src && touch src/lib.rs
RUN cargo build

# install Python stuff
RUN apt-get update && apt install -y python3-venv python3-pip

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install maturin==0.8.2 toml==0.10.1

RUN pip install pyarrow==1.0.0

RUN rm -rf src
COPY src src
COPY README.md README.md

RUN maturin develop

# copy tests to run as last, since they do not affect anything
COPY tests tests

CMD python -m unittest discover tests
