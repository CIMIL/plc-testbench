# Installation

PLCTestbench supports Python **3.11**. [uv](https://docs.astral.sh/uv/) is the recommended development and installation workflow.

```bash
git clone https://github.com/CIMIL/plc-testbench.git
cd plc-testbench
uv sync --all-groups
```

Run commands in the managed environment with `uv run`, for example:

```bash
uv run pytest test -q
```

## Database

TinyDB is the default database backend and runs inside the Python process, so a standard local installation needs no database server or external application. MongoDB is also supported for deployments that need a separately managed database.

See [Configuration](../guides/configuration.md#database-backends) for how to choose and configure either backend. The [quickstart](quickstart.md) uses TinyDB and includes MongoDB setup as an appendix.

## Optional integrations

Some features need platform-specific or external software:

- **Burg PLC** is a Linux-only git dependency.
- **PEAQ** requires GStreamer and the gstpeaq plugin.
- **Human/listening-test metrics** require webMUSHRA and pyMUSHRA.
- The deep-learning PLC implementations use the ONNX files committed under `dl_models/`; TensorFlow and PyTorch are not runtime requirements.

Read [models and metrics](../guides/models-and-metrics.md) before enabling any of these integrations.

## Documentation tooling

The documentation builder is an opt-in package extra:

```bash
pip install "plctestbench[docs]"
plctestbench-docs build site --strict
plctestbench-docs serve
```
