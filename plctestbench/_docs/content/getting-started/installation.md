# Installation

PLCTestbench supports Python **3.11**. [uv](https://docs.astral.sh/uv/) is the recommended development and installation workflow.

```bash
git clone https://github.com/LucaVignati/plc-testbench.git
cd plc-testbench
uv sync --all-groups
```

Run commands in the managed environment with `uv run`, for example:

```bash
uv run pytest test -q
```

## Database

A testbench run stores results in MongoDB by default. Start a local instance or configure a hosted MongoDB service before running an experiment:

```bash
docker run -d -p 27017:27017 --name mongodb \
  -e MONGO_INITDB_ROOT_USERNAME=myUserAdmin \
  -e MONGO_INITDB_ROOT_PASSWORD=admin \
  mongo:6.0.8
```

`TestbenchConfiguration` also exposes a TinyDB platform for configurations that do not use MongoDB.

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
