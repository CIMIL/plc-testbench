# Models and metrics

## PLC algorithms

PLCTestbench includes baseline implementations such as zeros and last-packet replacement, low-cost concealment, Burg bindings, deep-learning PLC, external C++ bindings, and `AdvancedPLC` compositions.

## Deep-learning models

`VermaPLC` and `PARCnetPLC` run through ONNX Runtime on CPU. Their exported model files are committed in `dl_models/`, so no conversion step is required for normal use.

`VermaPLC` expects its exported packet and spectrogram dimensions. Changing packet size or its feature-extraction settings requires a compatible model export.

## Output analysers

The project provides MSE, MAE, spectral-energy, PEAQ, PESQ, PLCMOS, and human/listening-test analysers. External metrics can require binaries or services beyond the Python package.

- Install GStreamer plus gstpeaq for PEAQ.
- Install the required PESQ tooling for PESQ.
- Configure webMUSHRA and pyMUSHRA for listening tests.

Use the [`plctestbench.output_analyser`](../reference/output_analyser.md) and [`plctestbench.plcmos`](../reference/plcmos.md) references to inspect options and return types.
