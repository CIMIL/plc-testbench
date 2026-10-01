# Models and metrics

PLCTestbench organizes an experiment into packet-loss simulators, packet-loss concealment (PLC) algorithms, and output analysers. Crossfade settings can be attached to most PLC algorithms to smooth transitions at loss boundaries.

This page is a high-level catalogue of the concrete modules implemented by the project. For constructor arguments, validation rules, return types, and implementation details, use the [API reference](../reference/index.md).

## Packet-loss simulators

Packet-loss simulators divide a track into packets and produce the loss mask used by every PLC algorithm in the same experiment.

| Module | Description |
| --- | --- |
| `BinomialPLS` | Makes an independent Bernoulli loss decision for every packet. It is the simplest model for uniformly distributed random loss at a chosen packet error ratio. |
| `MetronomePLS` | Produces deterministic, periodic bursts. It is useful for repeatable tests where the period, burst duration, and initial offset must be controlled exactly. |
| `GilbertElliotPLS` | Uses a two-state good/bad Markov model. Different transition and reception probabilities in the two states produce correlated, burst-like losses. |
| `CustomMaskPLS` | Reads a binary mask in which each position controls one packet. It is useful for replaying a known pattern or comparing algorithms against exactly the same hand-authored losses. |

The random simulators accept a seed for reproducibility. `packet_size` determines how many audio samples each loss decision covers and is inherited by downstream algorithms and analysers.

See [`plctestbench.loss_simulator`](../reference/loss_simulator.md) for the simulator API.

## PLC algorithms

PLC algorithms receive the original audio with a loss mask and synthesize replacement samples for missing packets.

| Module | Description |
| --- | --- |
| `ZerosPLC` | Replaces missing packets with zeros. This is a useful lower baseline for measuring the benefit of an actual concealment method. |
| `LastPacketPLC` | Repeats the most recent packet. Optional time and amplitude mirroring can reduce obvious repetition, with selectable handling for samples that leave the normalized range. |
| `LowCostPLC` | Finds periodic structure and suitable zero crossings in recent audio, then phase-aligns and repeats waveform periods. It is designed for low delay and low computational overhead and includes its own boundary transitions. |
| `BurgPLC` | Fits a stable autoregressive model to recent valid context with the Burg method and recursively predicts the missing packet. It is most effective when the signal is locally stationary. |
| `VermaPLC` | Runs a neural waveform predictor from an ONNX model. It combines a mel-spectrogram of recent context with the immediately preceding packet and processes channels independently. |
| `PARCnetPLC` | Combines an online autoregressive branch with a causal neural residual predictor. It also predicts an extra tail for transitions into subsequent packets and handles burst losses explicitly. |
| `AdvancedPLC` | Composes other PLC algorithms by frequency band and by linked, left/right, or mid/side channels. Use it when different parts of the signal need different concealment strategies. |

Most algorithms support generic fade-in and return-crossfade settings. `LowCostPLC` and `PARCnetPLC` also perform algorithm-specific transition processing, so their own transition behavior should be considered before adding generic crossfades.

For `AdvancedPLC`, see the [Python overview](advanced-plc.md) or the [OpenPLC Studio configuration guide](advanced-plc-platform.md).

### Optional external PLC

`ExternalPLC` delegates packet processing to the optional `cpp_plc_template` binding. It is implemented in the Python package but is not included in the default OpenPLC Studio module catalogue. Instantiation raises an import error when the external binding is unavailable.

`BurgPLC` likewise depends on the platform-specific `burg-plc` binding and is unavailable on platforms where that binding cannot be imported.

See [`plctestbench.plc_algorithm`](../reference/plc_algorithm.md) for all PLC classes.

## Deep-learning models

`VermaPLC` and `PARCnetPLC` use ONNX Runtime on CPU. Compatible exported models are included in `dl_models/`, so their default configurations do not require a model-conversion step.

- The Verma model expects the packet, context, sample-rate, and mel-spectrogram dimensions used during export. Changing those dimensions requires a compatible ONNX model.
- PARCnet expects its neural model and autoregressive configuration to agree with the selected preset and packet dimensions.

These methods have higher context, model, and compute requirements than the baseline and signal-modeling algorithms. They are most useful when perceptual quality is more important than minimal processing cost.

## Output analysers

Output analysers compare each reconstructed track with its original reference. Some return one value for the whole track; others return frame- or packet-aligned values that can be plotted around individual losses.

### Built-in application analysers

The following analysers are exposed by the default OpenPLC Studio module catalogue and are also available through the Python API.

| Module | Description | Result scope |
| --- | --- | --- |
| `MSECalculator` | Computes mean squared sample error on overlapping windows after normalization. Large errors are penalized more strongly than with MAE; lower values indicate a closer waveform match. | Windowed |
| `MAECalculator` | Computes mean absolute sample error on overlapping windows after normalization. It is less sensitive to isolated large errors than MSE; lower values indicate a closer waveform match. | Windowed |
| `SpectralEnergyCalculator` | Compares the squared magnitude spectra of windowed original and reconstructed audio. It reveals where reconstruction changes spectral energy even when sample-wise errors are hard to interpret. | Windowed, frequency-resolved |
| `PEAQCalculator` | Runs GstPEAQ on the normalized full tracks and reports the Objective Difference Grade and Distortion Index. It is an objective perceptual comparison rather than a direct waveform distance. | Whole track |
| `WindowedPEAQCalculator` | Runs PEAQ on regions centered around loss events and places the results into a packet-aligned series. It helps localize quality degradation that a whole-track score can average away. | Loss-centered, packet-aligned |
| `PerceptualCalculator` | Uses a constant-Q time-frequency representation to estimate the audibility of glitches around packet losses. It is intended specifically for localized PLC artifacts. | Loss-centered, packet-aligned |
| `PLCMOSCalculator` | Uses Microsoft's PLCMOS ONNX models to estimate perceived PLC quality. Depending on the model version it can operate intrusively with the original reference or non-intrusively on reconstructed audio alone. Channel scores are averaged. | Whole track |
| `PESQCalculator` | Computes a PESQ speech-quality score after resampling to 16 kHz and downmixing to mono. Narrowband and wideband modes are supported by the underlying `pesqc2` implementation. | Whole track |

### Additional Python-package analysers

These concrete analysers are implemented in the package but are not included in the default OpenPLC Studio module catalogue.

| Module | Description | Result scope |
| --- | --- | --- |
| `HumanCalculator` | Selects suitable loss-centered stimuli, generates a listening test, and maps mean listener scores back to packet positions. It supports controlled stimulus selection and optional reference and anchor material. | Human test, packet-aligned |

### External requirements

- `PEAQCalculator` and `WindowedPEAQCalculator` invoke the external GstPEAQ `peaq` executable and require its GStreamer plugin environment.
- `HumanCalculator` relies on the listening-test workflow and its webMUSHRA/pyMUSHRA setup.
- `PLCMOSCalculator` uses ONNX Runtime and the PLCMOS model files supplied in `dl_models/`.
- `PESQCalculator` uses the installed `pesqc2` package.

See [`plctestbench.output_analyser`](../reference/output_analyser.md), [`plctestbench.perceptual_metric`](../reference/perceptual_metric.md), and [`plctestbench.plcmos`](../reference/plcmos.md) for details.

## Crossfade settings

Crossfades are configuration modules used by PLC algorithms rather than standalone experiment workers.

| Module | Description |
| --- | --- |
| `NoCrossfadeSettings` | Disables blending at the selected boundary. |
| `ManualCrossfadeSettings` | Exposes the transition length, curve function, exponent, and complementary mixing type. |
| `LinearCrossfadeSettings` | Provides a first-order power-curve preset. |
| `QuadraticCrossfadeSettings` | Provides a second-order power-curve preset. |
| `CubicCrossfadeSettings` | Provides a third-order power-curve preset. |
| `SinusoidalCrossfadeSettings` | Uses a sinusoidal transition curve. |

A PLC algorithm can use one full-band transition or different return crossfades in frequency bands. See [Crossfades in the Python API](crossfades.md) or [Configure crossfades in OpenPLC Studio](crossfades-platform.md) for the relevant workflow.

## Choosing a comparison set

A useful benchmark normally combines complementary modules rather than relying on one score:

1. Include `ZerosPLC` or `LastPacketPLC` as a simple baseline.
2. Add the PLC methods whose latency and compute requirements match the target application.
3. Use MSE or MAE for direct waveform error and a spectral or perceptual analyser for audible structure.
4. Add a whole-track perceptual score when its external dependencies and source material are appropriate.
5. Reuse the same seeded loss simulator or custom mask across every PLC algorithm.

The analyser results describe different properties and should not be treated as interchangeable. In particular, a low sample error does not guarantee that a localized glitch is inaudible, while one whole-track perceptual score can hide behavior at individual losses.
