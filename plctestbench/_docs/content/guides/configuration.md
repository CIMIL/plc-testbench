# Configuration

Workers receive a settings object. A testbench is configured as lists of `(worker class, settings instance)` pairs, making it possible to compare multiple implementations under the same input conditions.

```python
packet_loss_simulators = [
    (BinomialPLS, BinomialPLSSettings(per=0.1)),
    (MetronomePLS, MetronomePLSSettings()),
]

plc_algorithms = [
    (ZerosPLC, ZerosPLCSettings()),
    (LastPacketPLC, LastPacketPLCSettings()),
]
```

## Loss models

- **Binomial** distributes losses uniformly according to the packet error ratio (`per`).
- **Metronome** creates periodic bursts using a period, duration, and offset.
- **Gilbert-Elliott** models bursty loss through transition and packet probabilities for good and bad states.
- **CustomMaskPLS** applies a predefined binary packet-loss mask.

## Crossfades

PLC output can be faded against received audio at the start or end of a lost region. `fade_in` controls the left edge and `crossfade` controls the right edge; both default to no crossfade.

Use `ManualCrossfadeSettings` when selecting the length and curve explicitly, or use convenience settings such as `LinearCrossfadeSettings`, `QuadraticCrossfadeSettings`, `CubicCrossfadeSettings`, and `SinusoidalCrossfadeSettings`.

For a frequency-dependent transition, supply a `MultibandSettings` entry followed by one crossfade setting for each band. The number of crossfade settings must match the number of frequency bands.

See [`plctestbench.settings`](../reference/settings.md) for the complete settings types and validation rules.
