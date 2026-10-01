# Configure crossfades in OpenPLC Studio

Crossfades smooth the boundaries between received audio and audio reconstructed by a packet-loss concealment (PLC) algorithm. In OpenPLC Studio, they are configured inside each PLC algorithm selected in the run configurator.

A PLC algorithm exposes two buttons:

- **fade_in** controls the transition from recent reconstructed context into PLC-generated audio at the start of a lost packet.
- **crossfade** controls the transition from the continuing PLC prediction back to received audio when valid packets resume.

Both contain `NoCrossfadeSettings` by default, so no blending occurs until you replace that entry.

## Available crossfade types

| UI option | Curve | Parameters |
| --- | --- | --- |
| `NoCrossfadeSettings` | Disables the transition | None |
| `LinearCrossfadeSettings` | Power curve with exponent 1 | `length`, `type` |
| `QuadraticCrossfadeSettings` | Power curve with exponent 2 | `length`, `type` |
| `CubicCrossfadeSettings` | Power curve with exponent 3 | `length`, `type` |
| `SinusoidalCrossfadeSettings` | Sine curve from 0 to π/2 | `length`, `type` |
| `ManualCrossfadeSettings` | User-selected power or sinusoidal curve | `length`, `function`, `exponent`, `type` |

The UI initializes configurable crossfades with a length of `2.0` milliseconds.

## Crossfade parameters

### `length`

Transition duration in **milliseconds**. A longer transition blends the two signals over more samples. It must be zero or greater; a zero-length transition is effectively disabled.

For `fade_in`, the algorithm initializer requires the numeric `length` value to be no greater than the packet-loss simulator's `packet_size`; otherwise the run cannot start.

### `function`

Available on `ManualCrossfadeSettings`:

- `power` shapes the incoming gain as `x` raised to `exponent`.
- `sinusoidal` shapes it with a sine curve from 0 to π/2.

### `exponent`

Available on `ManualCrossfadeSettings`. It controls the shape of the `power` function and must be zero or greater. Common values are:

- `1` for a linear curve;
- `2` for a quadratic curve;
- `3` for a cubic curve.

The named Linear, Quadratic, and Cubic options provide these presets without exposing `function` or `exponent`.

### `type`

Controls the relationship between incoming and outgoing gain:

- `power` uses complementary squared gains and is the default.
- `amplitude` uses complementary linear gains.

## Configure a full-band transition

1. In the run configurator, add and select a PLC algorithm.
2. Click **fade_in** or **crossfade**.
3. Remove the default `NoCrossfadeSettings` entry with the **×** button.
4. Use the autocomplete to add the desired crossfade type.
5. Select the new list entry to open its parameters.
6. Set its `length` and any curve options.
7. Use **Back** to return to the list, then confirm the popup.

Keep exactly one entry in `fade_in`. Only the first fade-in setting is used by the algorithm.

For a full-band `crossfade`, keep `crossfade_frequencies` empty and keep exactly one entry in the **crossfade** popup.

## Configure a crossfade by frequency band

Only the return `crossfade` can be frequency-dependent. The `fade_in` transition remains full-band.

1. Configure `crossfade_frequencies` on the selected PLC algorithm before editing **crossfade**.
2. Enter crossover frequencies in ascending order. Each value creates another band.
3. Click **crossfade**. The UI automatically keeps one entry per band, appending `NoCrossfadeSettings` when a frequency creates a new band.
4. Replace the entries as needed and configure each one's parameters.
5. Keep the list ordered from the lowest-frequency band to the highest.
6. Set `crossover_order` on the PLC algorithm if the default value of `4` is not appropriate.
7. Confirm the popup and resolve any validation message before starting the run.

For crossovers at `200` and `2000` Hz, the list maps as follows:

| Position in **crossfade** | Frequency range |
| --- | --- |
| 1 | below 200 Hz |
| 2 | 200–2000 Hz |
| 3 | above 2000 Hz |

!!! important "Configure frequencies first"

    Adding or removing `crossfade_frequencies` changes the required number of crossfade entries. The UI preserves entries from the beginning of the list, appends `NoCrossfadeSettings` for new bands, and removes entries from the end when bands disappear. Review the entire list after changing frequencies.

## Frequency and band rules

- `crossfade_frequencies` must be strictly increasing.
- UI validation permits values from 0 to 20,000 Hz.
- For a runnable filter, each crossover must be greater than 0 and lower than half the source track's sample rate.
- The **crossfade** list must contain exactly one more entry than `crossfade_frequencies`.
- `crossover_order` must be at least `1`.

For example, a 16 kHz track requires crossover frequencies below 8 kHz even though the generic UI range extends to 20 kHz.

## Example: three different return transitions

To use a long low-frequency transition, a shorter mid-band transition, and no high-frequency transition:

1. Enter `200` and `2000` in `crossfade_frequencies`.
2. Open **crossfade**.
3. Configure the ordered list as:
   1. `LinearCrossfadeSettings`, length `8`
   2. `CubicCrossfadeSettings`, length `4`
   3. `NoCrossfadeSettings`
4. Confirm the popup.

The first setting applies below 200 Hz, the second from 200 to 2000 Hz, and the third above 2000 Hz.

## Advanced PLC limitation

The common crossfade controls are intentionally hidden for algorithms placed inside `AdvancedPLC` band settings. Configure crossfades on ordinary top-level PLC algorithm entries. See [Configure Advanced PLC in OpenPLC Studio](advanced-plc-platform.md) for the parameters available inside an advanced band.

## Troubleshooting

- **The number of crossfade settings must be one more than the number of crossfade frequencies.** Open **crossfade** and add or remove entries until the count is correct.
- **A newly created band has no transition.** This is expected: the UI fills new positions with `NoCrossfadeSettings`. Replace it explicitly.
- **The wrong curve affects a band.** Check the list order. Crossfade entries map from low frequencies to high frequencies.
- **Fade in length cannot be longer than the packet size.** Reduce `fade_in.length`, or review the simulator's packet size and the source sample rate.
- **The run fails while creating a crossover filter.** Ensure every frequency is greater than 0 and below the source track's Nyquist frequency.
