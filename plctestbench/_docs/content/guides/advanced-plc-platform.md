# Configure Advanced PLC in OpenPLC Studio

`AdvancedPLC` lets one PLC entry combine different concealment algorithms by frequency band and, for stereo audio, by channel representation. Use it when a single algorithm is not the best choice for the whole signal—for example, when low frequencies should use `LastPacketPLC` and higher frequencies should use `BurgPLC`.

This page describes the configuration exposed by the OpenPLC Studio run configurator. For the Python API, see [Advanced PLC](advanced-plc.md).

## What happens during a run

For each configured channel group, `AdvancedPLC`:

1. represents stereo audio as left/right (`dual_mono`) or mid/side (`mid_side`);
2. optionally uses one shared (`linked`) configuration for both channels;
3. splits the signal at the configured crossover frequencies;
4. runs one PLC algorithm on each resulting band, using the run's packet-loss mask;
5. sums the reconstructed bands; and
6. converts mid/side audio back to left/right when necessary.

With no crossover frequencies, there is one full-band algorithm. Each added crossover creates one more band. For example, crossovers at `200` and `2000` Hz produce three bands, ordered in the UI from low to high:

| Position in `band_settings` | Frequency range |
| --- | --- |
| 1 | below 200 Hz |
| 2 | 200–2000 Hz |
| 3 | above 2000 Hz |

## Parameters in the UI

| Parameter | Meaning | UI default |
| --- | --- | --- |
| `band_settings` | The ordered PLC algorithms used for each channel group's bands. Open the **Linked**, **Left**, **Right**, **Mid**, or **Side** button to edit them. | One `ZerosPLC` for left and one for right |
| `frequencies` | Crossover frequencies in hertz for each visible channel group. Values must be in ascending order. | No crossovers for left or right |
| `order` | Order of the crossover filters used to split the bands. Must be at least `1`. | `4` |
| `stereo_image_processing` | Selects `dual_mono` (left/right) or `mid_side` processing. | `dual_mono` |
| `channel_link` | When enabled, replaces the two channel-specific configurations with one **Linked** configuration. | Off |

The number of algorithms must always be one greater than the number of crossover frequencies **for each channel group**. The UI maintains this relationship when frequencies are added or removed: it appends `ZerosPLC` entries for new bands and removes entries from the end when bands are removed.

!!! note "Safe crossover values"

    The UI accepts values from 0 to 20,000 Hz, but a runnable crossover must be greater than 0 and lower than half the source track's sample rate (the Nyquist frequency). For a 16 kHz track, for example, use values below 8 kHz.

## Choose the channel layout first

Configure `channel_link` and `stereo_image_processing` before editing frequencies and algorithms. These controls determine which channel fields and buttons are shown.

### Linked channels

Enable `channel_link` to show one **Linked** frequency field and one **Linked** band-settings button. The same crossover layout and algorithm sequence is applied to both signal channels.

This is the simplest choice when both stereo channels should use the same processing strategy. The selected stereo representation still applies: `dual_mono` processes left/right, while `mid_side` first converts the signal to its common (mid) and difference (side) components.

### Independent left/right channels

Disable `channel_link` and select `dual_mono` to show **Left** and **Right** controls. Each channel can have different crossover frequencies, a different number of bands, and different algorithms.

Use this when the channels need independent treatment and preserving their direct left/right meaning is important.

### Independent mid/side channels

Disable `channel_link` and select `mid_side` to show **Mid** and **Side** controls:

- **Mid** contains the content common to the left and right channels: `(left + right) / 2`.
- **Side** contains their difference: `(left - right) / 2`.

This mode is useful when centered content and stereo-width information need different PLC strategies. After concealment, the application converts the result back to left/right audio.

Changing the channel layout copies the closest existing configuration into the new fields so work is not discarded. Review both frequency lists and all band algorithms after changing `channel_link` or `stereo_image_processing`; copied values may not be the final configuration you want.

## Configure the frequency bands

1. In the run configurator, add **AdvancedPLC** to the PLC algorithm list and select it.
2. Set `channel_link` and `stereo_image_processing` for the desired layout.
3. In `frequencies`, enter or select crossover values for every visible channel group.
4. Keep the values strictly increasing from low to high.
5. Check that the UI created one algorithm entry per resulting band.

The suggestions `100`, `200`, and `2000` are conveniences, not required values. You can type other frequencies that are valid for the input track.

Each channel group's frequency list is independent. For example, left may use `[200, 2000]` (three bands) while right uses `[1000]` (two bands).

## Assign an algorithm to each band

Open the button beside `band_settings` for the channel group you want to edit. The popup lists its algorithms in low-to-high frequency order.

- Use the autocomplete at the top to append a built-in algorithm or installed PLC plugin.
- Remove entries with the **×** button.
- Select an entry to edit that algorithm's own parameters.
- Use **Back** to return from an algorithm's parameters to the band list.
- Confirm the popup when the list and settings are complete.

`AdvancedPLC` itself is not offered as a band algorithm, so advanced configurations cannot be nested recursively.

The common per-algorithm crossfade fields (`crossfade`, `fade_in`, `crossfade_frequencies`, and `crossover_order`) are not shown inside Advanced PLC band settings. Configure the algorithm-specific fields that remain, such as Burg order or context length.

!!! important "Band order matters"

    The list position assigns the frequency band; the algorithm name does not. If the crossovers are `[200, 2000]`, the first entry processes frequencies below 200 Hz, the second processes 200–2000 Hz, and the third processes frequencies above 2000 Hz.

## Example: one linked three-band configuration

To use `LastPacketPLC` below 200 Hz, `BurgPLC` from 200 to 2000 Hz, and `ZerosPLC` above 2000 Hz:

1. Enable `channel_link`.
2. Leave `stereo_image_processing` as `dual_mono` unless mid/side representation is specifically required.
3. Enter `200` and `2000` in **Linked** under `frequencies`.
4. Open **Linked** under `band_settings`.
5. Remove or replace the automatically created entries, then add these algorithms in order:
   1. `LastPacketPLC`
   2. `BurgPLC`
   3. `ZerosPLC`
6. Open `BurgPLC` in the popup and configure its algorithm-specific parameters.
7. Confirm the popup and verify that no validation message remains.

## Validation and troubleshooting

Before starting the run, resolve any message shown above the Advanced PLC settings or inside a band popup.

- **The channel keys must be either linked, left/right or mid/side.** The `band_settings` and `frequencies` layouts do not match the selected channel controls. Toggle `channel_link` or `stereo_image_processing` to make the UI normalize the layout, then review it.
- **Each channel must have one more algorithm than its crossover frequencies.** Open the named channel's band settings and make the list length equal to `frequencies + 1`.
- **Frequency validation error.** Remove non-numeric values, duplicates, or out-of-order values, and keep each crossover within the valid range for the source track.
- **Unexpected algorithm on a band.** Recheck the list order; entries map from the lowest band to the highest band.
- **Unexpected settings after changing stereo mode or linking.** The UI copies an existing channel configuration when changing layouts. Reopen every visible channel and adjust the copied values.
