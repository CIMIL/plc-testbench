# Advanced PLC

`AdvancedPLC` composes independent PLC algorithms across frequency bands and audio channels.

- **Multiband PLC** applies a different algorithm to each frequency band.
- **Spatial PLC** applies different algorithms to left/right or mid/side channels.

The `frequencies` mapping names each channel and lists its crossover frequencies. The corresponding `band_settings` mapping supplies one `(algorithm, settings)` pair per band.

```python
frequencies = {"mid": [200, 2000], "side": [1000]}
band_settings = {
    "mid": [
        (ZerosPLC, ZerosPLCSettings()),
        (BurgPLC, BurgPLCSettings(order=512)),
        (BurgPLC, BurgPLCSettings(order=256)),
    ],
    "side": [
        (LastPacketPLC, LastPacketPLCSettings()),
        (BurgPLC, BurgPLCSettings(order=256)),
    ],
}
```

Set `channel_link=False` when channels need their own configuration. Use `left`/`right` or `mid`/`side` keys to match the selected stereo representation. With `channel_link=True`, provide a shared `linked` configuration.

The API references for [`plctestbench.plc_algorithm`](../reference/plc_algorithm.md), [`plctestbench.crossfade`](../reference/crossfade.md), and [`plctestbench.spatial`](../reference/spatial.md) describe the implementation interfaces.
