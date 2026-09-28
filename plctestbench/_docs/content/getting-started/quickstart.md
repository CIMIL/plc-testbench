# Quickstart

An experiment combines source tracks, packet-loss simulators, PLC algorithms, and output analysers. The `PLCTestbench` object orchestrates those configured workers.

```python
from plctestbench.models import TestbenchConfiguration
from plctestbench.plc_testbench import PLCTestbench

configuration = TestbenchConfiguration(
    root_folder="/path/to/experiment-data",
    db_ip="127.0.0.1",
    db_port=27017,
    db_username="myUserAdmin",
    db_password="admin",
)

# Configure original tracks, loss simulators, PLC algorithms, and analysers
# as (worker class, settings instance) pairs. See the configuration guide.
testbench = PLCTestbench(
    original_audio_tracks=original_audio_tracks,
    packet_loss_simulators=packet_loss_simulators,
    plc_algorithms=plc_algorithms,
    output_analysers=output_analysers,
    testbench_settings=configuration,
    user=user,
)
testbench.run()
```

!!! note

    The variables in the example are intentionally left as configured worker lists: their available settings depend on the loss model, PLC algorithm, and metric you select. See [configuration](../guides/configuration.md) and the generated [API reference](../reference/index.md) for the concrete classes.

After a run, use `PLCTestbench.plot()` to produce plots of original tracks, loss masks, reconstructed tracks, and analyser results:

```python
testbench.plot(to_file=True, original_tracks=True, output_analyses=True)
```

The repository’s `plctestbench.ipynb` notebook remains a longer worked example.
