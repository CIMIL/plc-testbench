# Quickstart

An experiment combines source tracks, packet-loss simulators, PLC algorithms, and output analysers. The `PLCTestbench` object orchestrates those configured workers.

```python
from plctestbench.models import DBPlatform, TestbenchConfiguration
from plctestbench.plc_testbench import PLCTestbench

configuration = TestbenchConfiguration(
    root_folder="/path/to/experiment-data",
    db_platform=DBPlatform.TINYDB,
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

## Use MongoDB

TinyDB is sufficient for the quickstart and requires no external service. To run the same experiment with MongoDB, first start or obtain access to a MongoDB server. A local Docker instance can be started with:

```bash
docker run -d -p 27017:27017 --name mongodb \
  -e MONGO_INITDB_ROOT_USERNAME=myUserAdmin \
  -e MONGO_INITDB_ROOT_PASSWORD=admin \
  mongo:6.0.8
```

Replace the quickstart configuration with:

```python
from plctestbench.models import DBPlatform, TestbenchConfiguration

configuration = TestbenchConfiguration(
    root_folder="/path/to/experiment-data",
    db_platform=DBPlatform.MONGODB,
    db_ip="127.0.0.1",
    db_port=27017,
    db_username="myUserAdmin",
    db_password="admin",
)
```

See [Database backends](../guides/configuration.md#database-backends) for backend behavior and configuration guidance.
