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

## Database backends

PLCTestbench stores run records, serialized worker configurations, and data-tree metadata through the database manager. Audio files and generated artifacts are stored separately under `TestbenchConfiguration.root_folder`.

Select the backend with `TestbenchConfiguration.db_platform`:

| Backend | Best suited to | External service | Configuration |
| --- | --- | --- | --- |
| `DBPlatform.TINYDB` | Local experiments, notebooks, tests, and development | None | No database connection fields |
| `DBPlatform.MONGODB` | Managed or shared deployments and data that must live in an explicitly operated database service | MongoDB | Host, port, username, and password |

### TinyDB

TinyDB is the default. It is embedded in the PLCTestbench process and stores records in local JSON files, so it works immediately after installing the Python package.

```python
from plctestbench.models import TestbenchConfiguration

configuration = TestbenchConfiguration(
    root_folder="/path/to/experiment-data",
)
```

The equivalent explicit configuration is:

```python
from plctestbench.models import DBPlatform, TestbenchConfiguration

configuration = TestbenchConfiguration(
    root_folder="/path/to/experiment-data",
    db_platform=DBPlatform.TINYDB,
)
```

The backend creates its JSON files automatically in the operating system's temporary-file directory. It does not use `db_ip`, `db_port`, `db_username`, or `db_password`. The artifact `root_folder` does not change the TinyDB file location.

TinyDB avoids all service setup and is the recommended starting point. Its automatically allocated temporary files are not a replacement for a deliberately managed, shared database deployment; choose MongoDB when that behavior is required.

### MongoDB

Select MongoDB explicitly and supply all four connection fields:

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

| Field | Purpose |
| --- | --- |
| `db_ip` | Host name or IP address of the MongoDB server |
| `db_port` | MongoDB TCP port; normally `27017` |
| `db_username` | Username passed to the MongoDB client |
| `db_password` | Password passed to the MongoDB client |

The database manager opens a MongoDB database associated with the active user's email. When no `User` is supplied to `PLCTestbench`, the package's default user is used.

The standalone quickstart includes a Docker example for starting a local MongoDB server. The same configuration fields can point to a hosted service, provided it is reachable and accepts the supplied credentials. MongoDB initialization fails if the server is unavailable or required connection values are missing.
