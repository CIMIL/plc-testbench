# Develop a custom PLC plugin

A plugin adds a packet-loss concealment (PLC) algorithm without modifying the
`plctestbench` package. The platform discovers a plugin from one Python file,
uses a YAML manifest in that file to build the configuration UI, and loads the
Python classes when a run is created.

!!! warning "Plugins are trusted code"

    A worker imports and executes plugin Python with the worker's permissions.
    Install only plugins that you trust. The manifest scanner does not provide a
    sandbox.

## The plugin contract

For an algorithm named `<Name>`, all four names below must agree:

| Part | Required name |
| --- | --- |
| Manifest `name` | `<Name>` |
| File | `<Name>Algorithm.py` |
| Settings class | `<Name>Settings` |
| Algorithm class | `<Name>` |

For example, the repository sample uses `MyPLC`,
`PLUGINS/MyPLCAlgorithm.py`, `MyPLCSettings`, and `MyPLC`.

A plugin file has three responsibilities:

1. **Describe its UI** in the module docstring. The backend parses this YAML
   without importing the plugin.
2. **Construct and validate settings** in a subclass of `PLCSettings`.
3. **Produce replacement audio** in a subclass of `PLCAlgorithm`.

Keeping the manifest and constructor in the same file makes the plugin
portable. It also creates two contracts that must stay synchronized: every
manifest setting name must be accepted by the settings constructor, and its
YAML value must have the type that the constructor expects.

## Start from the sample plugin

Create `PLUGINS/MyPLCAlgorithm.py` with the following structure. This is the
minimal sample included in the repository:

```python
"""
- name: MyPLC
  settings:
    - name: crossfade
      type: list_CrossfadeSettings
      default: []
      values: null
    - name: fade_in
      type: list_CrossfadeSettings
      default: []
      values: null
    - name: crossfade_frequencies
      type: list_int
      default: null
      values: null
    - name: crossover_order
      type: int
      default: null
      values: null
"""

import numpy as np
from plctestbench.plc_algorithm import PLCAlgorithm
from plctestbench.settings import CrossfadeSettings, PLCSettings


class MyPLCSettings(PLCSettings):
    def __init__(
        self,
        crossfade: list[CrossfadeSettings] = None,
        fade_in: list[CrossfadeSettings] = None,
        crossfade_frequencies: list[int] = None,
        crossover_order: int = None,
    ) -> None:
        super().__init__(
            crossfade,
            fade_in,
            crossfade_frequencies,
            crossover_order,
        )
        self.__validate__()


class MyPLC(PLCAlgorithm):
    def __init__(self, settings: MyPLCSettings):
        super().__init__(settings)

    def _predict(self, buffer: np.ndarray) -> np.ndarray:
        return np.zeros(np.shape(buffer))
```

`MyPLC` is intentionally equivalent to zero filling. It is useful as a wiring
check: if it appears in the UI and completes a run, discovery, settings
hydration, dynamic loading, and audio processing are all connected correctly.

The four manifest fields expose the processing already supplied by
`PLCAlgorithm`:

- `crossfade` blends a prediction into the first valid packet after a loss.
- `fade_in` blends the previous reconstructed context into a lost packet.
- `crossfade_frequencies` permits a different crossfade per frequency band.
- `crossover_order` configures the filters used for multiband crossfading.

Calling `PLCSettings.__init__` is therefore important even when the algorithm
has no custom options. Empty or `None` values are converted to safe defaults:
no crossfade, no fade-in, no crossover frequencies, and crossover order `4`.

!!! note

    The checked-in sample also contains `value` keys in its manifest. They are
    not required for a new plugin; the platform creates current values from
    `default` when the user configures a run.

## How a packet is processed

The base class runs the track packet by packet. Its normal lifecycle is:

```text
_prepare_to_play()                         # once per track
    |
    +-- _tick(buffer, is_valid)            # once per packet
          +-- _a_priori(buffer, is_valid)
          +-- valid packet: _crossfade(...)
          |      `-- may call _predict(...) while a crossfade is active
          +-- lost packet:  _predict(...), _fade_in(...)
          `-- _a_posteriori(buffer, is_valid)
```

The most useful extension points are:

| Method | When to override it |
| --- | --- |
| `_predict(buffer)` | Almost every plugin. Return one concealed packet. |
| `_prepare_to_play()` | Allocate or reset model state once the channel count is known. Call `super()` if the plugin uses `self.context`. |
| `_a_priori(buffer, is_valid)` | Pre-process every packet before concealment. |
| `_a_posteriori(buffer, is_valid)` | Update custom state after every packet. Call `super()` to keep the base context current. |
| `_tick(buffer, is_valid)` | Only when replacing the complete valid/lost-packet and crossfade lifecycle. |
| `run(...)` | Rarely. Override only if the algorithm cannot operate packet by packet. |

The important runtime values are:

- `buffer.shape == (packet_size, number_of_channels)`;
- `self.packet_size` is measured in samples;
- `self.n_channels` is available after `_prepare_to_play` begins;
- `self.context` contains the reconstructed history managed by the base class;
- `settings.get("fs")` is the source sample rate; and
- `settings.get("packet_size")` is inherited from the selected loss simulator.

`fs`, `packet_size`, and the progress monitor are inherited through the
experiment graph. They are not constructor arguments in `MyPLCSettings`.
Consequently, access them in the algorithm constructor or processing methods,
not while initially constructing the settings object outside a testbench.

!!! danger "Do not use the lost input as ground truth"

    The `buffer` passed to `_predict` has the right shape, but a simulated run
    may still contain the original samples for the packet marked as lost. A
    real concealment algorithm must predict from `self.context` or its own
    state, not from the contents of `buffer`. Reading the lost buffer would
    leak the reference audio and produce unrealistically good results.

Return exactly one packet with the same two-dimensional shape. Supporting all
columns rather than assuming mono makes the plugin work for both mono and
stereo tracks. Prefer finite floating-point output and keep audio in the
expected `[-1, 1]` range, or explicitly document and handle any clipping.

## Add an algorithm-specific setting

The following example holds the most recent reconstructed sample and applies
an exponential decay. It demonstrates a custom parameter, UI validation,
runtime validation, multichannel output, and behavior across consecutive lost
packets.

Save it as `PLUGINS/DecayPLCAlgorithm.py`:

```python
"""
- name: DecayPLC
  settings:
    - name: crossfade
      type: list_CrossfadeSettings
      default: []
      values: null
    - name: fade_in
      type: list_CrossfadeSettings
      default: []
      values: null
    - name: crossfade_frequencies
      type: list_int
      default: []
      values: null
      validation:
        item:
          min: 0
          max: 20000
        sorted: ascending
    - name: crossover_order
      type: int
      default: 4
      values: null
      validation:
        min: 1
    - name: decay
      type: float
      default: 0.98
      values: null
      validation:
        min: 0.0
        max: 1.0
        step: 0.01
"""

import numpy as np
from plctestbench.plc_algorithm import PLCAlgorithm
from plctestbench.settings import CrossfadeSettings, PLCSettings


class DecayPLCSettings(PLCSettings):
    def __init__(
        self,
        crossfade: list[CrossfadeSettings] | CrossfadeSettings = None,
        fade_in: list[CrossfadeSettings] | CrossfadeSettings = None,
        crossfade_frequencies: list[int] = None,
        crossover_order: int = None,
        decay: float = 0.98,
    ) -> None:
        super().__init__(
            crossfade,
            fade_in,
            crossfade_frequencies,
            crossover_order,
        )
        self.settings["decay"] = decay
        self.__validate__()

    def __validate__(self) -> None:
        # Preserve all invariants imposed by PLCSettings before checking ours.
        super().__validate__()
        self.assert_setting_is_number_in_range("decay", 0.0, 1.0)


class DecayPLC(PLCAlgorithm):
    def __init__(self, settings: DecayPLCSettings) -> None:
        super().__init__(settings)
        self.decay = settings.get("decay")

    def _predict(self, buffer: np.ndarray) -> np.ndarray:
        # self.context includes earlier predictions, so a burst continues to
        # decay instead of restarting from the last originally received packet.
        last_sample = self.context[-1, :]
        envelope = self.decay ** np.arange(1, self.packet_size + 1)
        prediction = envelope[:, np.newaxis] * last_sample[np.newaxis, :]
        return prediction.astype(buffer.dtype, copy=False)
```

The manifest's `validation` gives immediate feedback in the UI and the settings
class repeats the invariant at runtime. Both are intentional: UI validation
improves usability, while Python validation protects direct API callers,
restored configurations, and future clients.

`super().__validate__()` preserves checks such as increasing crossover
frequencies, one crossfade per band, and a positive crossover order. Omitting
it can allow an invalid base configuration into the audio worker.

## Write the manifest

The module docstring must be the first Python statement and must parse as a YAML
list containing exactly one object. The backend checks the manifest, filename,
Python syntax, and the presence of both top-level classes without executing the
file. Imports and constructor behavior are checked only when a worker loads the
plugin for a run.

A setting has this shape:

```yaml
- name: setting_name
  type: float
  default: 0.5
  values: null
  validation:
    min: 0.0
    max: 1.0
```

The configurator currently renders these types:

| Type | Python value | UI |
| --- | --- | --- |
| `int` | `int` | Integer input |
| `float` | `float` | Decimal input |
| `str` | `str` | Text input |
| `bool` | `bool` | Checkbox |
| `Enum` | Usually a string passed to the constructor | Select using `values` |
| `list_int` | `list[int]` | Multiple integer input |
| `list_CrossfadeSettings` | `list[CrossfadeSettings]` after backend hydration | Crossfade editor |

`dict_str_list_int` and `dict_str_list_PLCSettings` are specialized controls
used by `AdvancedPLC`; they are normally unnecessary for a single algorithm
plugin.

For an enum-like option, declare the choices and convert the selected string in
the settings constructor if the algorithm uses a Python `Enum`:

```yaml
- name: mode
  type: Enum
  default: conservative
  values: [conservative, aggressive]
```

```python
from enum import Enum


class Mode(Enum):
    conservative = "conservative"
    aggressive = "aggressive"

# In the settings constructor:
self.settings["mode"] = Mode(mode)
```

Available field validation includes:

- numbers: `min`, `max`, `exclusive_min`, `exclusive_max`, and `step`;
- strings: `min_length`, `max_length`, and `pattern`;
- lists: `min_items`, `max_items`, `unique`, `sorted`, and nested `item`
  validation.

Use YAML booleans (`true`/`false`), lists, numbers, and `null` deliberately;
quoting them changes their type. Keep runtime checks in `__validate__` even
when the same rule appears in the manifest. In particular, cross-field
invariants belong in Python settings validation so they are enforced for every
entry path.

## Implement settings safely

Follow this order in a custom settings constructor:

```python
class ExampleSettings(PLCSettings):
    def __init__(self, crossfade=None, fade_in=None,
                 crossfade_frequencies=None, crossover_order=None,
                 custom_value=10):
        super().__init__(
            crossfade,
            fade_in,
            crossfade_frequencies,
            crossover_order,
        )
        self.settings["custom_value"] = custom_value
        self.__validate__()

    def __validate__(self):
        super().__validate__()
        self.assert_setting_is_number_in_range("custom_value", min_value=1)
```

The order matters:

1. The parent initializes the built-in PLC controls.
2. The subclass stores every custom value in `self.settings` so that hashing,
   persistence, and `settings.get(...)` see it.
3. Validation runs only after all required keys exist.

Settings participate in node hashes and result caching. Put every value that
can change output into the settings object. Do not hide behavior-changing
configuration in mutable module globals or environment variables, because two
runs could then share a cache key while producing different audio.

Avoid mutable list or dictionary defaults in Python signatures. Use `None` and
allocate the value in the constructor. The platform can create many settings
instances in one process.

## Install the plugin

The directory is selected with `PLUGINS_DIRECTORY`. In this repository's
Docker Compose development setup, `dev.env` supplies a host path and
`compose.yaml` mounts it into both the backend API and worker containers.

```dotenv
PLUGINS_DIRECTORY=/absolute/path/to/plc-testbench-platform/PLUGINS
```

After adding or changing a file, restart both services so the inventory and
job workers use the same code:

```bash
docker compose restart backend workers
```

The API scanner rereads the directory when refreshed, but restarting remains
important after code changes because worker processes may already have imported
a previous module version.

In a deployed OpenPLC Studio installation, copy the file into its configured
plugins folder (normally `./plugins`) and restart the backend and workers. The
folder and files must be readable by the container user (uid `1000` in the
provided deployment).

A plugin may import third-party packages, but those packages must already be
installed in the backend/worker image. A host virtual environment is not
visible inside the containers. Add dependencies to the image and rebuild it;
do not install packages dynamically from plugin code.

## Verify discovery and execution

Use progressively stronger checks.

### 1. Check Python syntax

```bash
python -m py_compile PLUGINS/MyPLCAlgorithm.py
```

### 2. Check the platform inventory

With the development stack running, open **Settings → Plugins** and select
**Refresh**. An available plugin shows its parsed setting names, types, and
defaults. Invalid plugins remain visible with a diagnostic.

The same inventory is available through the reverse proxy:

```bash
curl --fail http://localhost:5001/api/plugins
```

The inventory check does not import the module. It can therefore pass even if
a third-party import is missing or the algorithm fails during construction.

### 3. Check dynamic loading

From a Python environment containing `plctestbench`:

```bash
PLUGINS_DIRECTORY="$PWD/PLUGINS" \
PYTHONPATH="$PWD/plc-testbench" \
python - <<'PY'
from plctestbench.utils import get_class

print(get_class("MyPLC"))
print(get_class("MyPLCSettings"))
PY
```

This catches import errors and naming mismatches. It does not provide inherited
`fs` and `packet_size`, so instantiate the algorithm through a testbench run
rather than directly for the final check.

### 4. Run representative audio

Add the plugin under **PLC algorithms** in the run configurator and test:

- one isolated loss and a burst of consecutive losses;
- mono and stereo tracks;
- more than one packet size and sample rate;
- no crossfade and a non-zero crossfade; and
- silence, full-scale samples, and very short tracks.

Check that output has the same sample count and channel count as input, contains
no NaN or infinite values, and is deterministic for identical settings. Follow
worker failures with:

```bash
docker compose logs -f backend workers
```

## Use a plugin from Python

The same loader used by the platform can resolve plugin classes for a direct
`PLCTestbench` configuration:

```python
import os
from plctestbench.utils import get_class

os.environ["PLUGINS_DIRECTORY"] = "/absolute/path/to/PLUGINS"

MyPLC = get_class("MyPLC")
MyPLCSettings = get_class("MyPLCSettings")

plc_algorithms = [
    (MyPLC, MyPLCSettings()),
]
```

Pass `plc_algorithms` to `PLCTestbench` together with the source tracks, loss
simulators, analysers, and testbench configuration. The testbench then injects
the sample rate, packet size, and progress monitor through the normal node
hierarchy.

## Common failures

| Symptom | Likely cause and fix |
| --- | --- |
| Plugin does not appear | Confirm `PLUGINS_DIRECTORY`, a readable `.py` file, and refresh the Plugins page. Restart backend and workers after changes. |
| `Expected filename 'XAlgorithm.py'` | Make the manifest `name` and filename match exactly, including capitalization. |
| `Missing required class(es)` | Define `<Name>` and `<Name>Settings` as top-level classes, not nested classes or aliases. |
| Available in inventory, but run import fails | The inventory parses the AST without executing imports. Install dependencies in the worker image and inspect worker logs. |
| `unexpected keyword argument` | A manifest setting is missing from the settings constructor, or was renamed in only one place. |
| `KeyError: packet_size` or `KeyError: fs` in a standalone test | Those values are inherited during testbench graph construction. Test through `PLCTestbench`, or explicitly build equivalent inherited settings in a unit test. |
| Broadcasting or assignment error | `_predict` returned the wrong shape. Return `(self.packet_size, self.n_channels)`. |
| Stereo channels sound identical | The implementation selected one channel and broadcast it. Index context by channel or process each column independently. |
| Results do not change after editing code | Restart both backend and worker processes to clear imported modules. |
| Configurator accepts a value but the worker fails | Keep manifest defaults/types aligned with the Python signature and repeat authoritative checks in `__validate__`. |

## Design checklist

Before distributing a plugin, verify that:

- the manifest name, filename, and two class names match;
- manifest defaults and constructor defaults represent the same behavior;
- `PLCSettings.__init__` and `super().__validate__()` are called;
- every behavior-changing option is stored in `self.settings`;
- `_predict` never reads simulated lost samples as reference audio;
- output shape, channel handling, dtype, and numeric range are controlled;
- per-track state is reset in `_prepare_to_play`;
- third-party dependencies exist in both API and worker images;
- isolated and burst losses are tested with crossfading enabled and disabled;
  and
- backend and workers are restarted together when the file changes.

See [Configuration](configuration.md) for worker configuration and
[Advanced PLC](advanced-plc.md) for composing algorithms across bands and
channels. The [`plctestbench.plc_algorithm`](../reference/plc_algorithm.md) and
[`plctestbench.settings`](../reference/settings.md) API pages document the base
classes used above.
