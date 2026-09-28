# PLCTestbench

<div class="hero" markdown>

# Measure packet-loss concealment with confidence

PLCTestbench is a Python framework for comparing packet-loss simulators, audio PLC algorithms, and quality metrics against the same source material.

[Get started](getting-started/installation.md){ .md-button .md-button--primary }
[Browse the API](reference/index.md){ .md-button }

</div>

<div class="mermaid">
flowchart LR
    A[Source audio] --> B[Packet-loss simulator]
    B --> C[PLC algorithm]
    C --> D[Output analyser]
    D --> E[Results and plots]
</div>

## What it includes

<div class="grid cards" markdown>

- :material-waveform: **Loss simulation**

  ---

  Compare binomial, metronome, Gilbert-Elliott, and predefined-mask packet-loss patterns.

- :material-auto-fix: **PLC algorithms**

  ---

  Benchmark simple, low-cost, Burg, deep-learning, external, and advanced multiband algorithms.

- :material-chart-line: **Quality analysis**

  ---

  Evaluate reconstructed audio with objective metrics and listening-test workflows.

- :material-cog-outline: **Composable experiments**

  ---

  Configure tracks, loss models, algorithms, and analysers independently for reproducible runs.

</div>

## Documentation at a glance

- Start with [installation](getting-started/installation.md), then follow the [quickstart](getting-started/quickstart.md).
- Use the [guides](guides/configuration.md) for settings, advanced PLC configurations, metrics, and integrations.
- The [API reference](reference/index.md) is generated from source signatures and docstrings at build time.
- Need these docs inside a service? See [serving in an application](guides/serving.md).
