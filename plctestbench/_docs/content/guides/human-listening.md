# Human listening tests (MUSHRA)

`HumanCalculator` now **prepares** reusable listening-test material. It does not
start a server or synchronously read listener scores. MUSHRA measures quality
from 0 (bad) to 100 (excellent); legacy SPLAT audibility scores have the opposite
meaning and must not be imported as MUSHRA quality scores.

## Platform run limits

The platform permits **at most one HumanCalculator instance per run**. A run
containing it must have **exactly one loss-simulator instance**, even if multiple
instances use the same settings. That run produces one study covering all
selected tracks and PLC conditions. Automated-only runs may still use multiple
simulators. The configurator blocks incompatible additions; API validation
rejects incompatible creation, imports, and retries instead of dropping modules.
These platform limits do not change the standalone bundle APIs below.

## Prepare and export independently of the platform

After running a testbench containing `HumanCalculator`, all automated nodes
finish and its core run status is `AWAITING_INPUT`. Each human output artifact
contains a prepared `HumanCalculatorData`, not a numeric packet-error array.
Retrieve those bundles from the output nodes, then explicitly group them by
HumanCalculator configuration and loss-simulator instance. Do not pool different
loss masks into the same comparison set.

```python
from plctestbench.human_listening import assemble_study

# bundles: prepared HumanCalculatorData from every track/PLC condition
# for ONE HumanCalculator configuration and ONE loss-simulator instance.
study = assemble_study(bundles, study_id="my-study-v1")
config_path = study.export("listening-material/my-study-v1")
```

Export writes `config.yaml`, `definition.json`, and WAV files. The destination
must not already exist, so accidental replacement of published material fails.
The YAML defines MUSHRA trials; the hosting application must supply instructions,
consent, navigation/completion behavior and a collection adapter. No collector
URL, PHP service, project-specific consent, or participant questionnaire is
silently included. The testbench requires neither webMUSHRA nor pyMUSHRA to
prepare or export material.

### Selection and audio

**Use full track** (`use_full_track=True`) is enabled by default. Each input
track is one stimulus, comparing its complete original and PLC versions, with
one trial per track per repetition. This suits pre-processed listening material
and overall track scores. Full-track mode bypasses `stimulus_length`,
`single_loss_per_stimulus`, `trials_per_track`, `choose_seed` and
`loudness_threshold_db`; the configurator hides those controls without discarding
their values. Tracks shorter than one second and tracks without loss events are
accepted. Source bounds cover the entire track; loss bounds describe the envelope
of all loss events, or the empty interval `[0, 0)` for loss-free material.
Repetitions (`iterations`, default 1) and shared audio processing still apply.

Disable **Use full track** for loss-centered excerpt selection. Excerpt defaults
are 3 seconds, up to 20 distinct trials per track, selection seed 1, and a −10 dB
reference-relative RMS threshold. `single_loss_per_stimulus` defaults to `False`.
Each contiguous loss burst counts as one event. Selection requires complete in-bounds clips;
with `single_loss_per_stimulus=True`, no other event may intersect the window.
If fewer excerpts qualify, the bundle contains all eligible excerpts and a
warning. A track with no eligible excerpt fails rather than disappearing from
the study.

Every PLC condition uses the same reference-based selection. Source arrays are
not modified. Mono/stereo material is consistently resampled to 48 kHz, given
matching fades of at most 10 ms, and never independently level-normalized.
References come from the original excerpt. A deterministic 513-tap Hann-window
3.5 kHz low-pass anchor is generated with centered filtering and reflected
boundary extension. webMUSHRA's built-in hidden-reference insertion is used;
browser-generated anchors are disabled in the exported YAML. At most ten PLC
conditions plus a hidden reference and anchor are supported.

Browser/device resampling and uncontrolled listening environments still matter;
using webMUSHRA is not by itself evidence of full ITU or laboratory compliance.

## Explicitly import and finalize results

Collection, consent, session completion, exclusions, and run-specific response
storage belong to the host application. Pass only the sessions it has selected
for inclusion to the standalone finalizer:

```python
records = [
    {
        "session_id": "participant-1",
        "trial_id": study.definition.trials[0].id,
        "condition_id": bundles[0].condition_id,
        "score": 0,
        "answered": True,
    },
]
finalized_bundles = study.finalize(records)
# human_nodes are the exclusively owned standalone output nodes corresponding
# to bundles, in the same order. Use a new artifact path for each result.
for index, (node, result) in enumerate(zip(human_nodes, finalized_bundles)):
    node.finalize_human_result(result, f"my-study-condition-{index}.pickle")
```

`finalize` checks response membership, finite score bounds, explicit answered
state, and duplicate `(session_id, trial_id, condition_id)` identities. Trial IDs
encode repetitions. It does not infer participant consent or completion from a
response row; that check is the host's responsibility. Empty input is permitted
and yields missing (`None`) summaries rather than fabricated zeros.

The returned data retains raw participant/trial/condition responses and exposes
JSON-safe excerpt summaries through `to_dict()`. A real score of zero remains
zero. Unobserved excerpts remain `None`. An excerpt rating is not a score at
every packet covered by the excerpt. Standalone plots show observed excerpt
centers, not a continuous packet-error trace. Participant-aware descriptive and
inferential analyses belong to the platform analysis layer; these basic excerpt
means are not ANOVA or a replacement for that layer.

`OutputAnalysisNode.finalize_human_result` updates both the result artifact and
its database cache reference/hash. Merely overwriting a pickle without updating
the cache would invalidate it and regenerate preparation on the next run. After
finalizing every human node, rerunning the standalone cached testbench can reach
`COMPLETED` without repeating reconstruction. Platforms that share cached nodes
between runs must keep participant results in separate run-scoped storage rather
than using this exclusively-owned standalone-node helper.

## Legacy settings and migration

Use `trials_per_track` instead of `pages` and `stimuli_per_page`. If legacy count
arguments are supplied, their product is migrated to the new setting (missing
legacy partners use their old defaults, 2 and 10). A conflicting explicit
`trials_per_track` fails. Legacy `reference=None` and `anchor=None` are accepted;
non-null external paths fail with migration guidance.

The former `ListeningTest(settings)` local-database SPLAT workflow raises an
explicit unsupported-workflow error. It no longer reads `webmushra.json` or
expects an installed `listening_tests/webmushra` directory.

See [`HumanCalculator`](../reference/output_analyser.md) and
[`HumanCalculatorSettings`](../reference/settings.md) for API settings and
validation details.
