# Chunked (Streaming) File Processing for PLCTestbench

## Summary

A 20-minute stereo WAV currently OOMs because the tool materialises the entire track as float32
arrays several times over: `AudioFile.load()` reads the whole file, `PLCAlgorithm.run()` allocates a
full `reconstructed_track` **plus** a `zero_pad` copy, and whole-file `OutputAnalyser`s build
full-length frame stacks and resampled copies. On top of that, nothing checks available memory
before starting.

This plan turns file processing into a **bounded-memory, chunk-by-chunk pipeline** while keeping the
public API, node tree, file layout, database documents and numeric output **exactly as they are
today**. Chunking is invisible plumbing: `Node.get_data()` and every `Worker.run(...)` signature stay
unchanged, and callers (notebook, React UI backend, `PlotManager`) need no edits.

Three guarantees were requested and are treated as acceptance criteria:

1. **Never OOM.** A pre-flight memory estimate runs before any heavy work; if the selected run graph
   cannot fit in the budget, the run aborts with an actionable message naming the offending
   track/stage and the estimated vs. allowed MB. A runtime safety net backs it up.
2. **API unchanged.** No signature, class name, setting key, file path, extension, hash or DB
   document shape changes.
3. **Output unchanged.** For every built-in PLC algorithm and streamable metric, chunked results are
   numerically identical to today's single-pass results (asserted by a golden-file test).

## Scope

In scope (all become chunk-aware):

- `FileWrapper` / `AudioFile`: file-backed, seekable sources plus streamed writing.
- `OriginalTrackNode`, `LostSamplesMaskNode`, `ReconstructedTrackNode`: drive chunk loops.
- `PacketLossSimulator` and all subclasses: chunk-accumulated mask generation.
- All built-in PLC algorithms: `ZerosPLC`, `LastPacketPLC`, `BurgPLC`, `LowCostPLC`, `VermaPLC`,
  `PARCnetPLC`, `ExternalPLC`, and `AdvancedPLC`.
- Streamable metrics: `MSECalculator`, `MAECalculator`, `SpectralEnergyCalculator`.

Out of scope (unchanged code paths, guarded by the memory pre-flight instead): `PEAQCalculator`,
`WindowedPEAQCalculator`, `PESQCalculator`, `PLCMOSCalculator`, `HumanCalculator`/`PerceptualCalculator`
(already segment-based and cheap), and `PlotManager` (already subsamples).

## Design overview

### 1. Chunked audio source (`file_wrapper.py`)

Add a file-backed accessor to `AudioFile` that never holds the full track:

- `AudioFile.load()` changes from one eager `file.read()` to **metadata-only**: open, read
  `samplerate`/`channels`/`subtype`/`endian`/`format`/`frames`, close, and set `self.data = None`.
- New `AudioFile.iter_chunks(start_frame, num_frames, chunk_frames)` generator yielding
  `(start, block)` using `sf.SoundFile.seek()` + `read(frames=..., dtype=DEFAULT_DTYPE,
  always_2d=True)`. Reads at any packet-aligned offset, so chunks are independently readable.
- New `AudioFile.write_chunk(block, first)` / `AudioFile.open_writer()` using
  `sf.SoundFile(path, "w", samplerate, channels, subtype, endian, format)`, writing chunks in order.
  `save()` keeps its current behaviour for small in-memory arrays (used by `ListeningTest`).
- `get_data()` keeps returning an `ndarray`. When `self.data is None` (file-backed mode) it lazily
  materialises the full array exactly as before **only for files below the memory budget**; above
  budget it raises `MemoryBudgetExceeded` with a clear message. This preserves the contract for every
  existing caller and for the UI while making whole-file access opt-in-safe.
- Add `get_num_frames()`, `get_total_duration_s()` helpers. `hash` computation for large file-backed
  files uses the same `calculate_hash(self.data.tobytes())` when materialised; when not materialised,
  hash the header + file size + mtime-independent content digest computed by streaming the file in
  fixed blocks, so `OriginalTrackNode` IDs and cached-result invalidation behaviour are preserved
  (same value as today for identical content).

### 2. Chunk planning and memory safety (`utils.py`, `settings.py`, `models.py`)

New module-level helpers in `utils.py`:

- `available_memory_bytes()` using stdlib only (Linux `os.sysconf('SC_AVPHYS_PAGES')`,
  macOS `sysctl`/`resource.getrlimit`, fallback `sys.maxsize`); no new dependency.
- `estimate_peak_bytes(num_frames, channels, dtype_bytes, n_workers_parallel)` → model of the
  largest simultaneous arrays (context + chunk input + chunk output + algorithm scratch), i.e.
  `O(ctx + chunk)`, not `O(num_frames)`.
- `plan_chunk_frames(track_frames, packet_size, context_frames, budget_bytes, channels)` → chunk
  length in **whole packets**, aligned to a multiple of `context_frames` so boundary math is exact.
- `MemoryBudgetExceeded(RuntimeError)` exception, raised with a message of the form:
  `"Cannot process '<track>' with <stage>: estimated peak <X> MB exceeds budget <Y> MB. Increase
  memory_budget_mb or split the track."`

`TestbenchConfiguration` gets a new **optional internal** field `memory_budget_mb: int = None`
(default = 75% of available RAM, per decision). `DataManager` computes the budget once and stores it;
it is **not** added to any node's `Settings`, so hashes, node IDs and DB documents are unaffected. It
is threaded to workers as a non-hashed attribute (same pattern as `progress_monitor`).

### 3. Chunk-driven nodes (`node.py`)

Node-level structure (`OriginalTrackNode → LostSamplesMaskNode → ReconstructedTrackNode →
OutputAnalysisNode`), `_run`, persistence and hashing stay identical. Internally:

- `LostSamplesMaskNode._run`: instead of one full `num_samples` loop, iterate
  `original_track.get_num_frames()` in chunk-sized, packet-aligned blocks and call a new
  `PacketLossSimulator.run_chunk(chunk_packets, global_packet_offset, id) -> np.ndarray` of
  **absolute lost-sample indices**. The simulator's generator state (`npr` stream, metronome
  `counter`, Gilbert-Elliott `current_state`, custom-mask `counter`) carries across chunks naturally
  because chunks are consecutive and the RNG is seeded once — so the produced index array is
  bit-identical to today's. The array is written incrementally to the same `.npy` path via a
  streaming/append-then-concatenate write, keeping the same filename and dtype. `get_data()` still
  returns the full index array (mask memory is ~1/32 of the audio memory and stays within budget;
  the pre-flight accounts for it).
- `ReconstructedTrackNode._run`: opens the destination WAV writer once, then for each chunk builds
  the packet-aligned slice (read from disk, not from a resident full array), calls the algorithm's
  context-aware chunk entry point, and appends the returned block. Only `(context + chunk)` audio
  exists at any time. Final trimmed length equals today's (`original_len` rounded up to a packet,
  then the padding trimmed), verified against the unwritten-length bookkeeping.
- `OutputAnalysisNode._run`: unchanged; metrics receive the same node objects and internally stream.

### 4. PLC algorithms — chunk-aware, output-identical (`plc_algorithm.py`)

`PLCAlgorithm` currently holds all cross-packet state in `self.context` (a ring/roll buffer of
`algorithm_context_length` samples) plus per-subclass state. The refactor makes this explicit:

- Extract the existing per-packet loop body of `run()` into `_run_packets(buffer_stream, is_valid_stream)`
  that advances the algorithm state packet by packet. `run()` becomes a thin wrapper that constructs
  the stream from a materialised array (**used unchanged by `AdvancedPLC`** and by any small input),
  guaranteeing single-pass parity by construction.
- New `run_chunked(audio_source, lost_packets_set, chunk_start, chunk_frames, is_first,
  is_last, id) -> ndarray`:
  1. Rewind the algorithm state to the chunk start. `self.context` is re-seeded by reading the
     **decoded (reconstructed) audio already on disk** for the preceding `algorithm_context_length`
     samples — not the original — exactly reproducing the state a single pass would have had.
     Subclass state that is a pure function of the input prefix (`BurgPLC.previous_valid` and
     `coefficients`, `LowCostConcealment._window`/`_crossfade`, `PARCnetPLC.is_burst`/
     `extra_pred_buffer`, `VermaPLC` context) is reconstructed the same way; all are derived from
     the reference signal and the loss mask, both of which are available file-backed.
  2. Run the packet loop for exactly this chunk's packets, with `is_valid` derived from the global
     lost-packet membership set.
  3. Return the chunk block; the node trims/clips the returned array to the chunk's real length
     (a chunk is always a whole number of packets, so packet indices never straddle chunks and the
     `lost_packets_idx[j]` look-ahead logic is reproduced exactly).
- `AdvancedPLC`: keeps its current `run()` entry point unchanged (it needs whole per-band tracks for
  the Linkwitz–Riley splitter and M/S codec). To stay within budget it wraps the per-channel
  `recursive_split_audio` in an overlap-based streaming crossover: `LinkwitzRileyCrossover` gains
  `split_chunked(source, chunk_frames)` that carries SOS filter state (`sosfilt`'s `zi`) and the
  M/S state across chunks, producing band chunks consumed per band algorithm and summed into one
  output writer. Band-algorithm results are numerically identical to today because the filters are
  fed in the same order with contiguous state.
- `ExternalPLC` reads through the same per-packet interface; no change to its C++ binding contract.

### 5. Streamable metrics (`output_analyser.py`, `utils.py`)

- `SimpleCalculator.run` currently builds full-length `x_rw`/`x_ew` frame stacks. Change it so
  `MSECalculator`/`MAECalculator`/`SpectralEnergyCalculator` iterate the source in window-aligned
  blocks with the **same** `N`/`hop`, Hann window, normalisation and frame boundaries, appending a
  small per-frame result list (`SimpleCalculatorData` still wraps the same full-length error array at
  the end, so `get_error()`, `__len__`, plotting and DB serialisation are unchanged). To keep
  identical framing across the whole track, the block size is a multiple of `hop`, and the first
  block is prefixed by the previous block's tail (`N - hop` samples) so windowed frames are bit-equal.
- `normalise` uses a global peak; compute the peak in a cheap first streaming pass per file, then
  reuse it in the second pass (same two-pass structure as today's indexing, no full array needed).
- Metrics stay signature-compatible: they still receive the `AudioFile` node objects.

### 6. Progress reporting and notification

- `PacketLossSimulator` and `PLCAlgorithm` keep their existing `desc=f"{str(self)}|{id}"` strings and
  packet-count totals; an additional outer `tqdm` bar reports chunks
  (`desc=f"{str(self)}|{id} (chunks)"`). Nothing the UI parses changes.
- On `MemoryBudgetExceeded`, `DataManager.run_testbench()` catches it, sets the run status
  (new `RunStatus.OUT_OF_MEMORY` value added to the enum in `models.py`; existing values untouched),
  prints a clear single-line message, and returns without corrupting partial results. Partially
  written files from aborted chunks are removed so the cache stays consistent.

## Important changes to behavior, public APIs, interfaces, types

- **Public API:** unchanged. `PLCTestbench(...)`, `TestbenchConfiguration`, all `*Settings`
  constructors, `Worker.run(...)` signatures, node classes, `get_data()` return types, file names and
  extensions, node IDs and hashes, and DB document shapes (`_id`, `filepath`, `file_hash`, `parent`,
  `persistent`) are all preserved.
- **New internal symbols:** `MemoryBudgetExceeded`, `available_memory_bytes`, `estimate_peak_bytes`,
  `plan_chunk_frames`, `AudioFile.iter_chunks`/`open_writer`/`write_chunk`/`get_num_frames`,
  `PacketLossSimulator.run_chunk`, `PLCAlgorithm.run_chunked`, `LinkwitzRileyCrossover.split_chunked`,
  and `TestbenchConfiguration.memory_budget_mb`. None are required by callers.
- **New enum member:** `RunStatus.OUT_OF_MEMORY` (additive only).
- **Lazy-loading side effect:** `AudioFile.data` is `None` until materialised; `get_data()` still
  returns a full `ndarray` for files within budget, and raises `MemoryBudgetExceeded` beyond it. Code
  paths that must not load whole files use `iter_chunks` instead.

## Test cases and verification scenarios

New `test/test_chunked_processing.py` (pytest), plus a fixtures helper generating a short synthetic
stereo WAV (deterministic, e.g. ~30 s @ 48 kHz, sine + noise):

1. **Golden-file equality (primary).** For each built-in PLC (`Zeros`, `LastPacket`, `Burg`,
   `LowCost`, `Verma` if TF importable, `PARCnet`, `Advanced`) and each streamable metric (`MSE`,
   `MAE`, `SpectralEnergy`), compare the chunked run against a stored golden output produced by the
   pre-change code path (`np.testing.assert_allclose(rtol=0, atol=0)` for PLC WAVs; exact equality
   for metric arrays).
2. **Chunk-size invariance.** Same run with chunk sizes forced to 1×, 2× and 4× the minimum viable
   chunk yields byte-identical reconstructed WAVs and identical metric arrays.
3. **PLCMOS/PESQ/PEAQ unaffected.** Assert their outputs on the small fixture equal the golden values
   (they still run the untouched whole-file path).
4. **Boundary continuity.** A loss burst placed exactly across a chunk boundary reconstructs the same
   samples as the single-pass run (dedicated regression for the `lost_packets_idx` look-ahead).
5. **Mask identity.** `LostSamplesMaskNode` `.npy` contents are bit-identical to the golden mask for
   `Binomial`, `Metronome`, `GilbertElliot` and `CustomMask` (RNG/counter state carried across
   chunks), and `get_data()` returns the same array.
6. **Memory pre-flight behaviour.** With `memory_budget_mb` forced below the estimated peak for a
   non-streamable metric, `run()` raises/aborts with a message naming the track, stage and MB values,
   leaves no partial/corrupt node files, and sets `RunStatus.OUT_OF_MEMORY`; with a sufficient budget
   the same run completes successfully.
7. **End-to-end.** `PLCTestbench(...).run()` on the synthetic file completes, persists nodes, and a
   second identical run is served from cache without recomputation (hash stability), then
   `plot(...)` and `PlotManager` produce the same outputs.
8. **API smoke.** A test asserting `AudioFile.get_data()` returns an `ndarray` of the right shape,
   `get_samplerate()/get_channels()` are populated without `data`, and that no worker signature
   changed (introspection check on the documented `run` signatures).

Manual acceptance: a 20-minute stereo WAV run with streamable metrics completes under the memory
budget while resident RSS stays roughly flat across chunks (observed via `/usr/bin/time -l` or
`resource.getrusage`), and a run selecting PEAQ/PESQ on that file aborts early with the clear message
instead of being OOM-killed.

## Assumptions and defaults

- Memory budget default = **75% of currently available RAM**, overridable via
  `TestbenchConfiguration.memory_budget_mb`.
- Chunk size is **auto-derived** (aligned to whole packets and the algorithm context length) and is
  **excluded from hashes/settings**, so node IDs and cached results are stable.
- The lost-samples mask keeps its **current `.npy` file and full-length integer index format**,
  built via chunk-accumulated decisions.
- Progress bars keep packet-based totals plus an extra outer chunk bar; UI-parsed descriptions are
  unchanged.
- The 25%-of-RAM headroom and the pre-flight estimate are deliberately conservative; an over-estimate
  results in an early abort (safe) rather than an OOM (unsafe), and an under-estimate is caught by the
  runtime `MemoryError` backstop that converts it into the same user-visible abort.
- `ExternalPLC` and the PLCMOS/PESQ/PEAQ metrics retain their current whole-file interfaces; they are
  protected by the budget check rather than streamed, since streaming them would change their numeric
  output and violate the "output unchanged" requirement.
- No new runtime dependency is introduced (`psutil` is avoided; memory probing uses the standard
  library).
