# pixtreme.infer.cda_vsr

The pure Python distribution `pixtreme-infer-cda-vsr` provides the import package
`pixtreme.infer.cda_vsr`: the CDA-VSR architecture module (causal recurrent
online video super-resolution) for the pixtreme inference companion. The module
reads the scale from a caller-supplied checkpoint's safetensors header, and the
caller selects the precision. The family contract does not fix a scale.

The wheel owns only the `pixtreme/infer/cda_vsr/` subtree. The namespace root
`pixtreme/__init__.py` is owned by [pixtreme](https://github.com/sync-dev-org/pixtreme)
and `pixtreme/infer/__init__.py` by
[pixtreme-infer](https://github.com/sync-dev-org/pixtreme-infer).

## Installation

Use Linux x86_64 with CPython 3.12 or later and an NVIDIA GPU and driver compatible
with CUDA 13. Install the distribution from PyPI:

```console
pip install pixtreme-infer-cda-vsr
```

For a project managed with uv:

```console
uv add pixtreme-infer-cda-vsr
```

## Status

Implemented:

- The public module names `FACTORY`, `FAMILY_CONTRACT`, and `build_engine`. `FACTORY.query_contract` describes
  an exact input shape and the accepted checkpoint structure without allocating device memory, and `FACTORY.open` returns a
  stateful instance called as `instance(frame, mv=mv, residual=residual, reset=False)`.
- Execution of separate I and P TensorRT engines with private recurrent state on the GPU.
  The instance selects I for the first and reset requests and P for later requests.
- `FACTORY.graph_factory()` selects CUDA Graph inference with the same checkpoint,
  precision, input shape, parameters, and output contract.
- The `mv` and `residual` parameter declarations, whose `unit` begins with the side-data profile
  identifier `profile=hevc-mv-residual/4`.
- `build_engine` prepares exact-shape I and P engines in `fp32` or `fp16` in the
  [tensoracle](https://github.com/sync-dev-org/tensoracle) engine cache and returns the checkpoint's
  `TemporalOpenContract`.

GPU execution has been verified on sm86. Other GPU generations have not been verified.

## CUDA Graph inference

Select the graph factory before opening an instance. Its family and query contracts
match `FACTORY`, and selecting or querying it performs no device allocation. Open
the instance inside an explicit non-default CuPy stream and keep that device and
stream for every request:

```python
import cupy as cp
from pixtreme.infer.cda_vsr import FACTORY

factory = FACTORY.graph_factory()
checkpoint = "/path/to/checkpoints/best-inference.safetensors"
contract = factory.query_contract(checkpoint, input_hw=(270, 480), precision="fp16")
with cp.cuda.Device(0), cp.cuda.Stream(non_blocking=True) as stream:
    instance = factory.open(checkpoint, input_hw=(270, 480), precision="fp16")
    try:
        # Prepare inputs on this stream, then submit each request:
        # output = instance(frame, mv=mv, residual=residual, reset=False)
        pass
    finally:
        stream.synchronize()
        instance.close()
```

Open prepares four graphs: I and P for each direction of the two state buffers.
Each graph owns a dedicated execution: two I executions and two P executions.
Preparation restores initial history before returning; the first request and
every explicit reset use I, and
other requests use P. Reset clears the existing state allocations. There is no
input-based reset detection.

Every request validates Frame metadata, parameters, reset, device, and stream
before modifying storage. The module copies RGB, motion vectors, and residuals
into fixed GPU buffers, replays the selected graph, and returns an independent
Frame with that request's metadata. Keep input storage alive until its copies
complete and establish CUDA dependencies before consuming outputs on another
stream. Later requests, resets, and close do not change returned Frames.

One instance accepts one input shape and precision on its opening device and
stream. Open a new instance to change those settings. Preparation errors fail
open and release partially prepared resources; there is no automatic fallback or
recapture. The normal factory retains its existing stream and request behavior.

## Prepare a checkpoint

The module accepts checkpoints in the safetensors format only. Use the checkpoint
producer's export to create a safetensors file with the header metadata
`cda_vsr.residual_unit=luma-range`.

Converting a `.pth` file with `tensoracle.convert` only changes the file format;
the result lacks the required residual-unit metadata and is rejected by this
module. Adding that metadata by hand does not replace the producer's export.
The producer's export must also apply any training-specific adjustments to the
weights. Manually converting a checkpoint and adding metadata does not perform
those adjustments and can change inference output.

The checkpoint producer chooses the weights for inference. Supply exactly one
weight set, under any common prefix or with no prefix; a file containing both
`params` and `params_ema` is rejected. Preserve all producer-written metadata if
copying an exported weight set to a separate safetensors file.

Inference and engine building do not import PyTorch.

## Open a checkpoint

Pass the safetensors checkpoint to `FACTORY`. tensoracle opens compatible I and P engines
from its cache, builds each missing engine, and manages the engine files. The cache follows
`XDG_CACHE_HOME` (otherwise `~/.cache`) under `tensoracle`. The graph version includes
the selected precision, I/P route, and residual gate, so each combination occupies a separate cache entry.
I/P selection happens in the instance, without a conditional inside either engine or an
`is_first` engine input. The graph's tensor types determine computation precision; all engines keep their
inputs, outputs and recurrent state in float32.

```python
from pixtreme.infer.cda_vsr import FACTORY

checkpoint = "/path/to/checkpoints/best-inference.safetensors"
contract = FACTORY.query_contract(checkpoint, input_hw=(270, 480), precision="fp16")
instance = FACTORY.open(checkpoint, input_hw=(270, 480), precision="fp16")
# output = instance(frame, mv=mv, residual=residual, reset=False)
# Synchronize outstanding work on the request stream before closing.
instance.close()
```

Checkpoint paths accept strings or `os.PathLike[str]` objects. The format is checked
from the file contents, regardless of filename extension. PyTorch checkpoints
(`torch.save` ZIP and legacy pickle) and engine files are rejected with instructions
to use the checkpoint producer's export described above. `query_contract` checks
that the safetensors header is readable; it never imports PyTorch or TensorRT,
allocates device memory, builds an engine, or writes
to the cache. It identifies weight sets by the common prefix before the documented
tensor names; names without a prefix form an unnamed set. Even a partially present
set counts. If there are multiple sets or none, query and open report the discovered
names and request a checkpoint containing exactly one inference set, before reading
tensor values or creating a cached engine.

The safetensors header metadata (`__metadata__`) must contain
`cda_vsr.residual_unit=luma-range`, recording the residual unit used during training.
`query_contract`, `open`, and `build_engine` reject absent metadata, a missing key,
or any other value with the observed value (or its absence) and the accepted value
`luma-range`, before reading tensor values or creating an engine cache. Preserve
the producer's metadata when copying an exported weight set.
Checkpoints without this record are rejected regardless of scale.

The same metadata selects the residual gate. When `cda_vsr.gate` is `sigmoid`,
the network uses `aligned * sigmoid(w)`, where `w` is
the output of the three `conv_weight_l` convolutions and their intervening
LeakyReLU activations. When the gate key is absent, the network uses
`aligned * (1 + w)`. Any other value
is rejected by `query_contract`, `open`, and `build_engine` with the observed value
and instructions to use `sigmoid` or omit the key, before reading tensor values or
creating an engine cache. The checkpoint producer records the gate used during
training. Both gate forms are computed in float32, including in fp16 engines.

Query reads the shape of `upconv1.weight` within that sole set from the header and
derives scale from its output channel count, `3 * scale**2`. The tensor must have
four dimensions and scale must be a positive integer; a missing tensor or an
incompatible shape is rejected with the observed and expected values. The
persistent state footprint follows the architecture and requested dimensions.
Query does not read tensor values.

A cache miss validates all 179 tensor names, shapes and float32 dtypes in the sole
set before saving an engine in the cache. Extra tensor names under that prefix
are rejected; for an unnamed set, every tensor belongs to the set. Unrelated
tensors outside a named set are ignored. The reconstruction head and base image resize use
the checkpoint's scale; the remaining network structure uses 64 features, 3 feature
extraction blocks, 24 I reconstruction blocks, 12 P reconstruction blocks and 4 DCN
groups. An incompatible checkpoint may therefore be queried successfully and fail
when opened. Each engine accepts the exact requested `(H, W)`
and precision (`"fp32"` or `"fp16"`). Unsupported precision and dimensions are
rejected before the checkpoint is read.

## Prepare the engine cache

Installing this module installs tensoracle and the TensorRT runtime it uses. Building
an engine on a cache miss and opening a cached engine both need a compatible NVIDIA
GPU; neither imports PyTorch. A checkpoint that fails validation does not leave an
engine in the cache.

To complete the initial build before processing frames, use the same open/close path:

```python
from pixtreme.infer.cda_vsr import build_engine

contract = build_engine(
    "/path/to/checkpoints/best-inference.safetensors",
    input_hw=(270, 480),
    precision="fp16",
)
```

`build_engine` accepts no destination and writes no caller-managed engine file.
tensoracle owns the two cached engine files. The returned contract equals
`FACTORY.query_contract` with the same checkpoint, dimensions and precision.
Subsequent `FACTORY.open` calls reuse both prepared engines. Keep checkpoints outside
the repository; weights and engines are not included in the distribution.

## Frame, state and stream requirements

Call instances with an RGB float32 Frame and C-contiguous CUDA float32 arrays
`mv=(2,H,W)` and `residual=(1,H,W)` on the instance's device. Motion vectors use
channel 0=x, 1=y and `reference = current + mv` in LR luma pixels. Residual values are
signed luma residuals divided by the stream's luma black-to-white range, following
`profile=hevc-mv-residual/4`: `219 * 2^(bit depth - 8)` for limited range and
`255 * 2^(bit depth - 8)` for full range. Streams without a range indication use
limited range. Values, units, matching picture identity,
producer completion, model reference structure, and absence of concurrent calls to
one instance are caller preconditions. The caller decides when to reset.

Make inputs ready on the current CuPy stream and retain their allocations until that
stream has finished reading them. Use the same stream for every request to an instance.
The first request may use a different stream from open; the module orders initial
state readiness. Establish dependencies before consuming returned Frames on other
streams, and synchronize outstanding work before close. Later requests, reset and
close do not change previously returned Frames. Every accepted request returns an
output, including first and reset requests. The module keeps its own persistent
state and does not retain caller buffers across calls. It allocates two state buffers
once at open and exchanges their input/output roles after each request, without copying
history between them. Reset clears the buffers without allocating them again. Calls
do not synchronize the host between requests.

## Dependencies

- Runtime dependencies are `pixtreme-infer>=0.14.1`, `safetensors>=0.4`,
  `tensoracle>=1.1.0`, and the TensorRT runtime `tensorrt-cu13>=11,<12` that tensoracle uses.
  The module does not rely on core extras.
- Development and tests use the published `pixtreme-infer` distribution from PyPI.
  The `dev` dependency group installs `tensoracle[convert]`, whose PyTorch the tests
  use to convert the public checkpoint and to compute reference values.

## Development

Run development commands from the root of a development checkout:

```console
uv sync --group dev
uv lock --check
uv run python -m pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv build --wheel
```

## Versioning and release

The version is derived by hatch-vcs from tags of the form `vX.Y.Z`
at the repository root; tags with other prefixes do not affect it.

Each version is distributed on PyPI as `pixtreme-infer-cda-vsr`. The same wheel and
its `SHA256SUMS` file are also available in GitHub Releases. Declare dependencies
on the PyPI distribution with a minimum version (`pixtreme-infer-cda-vsr>=X.Y.Z`).
Use PyPI as the dependency source; GitHub Release assets and Git references are
not dependency sources.

## License

This distribution is licensed under the [MIT License](LICENSE).
